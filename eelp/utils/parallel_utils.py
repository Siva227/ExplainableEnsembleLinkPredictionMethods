import logging

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion

from ..feature_extraction import (
    AdamicAdarScorer,
    AvgNeighborDegreeScorer,
    BetweennessCentralityScorer,
    ClosenessCentralityScorer,
    CommonNeighborsScorer,
    DegreeCentralityScorer,
    EigenvectorCentralityScorer,
    GlobalGraphPropertiesScorer,
    InfomapScorer,
    JaccardScorer,
    KatzCentralityScorer,
    LHNScorer,
    LoadCentralityScorer,
    LocalClusteringCoefficientScorer,
    LouvainScorer,
    MDLScorer,
    NumTrianglesScorer,
    PageRankScorer,
    PersonalizedPageRankScorer,
    PreferentialAttachmentScorer,
    ShortestPathScorer,
)
from ..graph_sampling import GraphSampler

GLOBAL_DICT = {"global": GlobalGraphPropertiesScorer}
TOPOL_EDGE_DICT = {
    "aa": AdamicAdarScorer,
    "cn": CommonNeighborsScorer,
    "jaccard": JaccardScorer,
    "lhn": LHNScorer,
    "ppr": PersonalizedPageRankScorer,
    "pa": PreferentialAttachmentScorer,
    "sp": ShortestPathScorer,
}
TOPOL_NODE_DICT = {
    "avg_neighbor_deg": AvgNeighborDegreeScorer,
    "bet_cen": BetweennessCentralityScorer,
    "close_cen": ClosenessCentralityScorer,
    "deg_cen": DegreeCentralityScorer,
    "eig_cen": EigenvectorCentralityScorer,
    "katz_cen": KatzCentralityScorer,
    "load_cen": LoadCentralityScorer,
    "lcc": LocalClusteringCoefficientScorer,
    "num_triangles": NumTrianglesScorer,
    "page_rank": PageRankScorer,
}
MODEL_DICT = {
    "infomap": InfomapScorer,
    "louvain": LouvainScorer,
    "mdl_sbm": MDLScorer,
    "mdl_dcsbm": MDLScorer,
}


def chunk(l, n):
    # loop over the list in n-sized chunks
    for i in range(0, len(l), n):
        # yield the current n-sized chunk to the calling function
        yield l[i : i + n]


def sample_input_graph(num_nodes, edge_list, num_samples=10000, sampling_method="rs"):
    G = create_graph_from_edges(num_nodes, edge_list)
    sampler = GraphSampler(G, sampling_method=sampling_method)
    tr_sample, ho_sample = sampler.sample(num_samples)
    return G, sampler, tr_sample, ho_sample


def create_graph_from_edges(num_nodes, edge_list):
    G = nx.Graph()
    G.add_nodes_from(list(range(num_nodes)))
    G.add_edges_from(edge_list)
    return G


def create_features(input_network, transformer_dict):
    features = FeatureUnion(
        [
            (k, v(input_network)) if k != "mdl_dcsbm" else (k, v(input_network, deg_corr=True))
            for k, v in transformer_dict.items()
        ]
    )
    return features


def create_pipeline(input_network, original_network):
    # global_features = create_features(original_network, GLOBAL_DICT)
    node_features = create_features(input_network, TOPOL_NODE_DICT)
    edge_features = create_features(input_network, TOPOL_EDGE_DICT)
    model_features = create_features(input_network, MODEL_DICT)
    pipe = ColumnTransformer(
        [
            ("global", GlobalGraphPropertiesScorer(original_network), ["node_i", "node_j"]),
            ("node_i", node_features, ["node_i"]),
            ("node_j", node_features, ["node_j"]),
            ("edge_features", edge_features, ["node_i", "node_j"]),
            ("model_features", model_features, ["node_i", "node_j"]),
        ]
    )
    return pipe


def compute_features(feature_pipe, sampled_edges):
    X = sampled_edges.loc[:, ["node_i", "node_j"]].reset_index(drop=True)
    y = sampled_edges.label.values
    X_feat = feature_pipe.fit_transform(X, y)
    feature_names = feature_pipe.get_feature_names_out()
    return feature_pipe, X_feat, y, feature_names


def model_selection(X, y):
    param_grid = {
        "max_depth": [3, 6],
        "n_estimators": [25, 50, 100],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(),
        param_grid=param_grid,
        scoring=["roc_auc", "f1", "precision", "recall"],
        refit="roc_auc",
    )
    grid_search.fit(X, y)
    grid_results = pd.DataFrame(grid_search.cv_results_)
    return grid_search, grid_results


def persist_features(e_sample, X_feat, y, feature_names):
    e_sample = e_sample.loc[:, ["node_i", "node_j"]].reset_index(drop=True)
    feat_df = e_sample.assign(
        **{k: X_feat[:, i] for i, k in enumerate(feature_names)}, label_true=y
    )
    return feat_df


def process_graphs(payload):
    logger = logging.getLogger(__name__)
    logger.debug("Starting process {}".format(payload["id"]))
    out_data = []
    for g_data in payload["input_graphs"]:
        try:
            G, sampler, tr_sample, ho_sample = sample_input_graph(
                g_data["num_nodes"],
                g_data["edge_list"],
                payload["num_samples"],
                payload["sampling_method"],
            )
            training_pipe = create_pipeline(sampler.G_tr, G)
            holdout_pipe = create_pipeline(sampler.G_ho, G)
            training_pipe, X_tr, y_tr, feat_names_tr = compute_features(training_pipe, tr_sample)
            holdout_pipe, X_ho, y_ho, feat_names_ho = compute_features(holdout_pipe, ho_sample)
            process_success = True
        except Exception:
            logger.exception("Feature Generation Failed: {}".format(g_data["network_index"]))
            continue
        try:
            if np.all(feat_names_tr == feat_names_ho):
                process_success = False
            if process_success:
                final_tr_df = persist_features(tr_sample, X_tr, y_tr, feat_names_tr)
                final_ho_df = persist_features(ho_sample, X_ho, y_ho, feat_names_ho)
                grid_search, grid_results = model_selection(
                    final_tr_df.loc[:, feat_names_tr], final_tr_df.label_true.values
                )
                final_ho_df.to_csv(g_data["output_path"] / "holdout_features.csv", index=False)
                final_tr_df.to_csv(g_data["output_path"] / "training_features.csv", index=False)
                grid_results.to_csv(g_data["output_path"] / "grid_search_results.csv", index=False)
                best_model = grid_search.best_estimator_
                feature_importances = best_model.feature_importances_
                ho_pred = best_model.predict(final_ho_df.loc[:, feat_names_tr])
                ho_proba = best_model.predict_proba(final_ho_df.loc[:, feat_names_tr])
        except Exception:
            logger.exception("Model Training Failed: {}".format(g_data["network_index"]))
            continue
            # holdout performance metrics
        try:
            cm = metrics.confusion_matrix(final_ho_df.label_true.values, ho_pred)
            (
                precision_total,
                recall_total,
                f_measure_total,
                _,
            ) = metrics.precision_recall_fscore_support(
                final_ho_df.label_true.values, ho_pred, average=None
            )
            auc_measure = metrics.roc_auc_score(final_ho_df.label_true.values, ho_proba[:, 1])
            output_dict = g_data.copy()
            output_dict["sampling_method"] = payload["sampling_method"]
            output_dict["num_samples"] = payload["num_samples"]
            output_dict["heldout_AUC"] = auc_measure
            output_dict["heldout_predision"] = precision_total
            output_dict["heldout_recall"] = recall_total
            output_dict["heldout_f_measure"] = f_measure_total
            output_dict["feature_importances"] = dict(zip(feat_names_tr, feature_importances))
            output_dict["train_edges"] = nx.to_pandas_edgelist(sampler.G_tr).values
            output_dict["holdout_edges"] = nx.to_pandas_edgelist(sampler.G_ho).values
            output_dict["holdout_confusion_matrix"] = cm
            output_dict["chunk_id"] = payload["id"]
            out_data.append(output_dict)
            # TODO: Save community assignments
            joblib.dump(best_model, g_data["output_path"] / "best_model.joblib")
        except Exception:
            logger.exception("Failed to capture metrics: {}".format(g_data["network_index"]))
            continue

    results_df = pd.DataFrame.from_records(out_data)
    results_df.to_pickle(payload["output_path"])
