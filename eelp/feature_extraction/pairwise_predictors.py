import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer


class GraphEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, input_network):
        self.input_network = input_network


class CommonNeighborsPredictor(GraphEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        common_neighbors = []
        for row in X.itertuples():
            cn = nx.common_neighbors(self.input_network, row.node_i, row.node_j)
            common_neighbors.append(len(list(cn)))
        return np.array(common_neighbors).reshape(-1, 1)


class AdamicAdarPredictor(GraphEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        aa = nx.adamic_adar_index(self.input_network, X.itertuples(index=False, name=None))
        return np.array([i[-1] for i in aa]).reshape(-1, 1)


class ShortestPathPredictor(GraphEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sp = []
        for row in X.itertuples():
            sp.append(nx.shortest_path_length(self.input_network, row.node_i, row.node_j))
        return np.array(sp).reshape(-1, 1)

