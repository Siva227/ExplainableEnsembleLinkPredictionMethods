import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


# TODO: Add check fitted
# TODO: Add ClassNamePrefix ClassNamePrefixFeaturesOutMixin
class GraphScorer(BaseEstimator, TransformerMixin):
    def __init__(self, input_network):
        self.input_network = input_network

    def make_dataset(self, X):
        X = check_array(X, accept_large_sparse=False, estimator=self)
        if X.shape[1] == 1:
            return pd.DataFrame(X, columns=["node_i"])
        elif X.shape[1] == 2:
            return pd.DataFrame(X, columns=["node_i", "node_j"])
        else:
            raise ValueError("Bad input shape")


class GlobalGraphScorer(GraphScorer):
    def __init__(self, input_network):
        super(GlobalGraphScorer, self).__init__(input_network)
        self.num_nodes = None
        self.num_edges = None
        self.average_degree = None
        self.degree_variance = None
        self.network_diameter = None
        self.degree_assortativity = None
        self.network_transitivity = None
        self.avg_clustering_coefficient = None

    def fit(self, X, y=None):
        self.num_nodes = nx.number_of_nodes(self.input_network)
        self.num_edges = nx.number_of_edges(self.input_network)
        self.network_diameter = nx.diameter(self.input_network)
        self.degree_assortativity = nx.degree_assortativity_coefficient(self.input_network)
        self.network_transitivity = nx.transitivity(self.input_network)
        self.avg_clustering_coefficient = nx.average_clustering(self.input_network)
        degrees = self.input_network.degree()
        self.average_degree = np.mean([degrees[idx] for idx in range(self.num_nodes)])
        self.degree_variance = np.var([degrees[idx] for idx in range(self.num_nodes)])
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        num_rows = X.shape[0]
        output_vector = np.array(
            [
                self.num_nodes,
                self.num_edges,
                self.average_degree,
                self.degree_variance,
                self.network_diameter,
                self.network_transitivity,
                self.degree_assortativity,
                self.avg_clustering_coefficient,
            ]
        )
        output_arr = np.tile(output_vector, (num_rows, 1))
        return output_arr
