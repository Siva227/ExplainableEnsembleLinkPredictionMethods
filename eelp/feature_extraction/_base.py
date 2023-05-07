import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_X_y


# TODO: Add check fitted
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
