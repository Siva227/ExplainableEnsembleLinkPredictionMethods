from sklearn.base import BaseEstimator, TransformerMixin


class GraphScorer(BaseEstimator, TransformerMixin):
    def __init__(self, input_network):
        self.input_network = input_network
