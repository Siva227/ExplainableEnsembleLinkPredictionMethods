import graph_tool as gt
import networkx as nx
import numpy as np
from community import community_louvain
from graph_tool.all import BlockState, minimize_blockmodel_dl
from infomap import Infomap

from ..utils import nx2gt
from ._base import GraphScorer


class LouvainScorer(GraphScorer):
    def __init__(
        self,
        input_network: nx.Graph,
        partition=None,
        weight="weight",
        resolution=1.0,
        randomize=None,
        random_state=None,
    ):
        super(LouvainScorer, self).__init__(input_network)
        self.weight = weight
        self.partition = partition
        self.resolution = resolution
        self.randomize = randomize
        self.random_state = random_state
        self.partition = None

    def fit(self, X, y=None):
        self.partition = community_louvain.best_partition(
            self.input_network,
            self.partition,
            self.weight,
            self.resolution,
            self.randomize,
            self.random_state,
        )
        return self

    def transform(self, X):
        score = []
        for e_pair in X.itertuples(name=None, index=False):
            in_copy = self.input_network.copy()
            if e_pair in in_copy.edges:
                in_copy.remove_edge(*e_pair)
            q_wo_edge = community_louvain.modularity(self.partition, in_copy)
            in_copy.add_edge(*e_pair)
            q_w_edge = community_louvain.modularity(self.partition, in_copy)
            score.append(q_w_edge - q_wo_edge)
        return np.array(score).reshape(-1, 1)


class InfomapScorer(GraphScorer):
    def __init__(self, input_network, args=None, two_level=True, num_trials=1):
        super(InfomapScorer, self).__init__(input_network)
        self.im = Infomap(args=args, two_level=two_level, silent=True, num_trials=num_trials)
        self.im.add_networkx_graph(self.input_network)
        self.im_modules = None
        self.im_code_length = None

    def fit(self, X, y=None):
        self.im.run()
        self.im_modules = self.im.get_modules()
        self.im_code_length = self.im.codelength
        return self

    def transform(self, X):
        im_score = []
        for e_pair in X.itertuples(name=None, index=False):
            if e_pair in self.input_network.edges:
                self.im.remove_link(*e_pair)
            self.im.run(initial_partition=self.im_modules, no_infomap=True)
            im_wo_edge = self.im.codelength
            self.im.add_link(*e_pair)
            self.im.run(initial_partition=self.im_modules, no_infomap=True)
            im_w_edge = self.im.codelength
            im_score.append(im_w_edge - im_wo_edge)
            if e_pair not in self.input_network.edges:
                self.im.remove_link(*e_pair)
        return np.array(im_score).reshape(-1, 1)


class MDLScorer(GraphScorer):
    def __init__(self, input_network, deg_corr=False):
        super(MDLScorer, self).__init__(input_network)
        self.gt_in = nx2gt(input_network)
        self.deg_corr = deg_corr
        self.block_state = None

    def fit(self, X, y=None):
        self.block_state: BlockState = minimize_blockmodel_dl(
            self.gt_in, state_args=dict(deg_corr=self.deg_corr)
        )
        return self

    def transform(self, X):
        dl_score = []
        for e_pair in X.itertuples(name=None, index=False):
            in_gt_copy = self.gt_in.copy()
            edge = in_gt_copy.edge(in_gt_copy.vertex(e_pair[0]), in_gt_copy.vertex(e_pair[1]))
            if edge:
                in_gt_copy.remove_edge(edge)
            bs_copy: BlockState = self.block_state.copy(in_gt_copy)
            ent_wo_edge = bs_copy.entropy()
            in_gt_copy.add_edge(in_gt_copy.vertex(e_pair[0]), in_gt_copy.vertex(e_pair[1]))
            bs_copy: BlockState = self.block_state.copy(in_gt_copy)
            ent_w_edge = bs_copy.entropy()
            dl_score.append(ent_w_edge - ent_wo_edge)
        return np.array(dl_score).reshape(-1, 1)
