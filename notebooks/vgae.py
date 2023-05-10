import os
import torch
from torch_goemetric.datasets import Plenetoid
from torch_geometric.transforms import transforms as T
import torch.nn as nn
from torch_geometric.nn import VGAE
from torch_geometric.transforms import RandomLinkSplit
import torch.nn.functional as F
import numpy as np
import args



# Defining global helper functions
def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)

# Defining the model
class VGAE(nn.Module):
    # Defining the constructor
    def __init__(self, adj):
        super(VGAE, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)

    # Defining the encoder
    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)  # Gaussian noise
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean  # Reparameterization trick
        return sampled_z
    
    # Defining the forward layer
    def forward(self, X):
        z = self.encode(X)
        return z
    
class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)  # Weight initialization
        self.adj = adj  # Adjacency matrix
        self.activation = activation  # Activation function
    
    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)  # Matrix multiplication
        x = torch.mm(self.adj, x)  # Convolution operation
        outputs = self.activation(x)  # Activation
        return outputs
    
class GAE(nn.Module):
    def __init__(self, adj):
        super(GAE, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        z = self.mean = self.gcn.mean(hidden)
        return z
    
    def forward(self, X):
        z = self.encode(X)
        A_pred = dot_product_decode(z)
        return A_pred
    
    
    
    




