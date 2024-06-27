import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks_nc.gcn import GraphConvolution

class GCN_nc(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GCN_nc, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GraphConvolution(num_features, hidden_size)
        self.conv2 = GraphConvolution(hidden_size, embedding_size)

    def forward(self, x, adj):
        h = self.conv1(x, adj)
        h = self.conv2(h, adj)
        z = F.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred
