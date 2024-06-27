import sys
import math
import torch
import numpy as np
import torch.nn.functional as F
from typing import Callable, Optional, Union
from torch_geometric.typing import Adj, Tensor, SparseTensor, OptPairTensor, OptTensor, Size
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch_geometric.nn.conv import MessagePassing, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import add_remaining_self_loops

class G_GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nconvs=3, dropout=0, pooling='mean', **kwargs):
        super().__init__()

        self.convs = torch.nn.ModuleList([])
        self.convs.append(GINConv(torch.nn.Linear(input_dim, hidden_dim), train_eps=True))
        
        for _ in range(nconvs-1):
            self.convs.append(GINConv(torch.nn.Linear(hidden_dim, hidden_dim), train_eps=True))
        self.project = Linear(hidden_dim, output_dim)
        self.norms = torch.nn.ModuleList([])
        for _ in range(nconvs):
            if nconvs == 1:
                norm = torch.nn.Identity()
            else:
                norm = torch.nn.BatchNorm1d(hidden_dim)
            self.norms.append(norm)

        self.dropout = dropout
        self.pooling = pooling

    def forward(self, edge_index, x, batch, edge_weight=None, output_embed=False):

        for i in range(len(self.convs)-1):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.norms[i](x)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index, edge_weight)

        if self.pooling == 'mean':
            x = global_mean_pool(x, batch=batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch=batch)

        if not output_embed: x = self.project(x)
        return x