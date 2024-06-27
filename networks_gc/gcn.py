import os.path as osp
from math import ceil
import torch
from torch_geometric.nn import DenseSAGEConv, DenseGCNConv, GCNConv
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import LayerNorm, InstanceNorm
from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder




class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nconvs=3, dropout=0, if_mlp=False, net_norm='none', pooling='mean', learn_adj=False, **kwargs):
        super(GCN, self).__init__()
        self.molhiv = False
        if kwargs['args'].dataset in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace']:
            nclass = 1
            self.molhiv = True

        if nconvs ==1:
            nhid = nclass

        self.mlp = if_mlp
        if self.mlp:
            GCNConv = nn.Linear
        else:
            from torch_geometric.nn import GCNConv
        self.convs = nn.ModuleList([])
        # self.convs.append(GCNConv(nfeat, nhid, learn_adj=learn_adj))
        self.convs.append(GCNConv(nfeat, nhid))
        for _ in range(nconvs-1):
            # self.convs.append(GCNConv(nhid, nhid, learn_adj=learn_adj))
            self.convs.append(GCNConv(nhid, nhid))

        self.norms = nn.ModuleList([])
        for _ in range(nconvs):
            if nconvs == 1:  norm = torch.nn.Identity()
            elif net_norm == 'none':
                norm = torch.nn.Identity()
            elif net_norm == 'batchnorm':
                norm = BatchNorm1d(nhid)
            elif net_norm == 'layernorm':
                norm = nn.LayerNorm([nhid], elementwise_affine=True)
            elif net_norm == 'instancenorm':
                norm = InstanceNorm(nhid, affine=False) #pyg
            elif net_norm == 'groupnorm':
                norm = nn.GroupNorm(4, nhid, affine=True)
            self.norms.append(norm)

        self.lin3 = torch.nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.pooling = pooling

    def forward(self, data, if_embed=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.dropout !=0 and self.training:
            x_mask = torch.distributions.bernoulli.Bernoulli(self.dropout).sample([x.size(0)]).to('cuda').unsqueeze(-1)
            x = x_mask * x

        for i in range(len(self.convs)):
            if self.mlp:
                x = self.convs[i](x) #, edge_index)
            else:
                x = self.convs[i](x, edge_index)
            x = self.perform_norm(i, x)
            x = F.relu(x)

        if self.pooling == 'mean':
            x = global_mean_pool(x, batch=data.batch)
        if self.pooling == 'sum':
            x = global_add_pool(x, batch=data.batch)
        if if_embed:
            return x
        if self.molhiv:
            x = self.lin3(x)
        else:
            x = F.log_softmax(self.lin3(x), dim=-1)
        return x

    def forward_edgeweight(self, data, if_embed=False):
        x, edge_index, edge_weight, batch = data

        for i in range(len(self.convs)):
            if self.mlp:
                x = self.convs[i](x) #, edge_index)
            else:
                x = self.convs[i](x, edge_index, edge_weight)
            x = self.perform_norm(i, x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        if self.pooling == 'mean':
            x = global_mean_pool(x, batch=batch)
        if self.pooling == 'sum':
            x = global_add_pool(x, batch=batch)
        if if_embed:
            return x
        if self.molhiv:
            x = self.lin3(x)
        else:
            x = F.log_softmax(self.lin3(x), dim=-1)
        return x


    def embed(self, data):
        return self.forward(data, if_embed=True)


    def perform_norm(self, i, x):
        batch_size, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = self.norms[i](x)
        x = x.view(batch_size, num_channels)
        return x

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin3.reset_parameters()
        