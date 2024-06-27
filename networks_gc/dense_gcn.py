import os.path as osp
from math import ceil
import torch
from torch_geometric.nn import DenseSAGEConv, DenseGCNConv, GCNConv
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import LayerNorm, InstanceNorm


class DenseGCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nconvs=3, dropout=0, if_mlp=False, net_norm='none', pooling='mean', **kwargs):
        super(DenseGCN, self).__init__()

        self.molhiv = False
        if kwargs['args'].dataset in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace']:
            nclass = 1
            self.molhiv = True

        if nconvs == 1:
            nhid = nclass

        self.mlp = if_mlp
        if self.mlp:
            DenseGCNConv = nn.Linear
        else:
            from torch_geometric.nn import DenseSAGEConv, DenseGCNConv
        self.convs = nn.ModuleList([])
        self.convs.append(DenseGCNConv(nfeat, nhid))
        for _ in range(nconvs-1):
            self.convs.append(DenseGCNConv(nhid, nhid))

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

        self.lin3 = torch.nn.Linear(nhid, nclass) if nconvs != 1 else lambda x: x
        self.dropout = dropout
        self.pooling = pooling

    def forward(self, x, adj, mask=None, if_embed=False):
        if self.dropout !=0:
            x_mask = torch.distributions.bernoulli.Bernoulli(self.dropout).sample([x.size(0), x.size(1)]).to('cuda').unsqueeze(-1)
            x = x_mask * x

        for i in range(len(self.convs)):
            if self.mlp:
                x = self.convs[i](x)
            else:
                x = self.convs[i](x, adj, mask)
            x = self.perform_norm(i, x)
            x = F.relu(x)

        if self.pooling == 'sum':
            x = x.sum(1)
        if self.pooling == 'mean':
            x = x.mean(1)
        if if_embed:
            return x
        if self.molhiv:
            x = self.lin3(x)
        else:
            x = F.log_softmax(self.lin3(x), dim=-1)

        return x


    def embed(self, x, adj, mask=None):
        return self.forward(x, adj, mask, if_embed=True)

    def perform_norm(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = self.norms[i](x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x
