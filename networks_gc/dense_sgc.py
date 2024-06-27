import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.dense.linear import Linear
import os.path as osp
from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import LayerNorm, InstanceNorm

class AKX(torch.nn.Module):
    def __init__(self, K=3):
        super(AKX, self).__init__()
        self.conv = SGConv(1, 1, K=K)
        self.conv.lin = torch.nn.Identity() # TODO

    def forward(self, x, adj, pool):
        x = self.conv(x, adj)
        if pool == 'mean':
            x = x.mean(1)
        if pool == 'sum':
            x = x.sum(1)
        return torch.norm(x).item()

def get_akx(x, adj, K, pool):
    conv = DenseSGConv(1, 1, K=K)
    conv.lin = torch.nn.Identity() # TODO
    x = conv(x, adj)
    if pool == 'mean':
        x = x.mean(1)
    if pool == 'sum':
        x = x.sum(1)
    return torch.norm(x)



class DenseSGC(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, ntrans=1, nconvs=3, dropout=0):
        super(DenseSGC, self).__init__()
        self.conv = DenseSGConv(nfeat, nhid, K=nconvs)

        self.norms = nn.ModuleList([])
        net_norm = 'none'
        for _ in range(ntrans+1):
            if ntrans == 0:  norm = torch.nn.Identity()
            elif net_norm == 'none':
                norm = torch.nn.Identity()
            elif net_norm == 'batchnorm':
                norm = BatchNorm1d(nhid)
            elif net_norm == 'layernorm':
                norm = nn.LayerNorm([nhid,111], elementwise_affine=True)
            elif net_norm == 'instancenorm':
                norm = InstanceNorm(nhid, affine=False) #pyg
            elif net_norm == 'groupnorm':
                norm = nn.GroupNorm(4, nhid, affine=True)
            self.norms.append(norm)

        self.lins = nn.ModuleList([])
        self.ntrans = ntrans
        for _ in range(ntrans):
            self.lins.append(torch.nn.Linear(nhid, nhid))
        self.lin_final = torch.nn.Linear(nhid, nclass) if ntrans == 0 else  lambda x:x
        self.dropout = dropout

        self.dropout = dropout

    def forward(self, x, adj, mask=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv(x, adj, mask)
        x = self.perform_norm(0, x)
        x = F.relu(x)
        for ii, lin in enumerate(self.lins):
            x = lin(x)
            x = self.perform_norm(ii+1, x)
            x = F.relu(x)

        x = x.mean(1)
        x = F.log_softmax(self.lin_final(x), dim=-1)
        return x

    def perform_norm(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = self.norms[i](x)
        # x = self.norm(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x


class DenseSGConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K=2, improved=False, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.K = K
        self.lin = Linear(in_channels, out_channels, bias=bias,
                          weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        for _ in range(self.K):
            x = torch.matmul(adj, x)

        out = self.lin(x)

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
