import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from torch_geometric.utils import degree
from torch_scatter import scatter
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from utils.utils_graph import DataGraph
from torch_geometric.data import InMemoryDataset, Data
import torch
import scipy.sparse as sp
from torchvision import datasets, transforms
import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import get_train_val_test
from torch_geometric.utils import train_test_split_edges
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from deeprobust.graph.utils import *
from torch_geometric.data import NeighborSampler
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
import math
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
from sklearn.cluster import KMeans
import copy
import argparse
from _utils import get_syn_data

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
parser.add_argument("--method", type=str, default="SGDD", help="Method")
parser.add_argument("--dataset", type=str, default="citeseer")
parser.add_argument("--carch", type=str, default="SGC")
parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
parser.add_argument(
    "--save_dir", type=str, default="save", help="synthetic dataset directory"
)
parser.add_argument("--nlayers", type=int, default=2)
parser.add_argument("--hidden", type=int, default=256)
parser.add_argument("--keep_ratio", type=float, default=1)
parser.add_argument("--reduction_rate", type=float, default=0.5)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--normalize_features", type=bool, default=True)
parser.add_argument("--seed", type=int, default=1, help="Random seed.")
parser.add_argument("--mlp", type=int, default=0)
parser.add_argument("--inner", type=int, default=0)
parser.add_argument("--epsilon", type=float, default=0.01)
parser.add_argument("--nruns", type=int, default=10)
args = parser.parse_args()
args.device = 'cuda:0'

if args.dataset in ['citeseer', 'cora']:
    args.epsilon = 0.05

class Dpr2Pyg(InMemoryDataset):

    def __init__(self, dpr_data, transform=None, **kwargs):
        root = 'data/' # dummy root; does not mean anything
        self.dpr_data = dpr_data
        super(Dpr2Pyg, self).__init__(root, transform)
        pyg_data = self.process()
        self.data, self.slices = self.collate([pyg_data])
        self.transform = transform

    def process____(self):
        dpr_data = self.dpr_data
        try:
            edge_index = torch.LongTensor(dpr_data.adj.nonzero().cpu()).cuda().T
        except:
            edge_index = torch.LongTensor(dpr_data.adj.nonzero()).cuda()
        # by default, the features in pyg data is dense
        try:
            x = torch.FloatTensor(dpr_data.features.cpu()).float().cuda()
        except:
            x = torch.FloatTensor(dpr_data.features).float().cuda()
        try:
            y = torch.LongTensor(dpr_data.labels.cpu()).cuda()
        except:
            y = dpr_data.labels

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = None
        data.val_mask = None
        data.test_mask = None
        return data

    def process(self):
        dpr_data = self.dpr_data
        if type(dpr_data.adj) == torch.Tensor:
            adj_selfloop = dpr_data.adj + torch.eye(dpr_data.adj.shape[0]).cuda()
            edge_index_selfloop = adj_selfloop.nonzero().T
            edge_index = edge_index_selfloop
            edge_weight = adj_selfloop[edge_index_selfloop[0], edge_index_selfloop[1]]
        else:
            adj_selfloop = dpr_data.adj + sp.eye(dpr_data.adj.shape[0])
            edge_index = torch.LongTensor(adj_selfloop.nonzero()).cuda()
            edge_weight = torch.FloatTensor(adj_selfloop[adj_selfloop.nonzero()]).cuda()

        # by default, the features in pyg data is dense
        try:
            x = torch.FloatTensor(dpr_data.features.cpu()).float().cuda()
        except:
            x = torch.FloatTensor(dpr_data.features).float().cuda()
        try:
            y = torch.LongTensor(dpr_data.labels.cpu()).cuda()
        except:
            y = torch.LongTensor(dpr_data.labels).cuda()


        data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)
        data.train_mask = None
        data.val_mask = None
        data.test_mask = None
        return data

    def get(self, idx):
        if self.len() == 1:
            return copy.copy(self._data)
        
        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        self._data_list[idx] = copy.copy(data)

        return data
        

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _download(self):
        pass

class GraphData:

    def __init__(self, features, adj, labels, idx_train=None, idx_val=None, idx_test=None):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

name = "reddit"
path = "data"
if name in ["cora", "citeseer", "pubmed"]:
    dataset = Planetoid(path, name)
elif name in ["ogbn-arxiv"]:
    dataset = PygNodePropPredDataset(name="ogbn-arxiv")
elif name in ["cornell", "texas", "wisconsin"]:
    dataset = WebKB(path, name)
elif name in ["chameleon", "squirrel"]:
    dataset = WikipediaNetwork(path, name)
else:
    dataset = DataGraph(name, data_dir=path)
    feat_full = dataset.feat_full
    adj_full = dataset.adj_full
    label_full = dataset.labels_full
    dataset = GraphData(feat_full, adj_full, label_full)
    dataset = Dpr2Pyg(dataset)[0]


def homophily(edge_index, y):
    degree_cal = degree(edge_index[1], num_nodes=y.size(0))

    edge_homo = (y[edge_index[0]] == y[edge_index[1]]).sum().item() / edge_index.size(1)

    tmp = y[edge_index[0]] == y[edge_index[1]]
    node_homo = (
        scatter(tmp, edge_index[1], dim=0, dim_size=y.size(0), reduce="add")
        / degree_cal
    )

    return edge_homo, node_homo.mean()

def homophily_weight(args):
    feat_syn, adj_syn, labels_syn = get_syn_data(args)
    edge_homo = 0
    for i in range(len(adj_syn)):
        for j in range(len(adj_syn)):
            edge_homo += (labels_syn[i] == labels_syn[j]) * adj_syn[i][j]
    edge_homo /= adj_syn.sum()
    return edge_homo

# print(homophily(dataset.edge_index, dataset.y))

print(args)
print(homophily_weight(args))