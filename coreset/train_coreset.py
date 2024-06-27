from deeprobust.graph.data import Dataset
import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
import numpy as np
import random
import time
import argparse
import torch
from deeprobust.graph.utils import accuracy
import torch.nn.functional as F
from configs import load_config
from utils.utils import *
from utils.utils_graph import DataGraph
from utils.utils_graphset import Dataset as GraphSetDataset
from networks_nc.gcn import GCN
from networks_gc.gcn import GCN as GCN_GC
from coreset import KCenter, Herding, Random
from torch_geometric.data import Batch
from tqdm import tqdm

class GraphSetData():
    def __init__(self, grpah_list, embeds):
        self.feat_train = embeds
        self.labels_train = np.array([graph.y.item() for graph in grpah_list])
        self.idx_train = np.arange(embeds.shape[0])

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--dataset_dir',type=str,default='data')
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--nlayers', type=int, default=2, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--inductive', type=int, default=0)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--method', type=str, choices=['kcenter', 'herding', 'random'],default='kcenter')
parser.add_argument('--reduction_rate', type=float, default=0.01)
parser.add_argument('--gpc', type=int, default=10)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
args = load_config(args)
# print(args)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

directory_path = f'save/{args.method}'
os.makedirs(directory_path, exist_ok=True)

"""
Graph Classification Dataset
"""
device = 'cuda'
ipc = False
if args.dataset in ['PROTEINS', 'NCI1', 'DD', 'NCI109',
                        'ogbg-molbbbp', 'ogbg-molbace', 'ogbg-molhiv',
                        'MNIST', 'CIFAR10']:
    ipc = True
    dataset = GraphSetDataset(args)
    training_set = dataset.train_dataset
    val_set = dataset.val_dataset
    test_set = dataset.test_dataset
    nclass = dataset.nclass
    nfeat = dataset.nfeat
    labels = dataset.labels
    data_train = Batch.from_data_list(dataset.packed_data[4]).to(device)
    model = GCN_GC(nfeat=nfeat, nhid=args.hidden, nclass=nclass, args=args).to(device)
    embeds = model.embed(data_train)
    data = GraphSetData(dataset.packed_data[4], embeds)
    args.reduction_rate = args.gpc / embeds.shape[0]
else:
    data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
    if args.dataset in data_graphsaint:
        data = DataGraph(args.dataset)
        data_full = data.data_full
        data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
    else:
        data_full = get_dataset(args.dataset, args.normalize_features,args.dataset_dir)
        data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)

    features = data_full.features
    adj = data_full.adj
    labels = data_full.labels
    idx_train = data_full.idx_train
    idx_val = data_full.idx_val
    idx_test = data_full.idx_test

    # Setup GCN Model
    model = GCN(nfeat=features.shape[1], nhid=256, nclass=labels.max()+1, device=device, weight_decay=args.weight_decay)

    model = model.to(device)
    model.fit(features, adj, labels, idx_train, idx_val, train_iters=600, verbose=False)

    model.eval()
    # You can use the inner function of model to test
    print("-------full------")
    model.test(idx_test)

    embeds = model.predict().detach()

if args.method == 'kcenter':
    agent = KCenter(data, args, ipc=ipc, device='cuda')
if args.method == 'herding':
    agent = Herding(data, args, ipc=ipc, device='cuda')
if args.method == 'random':
    agent = Random(data, args, ipc=ipc, device='cuda')

idx_selected = agent.select(embeds)


if args.dataset in ['PROTEINS', 'NCI1', 'DD', 'NCI109',
                        'ogbg-molbbbp', 'ogbg-molbace', 'ogbg-molhiv',
                        'MNIST', 'CIFAR10']:
    if args.save:
        np.save(f'save/{args.method}/idx_{args.dataset}_{args.gpc}.npy', idx_selected)
    # test in evaluator
    pass
else:
    if args.save:
        np.save(f'save/{args.method}/idx_{args.dataset}_{args.reduction_rate}_{args.seed}.npy', idx_selected)
    feat_train = features[idx_selected]
    adj_train = adj[np.ix_(idx_selected, idx_selected)]
    labels_train = labels[idx_selected]
    res = []
    runs = 10
    for _ in tqdm(range(runs)):
        model.initialize()
        model.fit_with_val(feat_train, adj_train, labels_train, data,
                    train_iters=600, normalize=True, verbose=False)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        # Full graph
        output = model.predict(data.feat_full, data.adj_full)
        loss_test = F.nll_loss(output[data.idx_test], labels_test)
        acc_test = accuracy(output[data.idx_test], labels_test)
        res.append(acc_test.item())
    res = np.array(res)
    print('Mean accuracy:', repr([res.mean(), res.std()]))