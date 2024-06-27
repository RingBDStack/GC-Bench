import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
import torch
import torch.nn.functional as F
from GM.agent_graph import GraphAgent
from torch_geometric.data import Data, Batch
import argparse
import random
import numpy as np
from utils import *
from utils.utils_graphset import *
from utils.utils_graphset import Dataset as GraphSetDataset
from networks_gc.gin import G_GIN
from ogb.graphproppred import Evaluator

device = 'cuda'

def discretize(adj):
    adj[adj> 0.5] = 1
    adj[adj<= 0.5] = 0
    return adj


@torch.no_grad()
def evaluator_acc(model, dataset, args, metric=None):
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    outputs = []
    labels = []
    for data in loader:
        data = data.to(device)
        out = model(data.edge_index, data.x, data.batch)
        outputs += out.max(1)[1].tolist()
        labels += data.y.tolist()
    correct = torch.FloatTensor(outputs).eq(torch.FloatTensor(labels)).double()
    correct = correct.sum()
    acc = correct / len(labels)
    return acc.cpu().item()

@torch.no_grad()
def evaluator_ogb(model, dataset, args, metric):
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    outputs = []
    labels = []
    for data in loader:
        data = data.to(device)
        outputs.append(model(data.edge_index, data.x, data.batch))
        labels.append(data.y.unsqueeze(1))
    outputs = torch.cat(outputs)
    labels = torch.cat(labels)
    result = metric.eval({'y_pred': outputs, 'y_true': labels})['rocauc']
    return result

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='ogbg-molhiv')
parser.add_argument('--dataset_dir', type=str, default='data')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--nlayers', type=int, default=5)
parser.add_argument('--init', type=str, default='noise')
parser.add_argument('--init_way', type=str, default='Random')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--reduction_rate', type=float, default=0.1)
parser.add_argument('--stru_discrete', type=int, default=1)
parser.add_argument('--nconvs', type=int, default=3)
parser.add_argument('--outer', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--ipc', type=int, default=1)
parser.add_argument('--nruns', type=int, default=10)
parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
parser.add_argument('--num_blocks', type=int, default=1)
parser.add_argument('--num_bases', type=int, default=0)
parser.add_argument('--model', type=str, default='GIN')
parser.add_argument('--pooling', type=str, default='mean')
parser.add_argument('--net_norm', type=str, default='none')
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--method', type=str, default='random')
parser.add_argument('--save_dir', type=str, default='save')

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

torch.set_num_threads(1)

print(args)

data = Dataset(args)
packed_data = data.packed_data

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.dataset == 'ogbg-molhiv':
    args.pooling = 'sum'

if args.dataset == 'CIFAR10':
    args.nruns = 3
    args.net_norm = 'instancenorm'

agent = GraphAgent(data=packed_data, args=args, device=device, nnodes_syn=get_mean_nodes(args))

if args.method in ['herding', 'kcenter', 'random']:
    # coreset0.00000
    indexes = np.load(f'{args.save_dir}/{args.method}/idx_{args.dataset}_{args.ipc}.npy')
    selected_data = packed_data[4][indexes]
else:
    if args.method == "DosCond":
        adj_syn = torch.load(f'{args.save_dir}/{args.method}/adj_{args.dataset}_{args.ipc}_{args.seed}.pt', map_location='cuda')
        feat_syn = torch.load(f'{args.save_dir}/{args.method}/feat_{args.dataset}_{args.ipc}_{args.seed}.pt', map_location='cuda')
        labels_syn = agent.labels_syn
    elif args.method == "KiDD":
        [A_S, feat_syn, y_S] = torch.load(f'{args.save_dir}/{args.method}/{args.dataset}_{args.ipc}_{args.seed}.pt', map_location='cuda')
        adj_syn = discretize(A_S)
        labels_syn = torch.argmax(y_S, dim=1)

    selected_data = np.ndarray((adj_syn.size(0),), dtype=object)
    for i in range(adj_syn.size(0)):
        x = feat_syn[i]
        g = adj_syn[i].nonzero().T
        y = labels_syn[i]
        selected_data[i] = (Data(x=x, edge_index=g, y=y))

if args.model == "GIN":
    print("==========Testing GIN===========")
    def train(model, optimizer, data, training_loss=F.nll_loss, last_activation="softmax"):
        model.train()
        optimizer.zero_grad()
        out = model(data.edge_index, data.x, data.batch, data.edge_weight)
        if last_activation == "softmax":
            out = F.log_softmax(out, dim=-1)
            loss = training_loss(out, data.y)
        elif last_activation == 'sigmoid':
            loss = training_loss(out, data.y.float().view(-1, 1))
        loss.backward()
        optimizer.step()

    if args.dataset in ['PROTEINS', 'NCI1', 'DD', 'NCI109']: # TUDataset
        evaluator = evaluator_acc
        metric = None
        training_loss = F.nll_loss
        last_activation = "softmax"

    elif args.dataset in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace']:
        evaluator = evaluator_ogb
        metric = Evaluator(args.dataset)
        training_loss = F.binary_cross_entropy_with_logits
        last_activation = "sigmoid"

    dataset = GraphSetDataset(args)
    training_set = dataset.train_dataset
    val_set = dataset.val_dataset
    test_set = dataset.test_dataset

    syn_training_set = SparseTensorDataset(selected_data)
    train_loader = DataLoader(syn_training_set, batch_size=128, shuffle=True, num_workers=0)

    best_tests = []
    nepochs = 200
    for run in range(args.nruns):
        if args.dataset in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace']:
            nclass = 1
        else:
            nclass = packed_data[0].num_classes
        model = G_GIN(input_dim=packed_data[0].num_features, hidden_dim=args.hidden, output_dim=nclass,
                        nconvs=args.nlayers, dropout=args.dropout, pooling=args.pooling).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        best_val_performance = 0
        best_test_performance = 0
        for epoch in range(nepochs):
            for data in train_loader:
                data = data.to(device)
                loss = train(model, optimizer, data, training_loss, last_activation)
            if epoch % 30 == 29:
                val_performance = evaluator(model, val_set, args, metric)
                test_performance = evaluator(model, test_set, args, metric)
                if val_performance > best_val_performance:
                    best_val_performance = val_performance
                    best_test_performance = test_performance
            scheduler.step()
        best_tests.append(test_performance)
    avg_test = np.array(best_tests).mean(axis=0)
    std_test = np.array(best_tests).std(axis=0)
    print(f"Mean TestAcc:{avg_test} Std TestAcc:{std_test}")

elif args.model == "GCN":
    print("==========Testing GCN===========")
    res = []
    # Test GCN
    for _ in range(args.nruns):
        if args.dataset in ['ogbg-molhiv']:
            res.append(agent.test_pyg_data(syn_data=selected_data, epochs=100))
        else:
            res.append(agent.test_pyg_data(syn_data=selected_data, epochs=500))
    res = np.array(res)
    print('Mean Train/Val/TestAcc:', res.mean(0))
    print('Std Train/Val/TestAcc:', res.std(0))
