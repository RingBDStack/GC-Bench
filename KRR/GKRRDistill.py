import os
import sys
import copy
import argparse
import time
from tqdm import tqdm

import torch
import torch_geometric.transforms as T
from torch_geometric import seed_everything
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj, add_self_loops
from ogb.graphproppred import Evaluator
from ogb.graphproppred import PygGraphPropPredDataset

from gntk import *
from models import *
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size_T', type=int, default=128,
                    help='Batch size of the original dataset.')
parser.add_argument('--batch_size_S', type=int, default=16,
                    help='Batch size of the synthetic dataset.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to update the synthetic graphs.')
parser.add_argument('--updateA', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--updateX', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--lrA', type=float, default=1e-6,
                    help='Initial learning rate for A, adjacency matrix.')
parser.add_argument('--lrX', type=float, default=1e-6,
                    help='Initial learning rate for X, feature matrix.')
parser.add_argument('--dataset', type=str, default='PROTEINS',
                    choices=['PROTEINS', 'NCI1', 'DD', 'NCI109',
                    'ogbg-molbbbp', 'ogbg-molbace', 'ogbg-molhiv'],
                    help="The dataset to be used.")
parser.add_argument("--dataset_dir", type=str, default="data", help="Data directory")
parser.add_argument('--device', type=str, default='1',
                    choices=['cpu', '0', '1', '2', '3'],
                    help="The GPU device to be used.")
parser.add_argument('--gpc', type=int, default=10,
                    help='Number of graphs per class to be synthetized.')
parser.add_argument('--scale', type=str, default='uniform',
                    choices=['uniform', 'degree'],
                    help="The normalization method of GNTK")
parser.add_argument('--initialize', type=str, default='herding',
                    choices=['herding', 'kcenter', 'random'],
                    help="The initialization of the synthetic graphs")
parser.add_argument('--anm', type=int, default=2,
                    help='The size of the synthetic graphs (average_nodes_multiplier).')
parser.add_argument('--orth_reg', type=float, default=1e-3,
                    help='the regularization parameter of the orthogonal_loss.')
parser.add_argument('--reg_lambda', type=float, default=1e-6,
                    help='the lambda hyperparameter of the KRR.')
parser.add_argument('--tau', type=float, default=1,
                    help='The starting temperature parameter for an annealing schedule.')
parser.add_argument('--test_gap', type=int, default=1,
                    help='How many updates to have a test.')

parser.add_argument('--test_gnn_depth', type=int, default=5,
                    help='The depth of the test GNN.')
parser.add_argument('--hidden_dim', type=int, default=64,
                    help='The hidden dim of the test GNN.')
parser.add_argument('--test_wd', type=float, default=5e-4,
                    help='The weight decay of the test GNN.')


args = parser.parse_args()

if args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:' + args.device

seed = 1234
seed_everything(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

""" folder for downloaded datasets """
if not os.path.isdir('dataset'):
    os.makedirs('dataset')
""" folder for distilled graphs """
if not os.path.isdir('synthetic_graphs'):
    os.makedirs('synthetic_graphs')

gntk = LiteNTK(num_layers=5, num_mlp_layers=1, scale=args.scale, reg_lambda=args.reg_lambda).to(device)

def discretize(adj, train=True, epoch=0):
    if train:
        """ Gumbel sigmoid sampling """
        num_graphs, num_nodes = adj.shape[0], adj.shape[1]
        delta = torch.zeros_like(adj).to(device)
        vals = torch.rand(num_graphs * num_nodes * (num_nodes+1) // 2).to(device)
        vals = vals.view(num_graphs, -1)
        i, j = torch.triu_indices(num_nodes, num_nodes)
        delta[:, i, j] = vals
        delta.transpose(1,2)[:, i, j] = vals
        adj = torch.log(delta) - torch.log(1-delta) + adj

        """ an annealing schedule for the temperature """
        temp_start = args.tau
        temp_end = 0.01
        t = max(temp_start*(temp_end/temp_start)**(epoch/200), temp_end)

        adj = torch.sigmoid(adj/t)

    else:
        """ Using the threshold 0.5 to discretize the adjacency matrix """
        adj = torch.sigmoid(adj)
        adj[adj> 0.5] = 1
        adj[adj<= 0.5] = 0
    return adj

def gntk_distill(A_S, X_S, y_S, A_T, X_T, y_T, epoch):
    A_S_discrete = discretize(A_S, train=True, epoch=epoch)
    pred, K_SS = gntk(A_S_discrete, X_S, y_S, A_T, X_T)

    """ Here the distillation loss can be cross-entropy or mse loss """
    # mse_loss = 0.5*torch.mean((pred - y_T) ** 2)
    mse_loss = F.nll_loss(F.log_softmax(pred, dim=-1), y_T.max(1)[1])

    """ Normalize the K_SS gram matrix """
    diag = torch.sqrt(K_SS.diag())
    tmp = diag[:, None] * diag[None, :]
    K_SS = K_SS / tmp

    orthogonal_loss = args.orth_reg * torch.norm(K_SS-torch.eye(K_SS.shape[0]).to(device))
    final_loss = mse_loss + orthogonal_loss

    optimizer_A_S.zero_grad()
    optimizer_X_S.zero_grad()
    mse_loss = mse_loss.detach().item()
    orthogonal_loss = orthogonal_loss.detach().item()

    final_loss.backward()

    if args.updateA:
        optimizer_A_S.step()
    if args.updateX:
        optimizer_X_S.step()
    A_S.data.clamp_(min=0, max=1)
    X_S.data.clamp_(min=0, max=1)

    return mse_loss, orthogonal_loss

def test(A_list, X_list, y_list, pooling, nclass):
    """ In the test phase, the distilled dataset will be used to train a GNN and verify its performance """
    A_list = discretize(A_list, train=False)
    y_list = torch.argmax(y_list, dim=1)
    if name in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace']:
        nclass = 1

    """ Construct data which can be used by the DataLoader """
    sampled = np.ndarray((A_list.size(0),), dtype=object)
    for i in range(A_list.size(0)):
        x = X_list[i]
        g = A_list[i].nonzero().T
        y = y_list[i]
        sampled[i] = (Data(x=x, edge_index=g, y=y))
    syn_training_set = SparseTensorDataset(sampled)
    train_loader = DataLoader(syn_training_set, batch_size=128, shuffle=True, num_workers=0)

    best_tests = []
    for run in range(test_runs):
        model = G_GIN(input_dim=nfeat, hidden_dim=hidden_dim, output_dim=nclass,
                    nconvs=gnn_depth, dropout=dropout, pooling=pooling).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        best_val_performance = 0
        best_test_performance = 0
        for epoch in range(nepochs):
            for data in train_loader:
                data = data.to(device)
                loss = train(model, optimizer, data, training_loss, last_activation)
            if epoch % 30 == 29:
                val_performance = evaluator(model, val_set, metric)
                test_performance = evaluator(model, test_set, metric)
                if val_performance > best_val_performance:
                    best_val_performance = val_performance
                    best_test_performance = test_performance
            scheduler.step()
        best_tests.append(test_performance)
    avg_test = np.array(best_tests).mean(axis=0)
    std_test = np.array(best_tests).std(axis=0)
    return avg_test, std_test

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

@torch.no_grad()
def evaluator_acc(model, dataset, metric=None):
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
def evaluator_ogb(model, dataset, metric):
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    outputs = []
    labels = []
    for data in loader:
        data = data.to(device)
        outputs.append(model(data.edge_index, data.x, data.batch))
        labels.append(data.y)
    outputs = torch.cat(outputs)
    labels = torch.cat(labels)
    result = metric.eval({'y_pred': outputs, 'y_true': labels})['rocauc']
    return result

if __name__ == '__main__':

    """ Distillation setting """
    name = args.dataset
    graph_per_class = args.gpc
    lr_A = args.lrA
    lr_X = args.lrX
    num_epoch = args.epochs
    batch_size_T = args.batch_size_T
    batch_size_S = args.batch_size_S
    initialize = args.initialize # random, herding, kcenter

    """  Test setting  """
    hidden_dim = args.hidden_dim
    gnn_depth = args.test_gnn_depth
    dropout = 0.0
    poolings = ['sum','mean']
    lr = 0.001
    weight_decay = args.test_wd
    nepochs = 200
    test_runs = 3

    print("Device: {}".format(device))
    print("GNTK scale: {}".format(args.scale))
    print("Dataset Name: {}".format(name))
    print("Graph per Class: {}".format(graph_per_class))
    print("Batch Size of T, S: {}, {}".format(batch_size_T, batch_size_S))
    print("Update A: {}, Update X: {}".format(args.updateA, args.updateX))
    print("lrA: {}, lrX: {}".format(args.lrA, args.lrX))
    print("Initialization: {}".format(args.initialize))
    print("anm: {}".format(args.anm))
    print("Orthogonal Reg: {}".format(args.orth_reg))
    print("reg_lambda: {}".format(args.reg_lambda))
    print("tau: {}".format(args.tau))
    print("Test GNN depth: {}".format(args.test_gnn_depth))
    print("Test GNN dimension: {}".format(args.hidden_dim))
    print("Test GNN weight decay: {}".format(args.test_wd))
    print()

    """ Loading dataset with different metrics """
    if name in ['PROTEINS', 'NCI1', 'DD', 'NCI109']: # TUDataset
        evaluator = evaluator_acc
        metric = None
        training_loss = F.nll_loss
        last_activation = "softmax"

        dataset = TUDataset(root='dataset', name=name, use_node_attr=True, transform=T.Compose([T.NormalizeFeatures()]))
        
        """ The original split is randomly generated for the TU datasets """
        split_file = "splits/OriginalSplit/"+name+"_split.txt"
        if not os.path.exists(split_file):
            print("{} original split does not exist!".format(name))
            sys.exit()
        splits = load_split(split_file)

        training_set = dataset[splits[0]]
        val_set = dataset[splits[1]]
        test_set = dataset[splits[2]]
        nclass = dataset.num_classes
        nfeat = dataset.num_node_features

    elif name in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace']: # ogb
        evaluator = evaluator_ogb
        metric = Evaluator(name)
        training_loss = F.binary_cross_entropy_with_logits
        last_activation = "sigmoid"

        dataset = PygGraphPropPredDataset(root='dataset', name=name, transform=T.Compose([RemoveEdgeAttr(), T.NormalizeFeatures()]))
        
        """ The original split is provided by the OGB datasets """
        splits = dataset.get_idx_split()
        training_set = dataset[splits["train"]]
        val_set = dataset[splits["valid"]]
        test_set = dataset[splits["test"]]
        nclass = dataset.num_classes
        nfeat = dataset.num_node_features

    else:
        print("No dataset exists".format(name))
        sys.exit()

    """ Start Distillation """

    """ Initialize synthetic graphs, which are of the same size """
    """ The adjacency matrices and node feature matrices are stored as tensor with size (# graphs, # nodes, # nodes/features) """
    n_S = graph_per_class * nclass
    n_avg_node = avg_num_node(name) * args.anm
    A_S = torch.zeros(size=(n_S, n_avg_node, n_avg_node), dtype=torch.float, requires_grad=True, device=device)
    X_S = torch.zeros(size=(n_S, n_avg_node, nfeat), dtype=torch.float, requires_grad=True, device=device)

    """ Initialize the syntehtic graphs with truncated real-world graphs """
    if initialize == "herding":
        file_name = "splits/herdingSelect/"+name+"_herding_"+str(graph_per_class)+".txt"
    elif initialize == "kcenter":
        file_name = "splits/kcenterSelect/"+name+"_kcenter_"+str(graph_per_class)+".txt"
    elif initialize == "random":
        file_name = "splits/randomSelect/"+name+"_random_"+str(graph_per_class)+".txt"
    with open(file_name, 'r', encoding='utf-8') as fin:
        selected_idx = [int(x) for x in fin.read().strip().split(' ')]
    
    class2selected_training_idx = {}
    for i in selected_idx:
        y = copy.deepcopy(training_set[i].y)
        if y.item() not in class2selected_training_idx:
            class2selected_training_idx[y.item()] = [i]
        else:
            class2selected_training_idx[y.item()].append(i)
    start_idx = 0
    y_S = []
    largest_real_graph_size = 0
    for cla in range(nclass):
        if initialize == "herding" or initialize == "kcenter" or initialize == "random":
            assert len(class2selected_training_idx[cla]) == graph_per_class
            sampled =[training_set[i] for i in class2selected_training_idx[cla]]
        data = Batch.from_data_list(sampled)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x, mask = to_dense_batch(x, batch=batch, max_num_nodes=None)
        adj = to_dense_adj(edge_index, batch=batch, max_num_nodes=None)

        """ Here the entries of adj is used for Gumbel-Max sampling so set them as large numbers """
        adj[adj> 0.5] = 5
        adj[adj<= 0.5] = -5
        min_num_node = min(n_avg_node, x.shape[1])
        largest_real_graph_size = max(largest_real_graph_size, x.shape[1])

        """ min_num_node is used to truncate the real graph if it is too large """
        X_S.data[start_idx:start_idx+graph_per_class, :min_num_node] = x[:, :min_num_node].detach().data
        A_S.data[start_idx:start_idx+graph_per_class, :min_num_node, :min_num_node] = adj[:, :min_num_node, :min_num_node].detach().data

        """ The synthetic dataset is balanceed (same # samples per class) """
        y_S += [cla] * graph_per_class
        start_idx += graph_per_class
    assert len(y_S) == graph_per_class * nclass

    """ Change the label to the one-hot fashion """
    y_S = torch.nn.functional.one_hot(torch.LongTensor(y_S)).float().to(device)

    print("Synthetic graph size: {}".format(n_avg_node))
    print("Largest Real graph size: {}".format(largest_real_graph_size))
    print()

    optimizer_A_S = torch.optim.Adam([A_S], lr=lr_A)
    optimizer_X_S = torch.optim.Adam([X_S], lr=lr_X)

    T_loader = DataLoader(training_set, batch_size=batch_size_T, shuffle=True, num_workers=0)
    cnt = 0
    for _ in range(num_epoch):

        """ Mini batch both the T and S dataset """
        for data in T_loader:
            data = data.to(device)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            batch_X_T, batch_mask = to_dense_batch(x, batch=batch, max_num_nodes=None)
            batch_A_T = to_dense_adj(edge_index, batch=batch, edge_attr=None, max_num_nodes=None)
            batch_y_T = torch.nn.functional.one_hot(data.y.squeeze()).float()

            selected_idx = random.sample(range(n_S), min(batch_size_S, n_S))
            batch_A_S, batch_X_S, batch_y_S = A_S[selected_idx], X_S[selected_idx], y_S[selected_idx]
            cla_loss, orthogonal_loss = gntk_distill(batch_A_S, batch_X_S, batch_y_S,
                                    batch_A_T, batch_X_T, batch_y_T, cnt)

            """ Report results every args.test_gap epochs """
            if cnt % args.test_gap == args.test_gap-1:
                save_name = "synthetic_graphs/"+name+"_"+str(graph_per_class)+"_"+args.scale+"_epoch_"+str(cnt)+"_lrA_"+str(args.lrA)+"_lrX_"+str(args.lrX)+".pt"
                torch.save([A_S.detach().cpu(), X_S.detach().cpu(), y_S.detach().cpu()], save_name)
                print("Epoch {} Performance".format(cnt))
                performance = test(A_S.detach(), X_S.detach(), y_S.detach(), "mean", nclass)
                print("Mean Pooling -- Test Mean: {:.3f}, Training Std: {:.3f}".format(performance[0], performance[1]))
                performance = test(A_S.detach(), X_S.detach(), y_S.detach(), "sum", nclass)
                print("Sum Pooling -- Test Mean: {:.3f}, Training Std: {:.3f}".format(performance[0], performance[1]))
                print()

            cnt += 1
            if cnt == num_epoch:
                print("Finish distillation.")
                sys.exit()
