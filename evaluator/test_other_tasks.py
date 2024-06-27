import torch
import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import separate
from pygod.detector import DOMINANT
from pygod.metric import eval_roc_auc
from pygod.generator import gen_contextual_outlier, gen_structural_outlier
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_networkx
import scipy.sparse as sp
import copy
import argparse
from utils.utils import *
from utils.utils_graph import DataGraph
import sklearn 
import sklearn.cluster
import random
from math import ceil
import copy
from _utils import get_syn_data
import networkx as nx
import matplotlib.pyplot as plt

def visualize_pyg_graph(data, ya, title):
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G)
    
    labels = data.y
    ya = ya
    
    # Assign colors to nodes based on their labels
    color_map = plt.cm.get_cmap('viridis', len(np.unique(labels)))
    node_colors = [color_map(label) for label in labels]
    
    # Highlight anomalies in red
    for idx, is_anomaly in enumerate(ya):
        if is_anomaly == 1:
            node_colors[idx] = 'red'
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=700, font_size=10)
    plt.title(title)
    plt.savefig(title)
    plt.show()

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) 
    r_inv = np.power(rowsum, -1).flatten()  
    r_inv[np.isinf(r_inv)] = 0.  
    r_mat_inv = sp.diags(r_inv) 
    mx = r_mat_inv.dot(mx)  
    return mx

class GraphData:

    def __init__(self, features, adj, labels, idx_train=None, idx_val=None, idx_test=None):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

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
            adj_selfloop = dpr_data.adj.cuda() + torch.eye(dpr_data.adj.shape[0]).cuda()
            edge_index_selfloop = adj_selfloop.nonzero().T
            edge_index = edge_index_selfloop
            edge_weight = adj_selfloop[edge_index_selfloop[0], edge_index_selfloop[1]]
        else:
            adj_selfloop = dpr_data.adj + sp.eye(dpr_data.adj.shape[0])
            edge_index = torch.LongTensor(adj_selfloop.nonzero()).cuda()
            edge_weight = torch.FloatTensor(adj_selfloop[adj_selfloop.nonzero()]).cuda()

        # by default, the features in pyg data is dense
        try:
            x = torch.FloatTensor(dpr_data.features.cpu().numpy()).float().cuda()
        except:
            x = torch.FloatTensor(dpr_data.features).float().cuda()
        try:
            y = torch.LongTensor(dpr_data.labels.cpu()).cuda()
        except:
            y = dpr_data.labels


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

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x 

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1
        )  # product of a pair of nodes on each edge

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    
def generate_labels_syn(args, data):
    from collections import Counter
    counter = Counter(data.labels_train)
    num_class_dict = {}
    n = len(data.labels_train)
    sorted_counter = sorted(counter.items(), key=lambda x:x[1])
    sum_ = 0
    labels_syn = []
    for ix, (c, num) in enumerate(sorted_counter):
        if ix == len(sorted_counter) - 1:
            num_class_dict[c] = int(n * args.reduction_rate) - sum_
            labels_syn += [c] * num_class_dict[c]
        else:
            num_class_dict[c] = max(int(num * args.reduction_rate), 1)
            sum_ += num_class_dict[c]
            labels_syn += [c] * num_class_dict[c]

    return labels_syn

def link_pred(nfeat, train_data, val_data, test_data, nruns):
        
    def train_link_predictor(
        model, train_data, val_data, optimizer, criterion, n_epochs=100, verbose=training_scheduler
    ):

        for epoch in range(1, n_epochs + 1):

            model.train()
            optimizer.zero_grad()
            z = model.encode(train_data.x, train_data.edge_index)

            # sampling training negatives for every training epoch
            neg_edge_index = negative_sampling(
                edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
                num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

            edge_label_index = torch.cat(
                [train_data.edge_label_index, neg_edge_index],
                dim=-1,
            )
            edge_label = torch.cat([
                train_data.edge_label,
                train_data.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)

            out = model.decode(z, edge_label_index).view(-1)
            loss = criterion(out, edge_label)
            loss.backward()
            optimizer.step()

            val_f1 = eval_link_predictor(model, val_data)

            if epoch % 10 == 0 and verbose:
                print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val F1: {val_f1:.3f}")

        return model

    @torch.no_grad()
    def eval_link_predictor(model, data):

        model.eval()
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        pred = out.cpu().numpy()
        pred = (pred > 0.5).astype(np.int32)
        return accuracy_score(data.edge_label.cpu().numpy(), pred)
        # return roc_auc_score(data.edge_label.cpu().numpy(),pred)

    res = []
    for _ in range(nruns):
        model = Net(nfeat, 128, 64).to(args.device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        model = train_link_predictor(model, train_data, val_data, optimizer, criterion)
        test_auc = eval_link_predictor(model, test_data)
        res.append(test_auc)
    print(f"Test mean: {np.array(res).mean(axis=0):.2f}, std: {np.array(res).std(axis=0):.2f}")

def anomaly_inject(data, nnodes, anomaly_rate, anomaly_type, save=False,save_dir=None):
    # inject anomalies into training data
    min_anomalies = 1
    max_anomalies = min(20, nnodes // 2)
    n = max(min_anomalies, min(ceil(nnodes * anomaly_rate), max_anomalies))
    k = max(5, min(50, nnodes // 2))
    m = max(2, min(10, nnodes // 5))
    if anomaly_type == 'context':
        data, ya = gen_contextual_outlier(data.cpu(), n=n, k=k)
    elif anomaly_type == 'structure':
        data, ya = gen_structural_outlier(data.cpu(), m=m, n=n)
    data.y = ya.long()
    if save:
        torch.save(data,f'{save_dir}/{anomaly_type}_{anomaly_rate}.pt')
    return data, ya

def anomaly_detect(train_data, train_nnodes,nruns, name, anomaly_rate=0.05, anomaly_type="context"):
    test_data = torch.load(f'testset_othertasks/{name}/{anomaly_type}_{anomaly_rate}.pt')
    train_data, ya = anomaly_inject(train_data, train_nnodes, anomaly_rate=anomaly_rate, anomaly_type=anomaly_type)
    train_data.y = ya.long()
    def train_anomaly_detector(model, graph):
        return model.fit(graph)

    def eval_anomaly_detector(model, graph):

        pred, score, prob, conf = model.predict(graph,
                                           return_pred=True,
                                           return_score=True,
                                           return_prob=True,
                                           return_conf=True)
        if isinstance(graph.y, torch.Tensor):
            labels = graph.y.cpu().numpy()
        else:
            labels = graph.y
        auc = eval_roc_auc(labels, score)
        # ap = average_precision_score(graph.y.cpu().numpy(), outlier_scores)
        # print(f'AUC Score: {auc:.3f}')
        # print(f'AP Score: {ap:.3f}')
        return auc
    
    res = []
    for _ in range(nruns):
        model = DOMINANT()
        model = train_anomaly_detector(model, train_data)
        res.append(eval_anomaly_detector(model, test_data))
    print(f"{anomaly_type} Auc mean: {np.array(res).mean(axis=0):.2f}, std: {np.array(res).std(axis=0):.2f}")

def generate_test_anomalies(data,save_dir):
    feat_test = data.feat_test
    adj_test = data.adj_test
    label_test = data.labels_test
    dataset = GraphData(feat_test, adj_test, label_test)
    graph = Dpr2Pyg(dataset)[0]
    anomaly_inject(graph,feat_test.shape[0],0.05,'structure',True,save_dir=save_dir)
    anomaly_inject(graph,feat_test.shape[0],0.05,'context',True,save_dir=save_dir)


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--method", type=str, default="Full", help="Method")
parser.add_argument("--task", type=str, default="LP", help="Task for evaluation")
parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
parser.add_argument("--transductive", type=int, default=1)
parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
parser.add_argument("--save_dir", type=str, default="save", help="synthetic dataset directory")
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--keep_ratio', type=float, default=1)
parser.add_argument('--reduction_rate', type=float, default=1.0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--epsilon', type=float, default=0.05)
parser.add_argument('--nruns', type=int, default=2)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
device = 'cuda'

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

data_pyg = ["cora", "citeseer", "pubmed", 'cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel']
if args.dataset in data_pyg:
    data_full = get_dataset(args.dataset, dataset_dir=args.data_dir)
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
else:
    if args.transductive:
        data = DataGraph(args.dataset, data_dir=args.data_dir)
    else:
        data = DataGraph(args.dataset, label_rate=args.label_rate, data_dir=args.data_dir)
    data_full = data.data_full


if args.task == "AD_gen":
    save_dir = f'testset_othertasks/{args.dataset}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    generate_test_anomalies(data,save_dir=save_dir)

elif args.task == "LP":
    syn_split = T.RandomLinkSplit(
        num_val=0.2,
        num_test=0.0,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
        )
    if args.method == 'Full':
        if args.transductive:
            feat_syn, adj_syn, labels_syn = data.feat_full, data.adj_full, data.labels_full
            adj_syn = sp.csr_matrix(adj_syn.shape)
            nnodes = feat_syn.shape[0]
            adj_syn[:nnodes,:nnodes] = sp.coo_matrix(data.adj_full[:nnodes,:nnodes])
        else:
            feat_syn, adj_syn, labels_syn = data.feat_train, data.adj_train, data.labels_train
    else:
        feat_syn, adj_syn, labels_syn = get_syn_data(args)
    dataset = GraphData(feat_syn, adj_syn, labels_syn)
    graph_syn = Dpr2Pyg(dataset)[0]
    train_data, val_syn, _ = syn_split(graph_syn)
    print("==========LP=========")
    feat_test = data.feat_test
    adj_test = data.adj_test 
    label_test = data.labels_test
    dataset = GraphData(feat_test, adj_test, label_test)
    graph = Dpr2Pyg(dataset)[0]
    test_data = graph
    neg_edge_index = negative_sampling(test_data.edge_index, num_nodes=test_data.x.shape[0], num_neg_samples=test_data.edge_index.size(1))

    edge_label_index = torch.cat([test_data.edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat([torch.ones(test_data.edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0)
    test_data = Data(x=test_data.x, y=test_data.y, edge_index=test_data.edge_index, edge_label_index=edge_label_index, edge_label=edge_label, num_nodes=test_data.x.shape[0])

    # split = T.RandomLinkSplit(
    #     num_val=0.0,
    #     num_test=1.0,
    #     is_undirected=True,
    #     add_negative_train_samples=False,
    #     neg_sampling_ratio=1.0,
    #     )
    # _, _, test_data = split(graph)
    link_pred(feat_syn.shape[1], train_data, val_syn, test_data, args.nruns)

elif args.task == "AD":
    feat_syn, adj_syn, labels_syn = get_syn_data(args)
    dataset = GraphData(feat_syn, adj_syn, labels_syn)
    graph_syn = Dpr2Pyg(dataset)[0]
    print("==========AD=========")
    syn_nnodes = feat_syn.shape[0]
    anomaly_detect(train_data=copy.deepcopy(graph_syn), train_nnodes=syn_nnodes,  nruns=args.nruns, name=args.dataset,  anomaly_type='context')
    anomaly_detect(train_data=copy.deepcopy(graph_syn), train_nnodes=syn_nnodes, nruns=args.nruns,  name=args.dataset, anomaly_type='structure')