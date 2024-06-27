from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, DenseDataLoader
import os.path as osp
from torch_geometric.datasets import MNISTSuperpixels
import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import random

dataset_max_nnodes = {
    'CIFAR10': 150,
    'DD': 5748,
    'MUTAG': 28,
    'NCI1': 111,
    'ogbg-molhiv': 222
}

class Complete(object):
    def __call__(self, data):
        if data.x is None:
            if hasattr(data, 'adj'):
                data.x = data.adj.sum(1).view(-1, 1)
            else:
                adj = to_scipy_sparse_matrix(data.edge_index).sum(1)
                data.x = torch.FloatTensor(adj.sum(1)).view(-1, 1)
        return data

class RemoveEdgeAttr(object):
    def __call__(self, data):
        if data.edge_attr is not None:
            data.edge_attr = None
        if data.x is None:
            if hasattr(data, 'adj'):
                data.x = data.adj.sum(1).view(-1, 1)
            else:
                adj = to_scipy_sparse_matrix(data.edge_index).sum(1)
                data.x = torch.FloatTensor(adj.sum(1)).view(-1, 1)

        data.y = data.y.squeeze(0)
        data.x = data.x.float()
        return data

class ConcatPos(object):
    def __call__(self, data):
        if data.edge_attr is not None:
            data.edge_attr = None
        data.x = torch.cat([data.x, data.pos], dim=1)
        data.pos = None
        return data

class Dataset:

    def __init__(self, args):
        # random seed setting
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        name = args.dataset
        
        if args.dataset_dir is None:
            path = osp.join(osp.dirname(osp.realpath(__file__)), "data")
        else:
            path = osp.join(args.dataset_dir)

        if name in ['DD', 'MUTAG', 'NCI1']:
            dataset = TUDataset(path, name=name, transform=T.Compose([Complete()]), use_node_attr=True)
            dataset = dataset.shuffle()
            n = (len(dataset) + 9) // 10
            test_dataset = dataset[:n]
            val_dataset = dataset[n:2 * n]
            train_dataset = dataset[2 * n:]
            nnodes = [x.num_nodes for x in dataset]
            print('mean #nodes:', np.mean(nnodes), 'max #nodes:', np.max(nnodes))

        if name in ['CIFAR10']:
            transform = T.Compose([ConcatPos()])
            train_dataset= GNNBenchmarkDataset(path, name=name, split='train', transform=transform)
            val_dataset= GNNBenchmarkDataset(path, name=name, split='val', transform=transform)
            test_dataset= GNNBenchmarkDataset(path, name=name, split='test', transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
            nnodes = [x.num_nodes for x in train_dataset]
            print('mean #nodes:', np.mean(nnodes), 'max #nodes:', np.max(nnodes))

        if name in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace']:
            dataset = PygGraphPropPredDataset(name=name, transform=T.Compose([RemoveEdgeAttr()]))
            split_idx = dataset.get_idx_split()
            train_dataset = dataset[split_idx["train"]]
            nnodes = [x.num_nodes for x in train_dataset]
            if not name in dataset_max_nnodes:
                dataset_max_nnodes[name] = np.max(nnodes)
            print('mean #nodes:', np.mean(nnodes), 'max #nodes:', np.max(nnodes))
            ### automatic evaluator. takes dataset name as input
            train_dataset = dataset[split_idx["train"]]
            val_dataset = dataset[split_idx["valid"]]
            test_dataset = dataset[split_idx["test"]]

        y_final = [g.y.item() for g in test_dataset]
        from collections import Counter; counter=Counter(y_final); print(counter)
        print("#Majority guessing:", sorted(counter.items())[-1][1]/len(y_final))

        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

        train_datalist = np.ndarray((len(train_dataset),), dtype=object)
        for ii in range(len(train_dataset)):
            train_datalist[ii] = train_dataset[ii]
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.nclass = dataset.num_classes
        self.nfeat = dataset.num_node_features
        self.labels = list(set(y_final))
        self.packed_data = [train_dataset, train_loader, val_loader, test_loader, train_datalist]

class TensorDataset(Dataset):
    def __init__(self, feat, adj, labels): # images: n x c x h x w tensor
        self.x = feat.detach()
        self.adj = adj.detach()
        self.y = labels.detach()

    def __getitem__(self, index):
        return self.x[index], self.adj[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

class SparseTensorDataset(Dataset):
    def __init__(self, data): # images: n x c x h x w tensor
        self.data  = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_max_nodes(args):
    return dataset_max_nnodes[args.dataset]

def get_mean_nodes(args):
    if args.dataset == 'CIFAR10':
        return 118
    if args.dataset == 'DD':
        return 285
    if args.dataset == 'MUTAG':
        return 18
    if args.dataset == 'NCI1':
        return 30
    if args.dataset == 'ogbg-molhiv':
        return 26
    if args.dataset == 'ogbg-molbbbp':
        return 24
    if args.dataset == 'ogbg-molbace':
        return 34

    raise NotImplementedError

def save_pyg_graphs(graphs, args):
    memory_dict = {}
    for d in graphs:
        y = d.y.item()
        if y not in memory_dict:
            memory_dict[y] = [d]
        else:
            memory_dict[y].append(d)

    for k, v in memory_dict.items():
        graph_dict = {}
        d, slices = InMemoryDataset.collate(v)
        graph_dict['x'] = d.x
        graph_dict['edge_index'] = d.edge_index
        graph_dict['y'] = d.y
        memory_dict[k] = (graph_dict, slices)

    torch.save(memory_dict, f'saved/memory/{args.dataset}_ours_{args.seed}_ipc{args.ipc}.pt')

def accuracy(output, labels):
	preds = output.max(1)[1].type_as(labels)
	correct = preds.eq(labels).double()
	correct = correct.sum()
	return correct / len(labels)

def accuracy_binary(output, labels):
	output = (output > 0.5).float() * 1
	correct = output.type_as(labels).eq(labels).double()
	correct = correct.sum()
	return correct / len(labels)

def f1(output, labels):
	preds = output.max(1)[1].type_as(labels)
	f1 = f1_score(labels, preds, average='weighted')
	return f1

def create_split(split_file, ngraph, dataset_split):
    ntraining, nvalidation, ntest = [math.floor(ngraph*x) for x in dataset_split]
    all_idx = [str(x) for x in list(range(ngraph))]
    with open(split_file, 'w', encoding='utf-8') as fout:
        random.shuffle(all_idx)
        training_idx = all_idx[:ntraining]
        val_idx = all_idx[ntraining:ntraining+nvalidation]
        test_idx = all_idx[ntraining+nvalidation:]
        fout.write(' '.join(training_idx)+'\t')
        fout.write(' '.join(val_idx)+'\t')
        fout.write(' '.join(test_idx)+'\n')
        training_idx = [int(x) for x in training_idx]
        val_idx = [int(x) for x in val_idx]
        test_idx = [int(x) for x in test_idx]
        split = [training_idx, val_idx, test_idx]
    return split

def load_split(split_file):
    with open(split_file, 'r', encoding='utf-8') as fin:
        for i in fin:
            training_idx, val_idx, test_idx = i.strip().split('\t')
            training_idx = [int(x) for x in training_idx.split(' ')]
            val_idx = [int(x) for x in val_idx.split(' ')]
            test_idx = [int(x) for x in test_idx.split(' ')]
            split = [training_idx, val_idx, test_idx]
    return split

def to_torch_coo_tensor(
    edge_index,
    size,
    edge_attr = None,
):

    size = (size, size)

    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)
    size += edge_attr.size()[1:]
    out = torch.sparse_coo_tensor(edge_index, edge_attr, size,
                                  device=edge_index.device)
    out = out.coalesce()
    return out

def batch_generator(dataset, all_training_idxs, current_idx, batch_size):
    selected_data = [copy.deepcopy(dataset[i]) for i in all_training_idxs[current_idx:current_idx+batch_size]]
    combined_edge_index = [i["edge_index"] for i in selected_data]
    combined_edge_weight = [i["edge_weight"] for i in selected_data]
    combined_x = [i["x"] for i in selected_data]
    combined_y = [i["y"] for i in selected_data]
    batch = []

    for i in range(len(selected_data)):
        batch += [i]*selected_data[i]["x"].shape[0]

    # combined_edge_index = torch.cat(combined_edge_index, dim=1)
    combined_edge_weight = torch.cat(combined_edge_weight)
    combined_x = torch.cat(combined_x)
    combined_y = torch.stack(combined_y, dim=0)
    batch = torch.LongTensor(batch)

    accumulated_idx = 0
    for i in range(len(combined_edge_index)):
        combined_edge_index[i] += accumulated_idx
        accumulated_idx += selected_data[i]["x"].shape[0]
    combined_edge_index = torch.cat(combined_edge_index, dim=1)

    return combined_edge_index, combined_edge_weight, combined_x, combined_y, batch

def avg_num_node(name):
    dataset2avg_num_node = {'MNIST':71, 'CIFAR10':118, 'DD':285, 'MUTAG':18,
                        'NCI1':30, 'NCI109':30,'PROTEINS':39,
                        'ogbg-molhiv':26, 'ogbg-molbbbp':24, 'ogbg-molbace':34}
    if name in dataset2avg_num_node:
        return dataset2avg_num_node[name]
    else:
        print("Unknown avg_num_node of the dataset {}".format(name))
        sys.exit()
