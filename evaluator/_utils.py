import numpy as np
import torch
from sklearn.preprocessing import normalize
import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from torch_geometric.datasets import Planetoid
from utils.utils import *
from utils.utils_graph import DataGraph
import glob

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

def get_files_and_seed(args,method, dataset, reduction_rate, file_str):
    save_dir = args.save_dir
    if args.method in ['kcenter','random','herding']:
        adj_files_pattern = os.path.join(save_dir, method, f'idx_{dataset}_{reduction_rate}{file_str}*.npy')
    else:
        adj_files_pattern = os.path.join(save_dir, method, f'adj_{dataset}_{reduction_rate}{file_str}*.pt')
    adj_files = glob.glob(adj_files_pattern)
    if adj_files:
        adj_file = adj_files[0]
        seed = adj_file.split('_')[-1].split('.')[0]
        return seed
    else:
        raise FileNotFoundError(f"No files found matching pattern {adj_files_pattern}")


def get_syn_data(args, seed=None):
    data_pyg = ["cora", "citeseer", "pubmed", 'cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel']
    if args.dataset in data_pyg:
        data_full = get_dataset(args.dataset, args.normalize_features,args.data_dir)
        data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
    else:
        data = DataGraph(args.dataset)
        data_full = data.data_full
    file_str = '_best_ntk_score_' if args.method == 'SFGC' else '_'
    if args.method in ['SFGC', 'GEOM']:
        save_dir = args.save_dir
        method = args.method
        dataset = args.dataset
        reduction_rate = args.reduction_rate
        adj_files_pattern = os.path.join(save_dir, method, f'adj_{dataset}_{reduction_rate}{file_str}*.pt')
        adj_files = glob.glob(adj_files_pattern)
        if adj_files:
            adj_file = adj_files[0]
            seed = adj_file.split('_')[-1].split('.')[0]
            # adj_syn = torch.load(adj_file, map_location='cuda')
            feat_file = os.path.join(save_dir, method, f'feat_{dataset}_{reduction_rate}{file_str}{seed}.pt')
            feat_syn = torch.load(feat_file, map_location='cuda')
            adj_syn = torch.eye(feat_syn.shape[0])
            label_file = os.path.join(save_dir, method, f'label_{dataset}_{reduction_rate}{file_str}{seed}.pt')
            labels_syn = torch.load(label_file, map_location='cuda')
            print(f"adj_syn:{adj_syn.shape}, feat_syn:{feat_syn.shape},labels_syn:{labels_syn.shape}")
    else:
        if seed is None:
            seed = get_files_and_seed(args, args.method, args.dataset, args.reduction_rate, file_str)
        else:
            seed = args.seed
        if args.method in ['kcenter','random','herding']:
            features = data.feat_full
            adj = data.adj_full
            labels = data.labels_full
            idx = np.load(
                f"{args.save_dir}/{args.method}/idx_{args.dataset}_{args.reduction_rate}_{seed}.npy"
            )
            feat_syn = torch.from_numpy(features[idx]).to(args.device)
            adj_syn = adj[np.ix_(idx, idx)].toarray()
            adj_syn = torch.FloatTensor(adj_syn).to(args.device)
            labels_syn = torch.from_numpy(labels[idx]).to(args.device)
        else:
            adj_syn = torch.load(f'{args.save_dir}/{args.method}/adj_{args.dataset}_{args.reduction_rate}_{seed}.pt', map_location='cuda')
            feat_syn = torch.load(f'{args.save_dir}/{args.method}/feat_{args.dataset}_{args.reduction_rate}_{seed}.pt', map_location='cuda')
            labels_syn = torch.LongTensor(generate_labels_syn(args, data)).to(args.device) 
    print('Sum:', adj_syn.sum(), adj_syn.sum()/(adj_syn.shape[0]**2))
    print('Sparsity:', adj_syn.nonzero().shape[0]/(adj_syn.shape[0]**2))

    if args.epsilon > 0:
        adj_syn[adj_syn < args.epsilon] = 0
        print('Sparsity after truncating:', adj_syn.nonzero().shape[0]/(adj_syn.shape[0]**2))
    
    feat_syn = feat_syn.to(args.device)
    adj_syn = adj_syn.cpu().numpy()
    adj_syn = normalize(adj_syn, norm='l1')
    return feat_syn, adj_syn, labels_syn

def data_preprocessing(dataset):
    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index, torch.ones(dataset.edge_index.shape[1]), torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()
    dataset.adj_label = dataset.adj

    dataset.adj += torch.eye(dataset.x.shape[0])
    dataset.adj = normalize(dataset.adj, norm="l1")
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)

    return dataset

def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


