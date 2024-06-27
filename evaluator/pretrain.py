import argparse
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from model import GCN_nc
from evaluation import eva
import _utils
import torch


def pretrain(args):
    feat_syn, adj_syn, labels_syn = _utils.get_syn_data(args)
    model = GCN_nc(
        num_features=feat_syn.shape[1],
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.alpha,
    ).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # data process
    # dataset = _utils.data_preprocessing(dataset)
    # adj = dataset.adj.to(device)
    # adj_label = dataset.adj_label.to(device)
    adj_label = torch.from_numpy(adj_syn).to(dtype=torch.float).to(device)
    adj_syn = torch.from_numpy(adj_syn).to(dtype=torch.float).to(device)
    adj_syn += torch.eye(feat_syn.shape[0]).to(device)
    adj_syn = normalize(adj_syn.cpu().numpy(), norm='l1')
    adj_syn = torch.from_numpy(adj_syn).to(dtype=torch.float).to(device)

    # M = _utils.get_M(adj_syn).to(device)

    # data and label
    x = feat_syn
    y = labels_syn.cpu().numpy()
    if isinstance(y, np.ndarray) and y.ndim != 1:
        y = np.argmax(y, axis=1)

    for epoch in range(args.max_epoch):
        model.train()
        # A_pred, z = model(x, adj_syn, M)
        A_pred, z = model(x, adj_syn)
        # print(adj_label)
        # print(f"A_pred:{A_pred.shape}, adj_label:{adj_label.shape}")
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # _, z = model(x, adj_syn, M)
            _, z = model(x, adj_syn)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(
                z.data.cpu().numpy()
            )
            acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)
       
    torch.save(
        model.state_dict(), f"./pretrain/predaegc_{args.dataset}_{args.method}_{args.reduction_rate}.pkl"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_clusters", default=6, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--embedding_size", default=16, type=int)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--method", type=str, default="SFGC", help="Method")
    parser.add_argument("--transductive", type=int, default=1)
    parser.add_argument(
        "--save_dir", type=str, default="save", help="synthetic dataset directory"
    )
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--keep_ratio", type=float, default=1)
    parser.add_argument("--reduction_rate", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--normalize_features", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--mlp", type=int, default=0)
    parser.add_argument("--inner", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=-1)
    parser.add_argument("--nruns", type=int, default=2)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    print(args)
    device = torch.device("cuda" if args.cuda else "cpu")

    # dataset = get_dataset(args.dataset, dataset_dir="data")
    # dataset = datasets[0]

    if args.dataset == "citeseer":
        args.lr = 0.005
        args.k = None
        args.n_clusters = 6
    elif args.dataset == "cora":
        args.lr = 0.005
        args.k = None
        args.n_clusters = 7
    elif args.dataset == "Pubmed":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 3
    else:
        args.k = None

    # args.input_dim = dataset.num_features

    print(args)
    pretrain(args)
