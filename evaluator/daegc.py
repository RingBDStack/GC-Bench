import argparse
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam

from torch_geometric.datasets import Planetoid

import _utils
from model import GCN_nc
from evaluation import eva


class DAEGC(nn.Module):
    def __init__(
        self, num_features, hidden_size, embedding_size, alpha, num_clusters, with_pretrain, v=1
    ):
        super(DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v
        self.gcn = GCN_nc(num_features, hidden_size, embedding_size, alpha)
        if with_pretrain:
            self.gcn.load_state_dict(torch.load(args.pretrain_path, map_location="cpu"))
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, adj):
        A_pred, z = self.gcn(x, adj)
        q = self.get_Q(z)

        return A_pred, z, q

    def get_Q(self, z):
        q = 1.0 / (
            1.0
            + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v
        )
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def trainer(args):
    feat_syn, adj_syn, labels_syn = _utils.get_syn_data(args)
    model = DAEGC(
        num_features=feat_syn.shape[1],
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.alpha,
        num_clusters=args.n_clusters,
        with_pretrain=args.pretrain
    ).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    adj_label = torch.from_numpy(adj_syn).to(dtype=torch.float).to(device)
    adj_syn = torch.from_numpy(adj_syn).to(dtype=torch.float).to(device)
    adj_syn += torch.eye(feat_syn.shape[0]).to(device)
    adj_syn = normalize(adj_syn.cpu().numpy(), norm='l1')
    adj_syn = torch.from_numpy(adj_syn).to(dtype=torch.float).to(device)

    x = feat_syn
    y = labels_syn.cpu().numpy()
    if isinstance(y, np.ndarray) and y.ndim != 1:
        y = np.argmax(y, axis=1)
    

    ori_dataset = Planetoid(args.data_dir, args.dataset)[0]
    ori_dataset.adj = torch.sparse_coo_tensor(
        ori_dataset.edge_index, torch.ones(ori_dataset.edge_index.shape[1]), torch.Size([ori_dataset.x.shape[0], ori_dataset.x.shape[0]])
    ).to_dense()
    ori_dataset.adj_label =  ori_dataset.adj
    ori_dataset.adj += torch.eye(ori_dataset.x.shape[0])
    ori_dataset.adj = normalize(ori_dataset.adj, norm="l1")
    ori_dataset.adj = torch.from_numpy(ori_dataset.adj).to(dtype=torch.float)
    adj_ori =  ori_dataset.adj.to(device)
    x_ori = torch.Tensor(ori_dataset.x).to(device)
    y_ori = ori_dataset.y.cpu().numpy()
    def test(phase):
        with torch.no_grad():
            _, z = model.gcn(x_ori, adj_ori)

        kmeans = KMeans(n_clusters=args.n_clusters)
        y_pred = kmeans.fit_predict(z.data.cpu().numpy())
        model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
        eva(y_ori, y_pred, phase)

    test("pretrain")
    best_acc = 0

    for epoch in range(args.epoch):
        model.train()
        if epoch % args.update_interval == 0:
            # update_interval
            A_pred, z, Q = model(x, adj_syn)

            q = Q.detach().data.cpu().numpy().argmax(1)  # Q
            acc, nmi, ari, f1 = eva(y, q, epoch)
            if acc > best_acc:
                best_acc = acc
                best_model_state = model.state_dict()
            test(epoch)

        A_pred, z, q = model(x, adj_syn)
        p = target_distribution(Q.detach())

        kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))

        loss = 10 * kl_loss + re_loss
        print(f"epoch {epoch}: loss {loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    model.load_state_dict(best_model_state)
    test("final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default="citeseer")
    parser.add_argument("--epoch", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_clusters", default=6, type=int)
    parser.add_argument("--update_interval", default=1, type=int)  # [1,3,5]
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--embedding_size", default=16, type=int)
    parser.add_argument("--weight_decay", type=int, default=5e-3)
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--method", type=str, default="SGDD", help="Method")
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
    parser.add_argument("--pretrain", type=int, default=0)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    # datasets = _utils.get_dataset(args.dataset)
    # dataset = datasets[0]

    if args.dataset == "citeseer":
        args.lr = 0.0001
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

    args.pretrain_path = f"./pretrain/predaegc_{args.dataset}_{args.method}_{args.reduction_rate}.pkl"

    print(args)
    trainer(args)
