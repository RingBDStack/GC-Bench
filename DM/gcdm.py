import torch
import numpy as np
from networks_nc.parametrized_adj import PGE
from networks_nc.gcn import GCN
from networks_nc.sgc import SGC
from utils.utils import get_loops, init_feat
import deeprobust.graph.utils as utils
import torch.nn.functional as F
from torch_sparse import SparseTensor


class GCDM:
    def __init__(self, data, args, device="cuda", **kwargs):
        self.data = data
        self.args = args
        self.device = device

        # n = data.nclass * args.nsamples
        syn_label = self.generate_labels_syn(data)
        self.data.labels_syn = np.array(syn_label)
        self.labels_syn = torch.LongTensor(syn_label).to(device)
        n = self.data.labels_syn.shape[0]
        self.nnodes_syn = n
        d = data.feat_train.shape[1]
        print(
            f"target reduced size:{int(data.feat_train.shape[0] * args.reduction_rate)}"
        )
        print(f"actual reduced size:{n}")

        # from collections import Counter; print(Counter(data.labels_train))

        self.feat_syn = torch.FloatTensor(n, d).to(device)
        self.pge = PGE(nfeat=d, nnodes=n, device=device, args=args).to(device)

        self.reset_parameters()
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        print("adj_syn:", (n, n), "feat_syn:", self.feat_syn.shape)

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def generate_labels_syn(self, data):
        from collections import Counter

        if not hasattr(data, "labels_train"):
            data.labels_train = data.labels[data.idx_train]
        counter = Counter(data.labels_train)
        num_class_dict = {}
        # n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
            sum_ += num_class_dict[c]
            self.syn_class_indices[c] = [
                len(labels_syn),
                len(labels_syn) + num_class_dict[c],
            ]
            labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn

    def train(self, verbose=True):
        args = self.args
        data = self.data
        outer_loop, inner_loop = get_loops(args)
        if args.wandb:
            import wandb

            wandb.init(
                project="GCBM",
                name=f"{args.method}_{args.dataset}_{args.reduction_rate}_{args.seed}",
            )
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn

        features, adj, labels = data.feat_full, data.adj_full, data.labels_full
        if not args.transductive:
            features, adj, labels = data.feat_train, data.adj_train, data.labels_train
        features, adj, labels = utils.to_tensor(
            features, adj, labels, device=self.device
        )
        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)
        adj = adj_norm
        adj = SparseTensor(
            row=adj._indices()[0],
            col=adj._indices()[1],
            value=adj._values(),
            sparse_sizes=adj.size(),
        ).t()
        device = self.device
        if args.sgc:
            model = SGC(
                nfeat=feat_syn.shape[1],
                nhid=args.hidden,
                dropout=0.5,
                weight_decay=0e-4,
                nlayers=2,
                with_bn=False,
                nclass=data.nclass,
                device=device,
            ).to(device)
        else:
            model = GCN(
                nfeat=feat_syn.shape[1],
                nhid=args.hidden,
                dropout=0.5,
                weight_decay=0e-4,
                nlayers=2,
                with_bn=False,
                nclass=data.nclass,
                device=device,
            ).to(device)
        syn_class_indices = self.syn_class_indices
        feat_syn, feature_init = init_feat(feat_syn, args, data, syn_class_indices)
        adj_syn = pge.inference(feat_syn)
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        for it in range(args.epochs + 1):
            wandb_log = {}
            model.initialize()
            model_parameters = list(model.parameters())
            self.optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
            model.train()

            labels_unique = self.num_class_dict.keys()
            loss_avg = 0

            for ol in range(outer_loop):
                with torch.no_grad():
                    embedding_real, _ = model.forward(features, adj, get_embedding=True)
                    if args.transductive:
                        embedding_real = embedding_real[data.idx_train]
                        train_labels = labels[data.idx_train]
                    else:
                        train_labels = labels
                    mean_emb_real = torch.zeros(
                        (len(labels_unique), embedding_real.size(1)),
                        device=embedding_real.device,
                    )
                    for i, label in enumerate(labels_unique):
                        label_mask = train_labels == label
                        mean_emb_real[i] = torch.mean(embedding_real[label_mask], dim=0)

                adj_syn = pge(feat_syn)
                embedding_syn, _ = model.forward(feat_syn, adj_syn, get_embedding=True)
                mean_emb_syn = torch.zeros(
                    (len(labels_unique), embedding_syn.size(1)),
                    device=embedding_syn.device,
                )
                for i, label in enumerate(labels_unique):
                    label_mask = labels_syn == label
                    mean_emb_syn[i] = torch.mean(embedding_syn[label_mask], dim=0)

                # loss_emb = torch.sum(
                #     torch.mean((mean_emb_syn - mean_emb_real) ** 2, dim=1)
                # )
                loss_emb = torch.mean((mean_emb_syn - mean_emb_real) ** 2).sum()
                loss_avg += loss_emb.item()

                self.optimizer_pge.zero_grad()
                self.optimizer_feat.zero_grad()

                loss_emb.backward()
                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                if ol == outer_loop - 1:
                    break

                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn_inner)
                adj_syn_inner_norm = utils.normalize_adj_tensor(
                    adj_syn_inner, sparse=False
                )

                for _ in range(inner_loop):
                    self.optimizer_model.zero_grad()
                    embedding_syn, _ = model.forward(
                        feat_syn_inner, adj_syn_inner_norm, get_embedding=True
                    )
                    mean_emb_syn = torch.zeros(
                        (len(labels_unique), embedding_syn.size(1)),
                        device=embedding_syn.device,
                    )
                    for i, label in enumerate(labels_unique):
                        label_mask = labels_syn == label
                        mean_emb_syn[i] = torch.mean(embedding_syn[label_mask], dim=0)
                    loss_syn_inner = torch.mean(
                        (mean_emb_syn - mean_emb_real) ** 2
                    ).sum()
                    # loss_syn_inner = torch.sum(
                    #     torch.mean((mean_emb_syn - mean_emb_real) ** 2, dim=1)
                    # )
                    loss_syn_inner.backward()
                    self.optimizer_model.step()
                    with torch.no_grad():
                        embedding_real, _ = model.forward(
                            features, adj, get_embedding=True
                        )
                        if args.transductive:
                            embedding_real = embedding_real[data.idx_train]
                        mean_emb_real = torch.zeros(
                            (len(labels_unique), embedding_real.size(1)),
                            device=embedding_real.device,
                        )
                        for i, label in enumerate(labels_unique):
                            label_mask = train_labels == label
                            mean_emb_real[i] = torch.mean(
                                embedding_real[label_mask], dim=0
                            )
                self.feat_syn = feat_syn
                self.pge = pge

            loss_avg /= data.nclass * outer_loop
            wandb_log["loss_avg"] = loss_avg
            if it % 50 == 0 and verbose:
                print("Epoch {}, loss_avg: {}".format(it, loss_avg))
            if it % 100 == 0:
                res = []
                runs = 1 if args.dataset in ["ogbn-arxiv"] else 3
                for i in range(runs):
                    if args.dataset in ["ogbn-arxiv"]:
                        res.append(self.test_with_val())
                    else:
                        res.append(self.test_with_val())

                res = np.array(res)
                train_acc_mean = res.mean(0)[0]
                test_acc_mean = res.mean(0)[1]
                train_acc_std = res.std(0)[0]
                test_acc_std = res.std(0)[1]
                wandb_log["train_acc_mean"] = train_acc_mean
                wandb_log["test_acc_mean"] = test_acc_mean
                wandb_log["train_acc_std"] = train_acc_std
                wandb_log["test_acc_std"] = test_acc_std
                if args.wandb:
                    wandb.log(wandb_log)
                if verbose:
                    print("Train/Test Mean Accuracy:", repr([res.mean(0), res.std(0)]))

    def test_with_val(self, verbose=True):
        res = []

        data, device = self.data, self.device
        feat_syn, pge, labels_syn = self.feat_syn.detach(), self.pge, self.labels_syn

        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        model = GCN(
            nfeat=feat_syn.shape[1],
            nhid=self.args.hidden,
            dropout=0.5,
            weight_decay=5e-4,
            nlayers=2,
            nclass=data.nclass,
            device=device,
        ).to(device)

        if self.args.dataset in ["ogbn-arxiv"]:
            model = GCN(
                nfeat=feat_syn.shape[1],
                nhid=self.args.hidden,
                dropout=0.5,
                weight_decay=0e-4,
                nlayers=2,
                with_bn=False,
                nclass=data.nclass,
                device=device,
            ).to(device)

        adj_syn = pge.inference(feat_syn)
        args = self.args

        if args.save:
            torch.save(
                adj_syn,
                f"{args.save_dir}/{args.method}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt",
            )
            torch.save(
                feat_syn,
                f"{args.save_dir}/{args.method}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt",
            )

        if self.args.lr_adj == 0:
            n = len(labels_syn)
            adj_syn = torch.zeros((n, n))

        model.fit_with_val(
            feat_syn,
            adj_syn,
            labels_syn,
            data,
            train_iters=600,
            normalize=True,
            verbose=False,
        )

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        labels_train = torch.LongTensor(data.labels_train).cuda()
        output = model.predict(data.feat_train, data.adj_train)
        loss_train = F.nll_loss(output, labels_train)
        acc_train = utils.accuracy(output, labels_train)
        if verbose:
            print(
                "Train set results:",
                "loss= {:.4f}".format(loss_train.item()),
                "accuracy= {:.4f}".format(acc_train.item()),
            )
        res.append(acc_train.item())

        # Full graph
        output = model.predict(data.feat_full, data.adj_full)
        loss_test = F.nll_loss(output[data.idx_test], labels_test)
        acc_test = utils.accuracy(output[data.idx_test], labels_test)
        res.append(acc_test.item())
        if verbose:
            print(
                "Test set results:",
                "loss= {:.4f}".format(loss_test.item()),
                "accuracy= {:.4f}".format(acc_test.item()),
            )
        return res
