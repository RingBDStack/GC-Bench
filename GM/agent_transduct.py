import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
from utils.utils import match_loss, regularization, row_normalize_tensor, init_feat, get_loops
from utils.init_coreset import init_coreset
import deeprobust.graph.utils as utils
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from networks_nc.gcn import GCN
from networks_nc.sgc import SGC
from networks_nc.gt import GT
from networks_nc.sgc_multi import SGC as SGC1
from networks_nc.parametrized_adj import PGE
from networks_nc.IGNR import GraphonLearner as IGNR
import scipy.sparse as sp
from torch_sparse import SparseTensor


class GCond:

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
        if args.method == "SGDD":
            self.pge = IGNR(
                node_feature=d, nfeat=128, nnodes=n, device=device, args=args
            ).to(device)
        else:
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

        if self.args.save:
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

    def train(self, verbose=True):
        args = self.args
        data = self.data
        if args.wandb:
            import wandb
            wandb.init(
                project="GCBM",
                name=f"{args.method}_{args.dataset}_{args.reduction_rate}_{args.seed}",
            )
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn
        features, adj, labels = data.feat_full, data.adj_full, data.labels_full
        f = data.idx_train

        syn_class_indices = self.syn_class_indices

        features, adj, labels = utils.to_tensor(
            features, adj, labels, device=self.device
        )

        feat_sub, adj_sub = self.get_sub_adj_feat(features)
        self.feat_syn.data.copy_(feat_sub)

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

        outer_loop, inner_loop = get_loops(args)
        if "coreset" in args.init_way:
            idx_selected = init_coreset(data, args)
            feat_syn = features[idx_selected]
            self.labels_syn = labels[idx_selected]
        else:
            feat_syn, feature_init = init_feat(feat_syn, args, data, syn_class_indices)
        # self.feat_syn.data.copy_(feat_syn)
        self.feat_syn = feat_syn
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        loss_avg = 0
        # EXGC
        previous_loss = 1e10
        epochs_without_improvement = 0
        train_acc_mean = 0
        test_acc_mean = 0

        stop_mask = torch.bernoulli(torch.ones(*feat_syn.shape) * self.args.prune).to(
            self.device
        )

        for it in range(args.epochs + 1):
            wandb_log = {}
            if args.dataset in ["ogbn-arxiv"]:
                model = SGC1(
                    nfeat=feat_syn.shape[1],
                    nhid=self.args.hidden,
                    dropout=0.0,
                    with_bn=False,
                    weight_decay=0e-4,
                    nlayers=2,
                    nclass=data.nclass,
                    device=self.device,
                ).to(self.device)
            else:
                if args.sgc == 1:
                    model = SGC(
                        nfeat=data.feat_train.shape[1],
                        nhid=args.hidden,
                        nclass=data.nclass,
                        dropout=args.dropout,
                        nlayers=args.nlayers,
                        with_bn=False,
                        device=self.device,
                    ).to(self.device)
                else:
                    model = GCN(
                        nfeat=data.feat_train.shape[1],
                        nhid=args.hidden,
                        nclass=data.nclass,
                        dropout=args.dropout,
                        nlayers=args.nlayers,
                        device=self.device,
                    ).to(self.device)

            model.initialize()

            model_parameters = list(model.parameters())

            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
            model.train()

            for ol in range(outer_loop):
                if args.method == "SGDD":
                    adj_syn, opt_loss = self.pge(self.feat_syn, Lx=data.adj_mx)
                else:
                    adj_syn = pge(self.feat_syn)
                adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
                feat_syn_norm = feat_syn

                BN_flag = False
                for module in model.modules():
                    if "BatchNorm" in module._get_name():  # BatchNorm
                        BN_flag = True
                if BN_flag:
                    model.train()  # for updating the mu, sigma of BatchNorm
                    output_real = model.forward(features, adj_norm)
                    for module in model.modules():
                        if "BatchNorm" in module._get_name():  # BatchNorm
                            module.eval()  # fix mu and sigma of every BatchNorm layer

                loss = torch.tensor(0.0).to(self.device)
                for c in range(data.nclass):
                    batch_size, n_id, adjs = data.retrieve_class_sampler(
                        c, adj, transductive=True, args=args
                    )
                    if args.nlayers == 1:
                        adjs = [adjs]
                    adjs = [adj.to(self.device) for adj in adjs]
                    # sample neighbors even in graph transformer
                    output = model.forward_sampler(features[n_id], adjs)
                    loss_real = F.nll_loss(output, labels[n_id[:batch_size]])
                    # output = model.forward(features, adj)
                    # loss_real = F.nll_loss(output, labels)
                    # def get_contributing_params(y, top_level=True):
                    #     nf = y.grad_fn.next_functions if top_level else y.next_functions
                    #     for f, _ in nf:
                    #         try:
                    #             yield f.variable
                    #         except AttributeError:
                    #             pass  # node has no tensor
                    #         if f is not None:
                    #             yield from get_contributing_params(f, top_level=False)
                    # not_contributing = set(model_parameters) - set(get_contributing_params(output))
                    # for name, param in model.named_parameters():
                    #     if param.grad is None:
                    #         print(name)
                    gw_real = torch.autograd.grad(loss_real, model_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    output_syn = model.forward(feat_syn, adj_syn_norm)

                    ind = syn_class_indices[c]
                    loss_syn = F.nll_loss(
                        output_syn[ind[0] : ind[1]], labels_syn[ind[0] : ind[1]]
                    )
                    gw_syn = torch.autograd.grad(
                        loss_syn, model_parameters, create_graph=True
                    )
                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                    loss += coeff * match_loss(
                        gw_syn, gw_real, args, device=self.device
                    )

                loss_avg += loss.item()
                # TODO: regularize
                if args.alpha > 0:
                    loss_reg = args.alpha * regularization(
                        adj_syn, utils.tensor2onehot(labels_syn)
                    )
                else:
                    loss_reg = torch.tensor(0)

                loss = loss + loss_reg

                if args.opt_scale > 0:  # if not SGDD, opt_scale default 0
                    loss_opt = args.opt_scale * opt_loss
                else:
                    loss_opt = torch.tensor(0)

                wandb_log["loss_opt"] = loss_opt
                loss = loss + loss_opt

                # update sythetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()
                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                # EXGC
                if args.method == "EXGC":
                    if ol == outer_loop - 1:
                        zero_indices = (stop_mask == 0).nonzero()
                        num_zeros = zero_indices.size(0)
                        if num_zeros != 0:
                            current_zeros_ratio = num_zeros / stop_mask.numel()

                            num_elements = stop_mask.numel()

                            num_to_convert = int(num_elements / self.args.circulation)
                            indices_to_convert = zero_indices[
                                torch.randperm(num_zeros)[:num_to_convert]
                            ]

                            stop_mask[
                                indices_to_convert[:, 0], indices_to_convert[:, 1]
                            ] = 1
                            print(
                                "it",
                                it,
                                "num_zeros",
                                num_zeros,
                                "current_zeros_ratio",
                                current_zeros_ratio,
                                "num_to_convert",
                                num_to_convert,
                                "nonzero",
                                torch.nonzero(stop_mask).size(0),
                            )

                    feat_syn.data = feat_syn.data * stop_mask

                if args.debug and ol % 5 == 0:
                    print("Gradient matching loss:", loss.item())

                if ol == outer_loop - 1:
                    # print('loss_reg:', loss_reg.item())
                    # print('Gradient matching loss:', loss.item())
                    break

                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn_inner)
                adj_syn_inner_norm = utils.normalize_adj_tensor(
                    adj_syn_inner, sparse=False
                )
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(
                        feat_syn_inner_norm, adj_syn_inner_norm
                    )
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    # print(loss_syn_inner.item())
                    optimizer_model.step()  # update gnn param

            loss_avg /= data.nclass * outer_loop
            if it % 50 == 0:
                print("Epoch {}, loss_avg: {}".format(it, loss_avg))

            eval_epochs = [
                50,
                100,
                200,
                300,
                400,
                500,
                600,
                800,
                1000,
                1200,
                1600,
                2000,
                3000,
                4000,
                5000,
            ]

            if verbose and it in eval_epochs:
                # if verbose and (it+1) % 50 == 0:
                res = []
                runs = 1 if args.dataset in ["ogbn-arxiv"] else 3
                for i in range(runs):
                    if args.dataset in ["ogbn-arxiv"]:
                        res.append(self.test_with_val())
                    else:
                        res.append(self.test_with_val())

                res = np.array(res)
                print("Train/Test Mean Accuracy:", repr([res.mean(0), res.std(0)]))
                train_acc_mean = res.mean(0)[0]
                test_acc_mean = res.mean(0)[1]

            wandb_log["loss_avg"] = loss_avg
            wandb_log["train_acc_mean"] = train_acc_mean
            wandb_log["test_acc_mean"] = test_acc_mean
            if args.wandb:
                import wandb

                wandb.log(wandb_log)

            if loss_avg < previous_loss:
                previous_loss = loss_avg
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Check if the maximum number of epochs without improvement has been reached
            if (
                args.early_stopping
                and epochs_without_improvement
                >= self.args.max_epochs_without_improvement
            ):
                print(
                    f"Training stopped as the loss did not decrease for {self.args.max_epochs_without_improvement} consecutive epochs."
                )
                break

    def get_sub_adj_feat(self, features):
        data = self.data
        args = self.args
        idx_selected = []

        from collections import Counter

        counter = Counter(self.labels_syn.cpu().numpy())

        if args.method == "SGDD":
            ids_per_cls_train = [
                (self.data.labels_train == c).nonzero()[0] for c in counter.keys()
            ]
            idx_selected = data.sampling(
                ids_per_cls_train, counter, features, 0.5, counter
            )
            features = features[idx_selected]
            return features, None

        for c in range(data.nclass):
            tmp = data.retrieve_class(c, num=counter[c])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[self.data.idx_train][idx_selected]

        # adj_knn = torch.zeros((data.nclass*args.nsamples, data.nclass*args.nsamples)).to(self.device)
        # for i in range(data.nclass):
        #     idx = np.arange(i*args.nsamples, i*args.nsamples+args.nsamples)
        #     adj_knn[np.ix_(idx, idx)] = 1

        from sklearn.metrics.pairwise import cosine_similarity

        # features[features!=0] = 1
        k = 2
        sims = cosine_similarity(features.cpu().numpy())
        sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
        for i in range(len(sims)):
            indices_argsort = np.argsort(sims[i])
            sims[i, indices_argsort[:-k]] = 0
        adj_knn = torch.FloatTensor(sims).to(self.device)
        return features, adj_knn


