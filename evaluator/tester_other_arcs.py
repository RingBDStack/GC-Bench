import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
import deeprobust.graph.utils as utils
from copy import deepcopy
import numpy as np
from networks_nc.gcn import GCN
from networks_nc.sgc import SGC
from networks_nc.sgc_multi import SGC as SGC1
from networks_nc.myappnp import APPNP
from networks_nc.myappnp1 import APPNP1
from networks_nc.mycheby import Cheby
from networks_nc.mygraphsage import GraphSage
from networks_nc.gat import GAT
from networks_nc.gt import GT
import scipy.sparse as sp


class Evaluator:

    def __init__(self, data, args, device="cuda", **kwargs):
        self.data = data
        self.args = args
        self.device = device
        n = int(data.feat_train.shape[0] * args.reduction_rate)
        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.adj_param = nn.Parameter(torch.FloatTensor(n, n).to(device))
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))
        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)
        self.reset_parameters()
        print("adj_param:", self.adj_param.shape, "feat_syn:", self.feat_syn.shape)

    def reset_parameters(self):
        self.adj_param.data.copy_(torch.randn(self.adj_param.size()))
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def generate_labels_syn(self, data):
        from collections import Counter

        counter = Counter(data.labels_train)
        num_class_dict = {}
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                self.syn_class_indices[c] = [
                    len(labels_syn),
                    len(labels_syn) + num_class_dict[c],
                ]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [
                    len(labels_syn),
                    len(labels_syn) + num_class_dict[c],
                ]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn

    def test_pymodel(self, nlayers, model_type, verbose=False):
        res = []
        args = self.args

        if args.dataset in ["cora", "citeseer"]:
            args.epsilon = (
                0.5  # Make the graph sparser as GAT does not work well on dense graph
            )
        else:
            args.epsilon = 0.01

        print("======= testing %s" % model_type)
        data, device = self.data, self.device

        feat_syn, adj_syn, labels_syn = self.get_syn_data(model_type)
        # with_bn = True if self.args.dataset in ['ogbn-arxiv'] else False
        with_bn = False
        if model_type == "GAT":
            model = GAT(
                nfeat=feat_syn.shape[1],
                nhid=16,
                heads=16,
                dropout=0.0,
                weight_decay=0e-4,
                nlayers=self.args.nlayers,
                lr=0.001,
                nclass=data.nclass,
                device=device,
                dataset=self.args.dataset,
            ).to(device)
        noval = True if args.dataset in ["reddit", "flickr"] else False
        model.fit(
            feat_syn,
            adj_syn,
            labels_syn,
            np.arange(len(feat_syn)),
            noval=noval,
            data=data,
            train_iters=10000 if noval else 3000,
            normalize=True,
            verbose=verbose,
        )

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        if args.dataset in ["reddit", "flickr"]:
            output = model.predict(data.feat_test, data.adj_test)
            loss_test = F.nll_loss(output, labels_test)
            acc_test = utils.accuracy(output, labels_test)
            res.append(acc_test.item())
            if verbose:
                print(
                    "Test set results:",
                    "loss= {:.4f}".format(loss_test.item()),
                    "accuracy= {:.4f}".format(acc_test.item()),
                )

        else:
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
        return res

    def get_syn_data(self, model_type=None, coreset=False):
        data, device = self.data, self.device
        args = self.args
        if coreset:
            features = data.feat_full
            adj = data.adj_full
            labels = data.labels_full
            idx = np.load(
                f"{args.save_dir}/{args.method}/idx_{args.dataset}_{args.reduction_rate}_{args.seed}.npy"
            )
            feat_syn = torch.from_numpy(features[idx]).to(self.device)
            adj_syn = adj[np.ix_(idx, idx)].toarray()
            adj_syn = torch.FloatTensor(adj_syn).to(self.device)
            labels_syn = torch.from_numpy(labels[idx]).to(self.device)
            self.feat_syn = feat_syn
            self.adj_syn = adj_syn
            self.labels_syn = labels_syn
        else:
            feat_syn, adj_param, labels_syn = (
                self.feat_syn.detach(),
                self.adj_param.detach(),
                self.labels_syn,
            )

            args = self.args
            if args.method == "GEOM" or args.method == "SFGC":
                labels_syn = torch.load(
                    f"{args.save_dir}/{args.method}/label_{args.dataset}_{args.reduction_rate}_best_ntk_score_{args.carch}.pt",
                    map_location="cuda",
                )
                # labels_syn = torch.argmax(labels_syn, dim=1)
                adj_syn = torch.load(
                f"{args.save_dir}/{args.method}/adj_{args.dataset}_{args.reduction_rate}_best_ntk_score_{args.carch}.pt",
                map_location="cuda",
                )
                feat_syn = torch.load(
                    f"{args.save_dir}/{args.method}/feat_{args.dataset}_{args.reduction_rate}_best_ntk_score_{args.carch}.pt",
                    map_location="cuda",
                )
            else:
                adj_syn = torch.load(
                    f"{args.save_dir}/{args.method}/adj_{args.dataset}_{args.reduction_rate}_{args.carch}.pt",
                    map_location="cuda",
                )
                feat_syn = torch.load(
                    f"{args.save_dir}/{args.method}/feat_{args.dataset}_{args.reduction_rate}_{args.carch}.pt",
                    map_location="cuda",
                )
            # adj_syn = torch.eye(feat_syn.shape[0]).to(self.device)
            # print("adj_syn now is eye matrix")

        if model_type == "MLP":
            adj_syn = adj_syn.to(self.device)
            adj_syn = adj_syn - adj_syn
        else:
            adj_syn = adj_syn.to(self.device)

        # print('Sum:', adj_syn.sum(), adj_syn.sum()/(adj_syn.shape[0]**2))
        # print('Sparsity:', adj_syn.nonzero().shape[0]/(adj_syn.shape[0]**2))

        if self.args.epsilon > 0:
            adj_syn[adj_syn < self.args.epsilon] = 0
            # print('Sparsity after truncating:', adj_syn.nonzero().shape[0]/(adj_syn.shape[0]**2))
        feat_syn = feat_syn.to(self.device)

        # edge_index = adj_syn.nonzero().T
        # adj_syn = torch.sparse.FloatTensor(edge_index,  adj_syn[edge_index[0], edge_index[1]], adj_syn.size())

        return feat_syn, adj_syn, labels_syn

    def test(self, nlayers, model_type, verbose=True):
        res = []

        args = self.args
        data, device = self.data, self.device

        if self.args.method in ["herding", "kcenter", "random"]:
            feat_syn, adj_syn, labels_syn = self.get_syn_data(model_type, coreset=True)
        else:
            feat_syn, adj_syn, labels_syn = self.get_syn_data(model_type)

        # adj_syn = torch.eye(feat_syn.shape[0]).to(device)

        print("======= testing %s" % model_type)
        if model_type == "MLP":
            model_class = GCN
        else:
            model_class = eval(model_type)
        weight_decay = 5e-4
        dropout = 0.5 if args.dataset in ["reddit"] else 0

        model = model_class(
            nfeat=feat_syn.shape[1],
            nhid=args.hidden,
            dropout=dropout,
            weight_decay=weight_decay,
            nlayers=nlayers,
            nclass=data.nclass,
            device=device,
        ).to(device)

        # with_bn = True if self.args.dataset in ['ogbn-arxiv'] else False
        if args.dataset in ["ogbn-arxiv", "arxiv"]:
            model = model_class(
                nfeat=feat_syn.shape[1],
                nhid=args.hidden,
                dropout=0.0,
                weight_decay=weight_decay,
                nlayers=nlayers,
                with_bn=False,
                nclass=data.nclass,
                device=device,
            ).to(device)

        noval = True if args.dataset in ["reddit", "flickr"] else False
        model.fit_with_val(
            feat_syn,
            adj_syn,
            labels_syn,
            data,
            train_iters=600,
            normalize=True,
            verbose=False,
            noval=noval,
        )

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        if model_type == "MLP":
            output = model.predict_unnorm(data.feat_test, sp.eye(len(data.feat_test)))
        else:
            output = model.predict(data.feat_test, data.adj_test)

        if args.dataset in ["reddit", "flickr"]:
            loss_test = F.nll_loss(output, labels_test)
            acc_test = utils.accuracy(output, labels_test)
            res.append(acc_test.item())
            if verbose:
                print(
                    "Test set results:",
                    "loss= {:.4f}".format(loss_test.item()),
                    "accuracy= {:.4f}".format(acc_test.item()),
                )

        # if not args.dataset in ['reddit', 'flickr']:
        else:
            # Full graph
            output = model.predict(data.feat_full, data.adj_full)
            loss_test = F.nll_loss(output[data.idx_test], labels_test)
            acc_test = utils.accuracy(output[data.idx_test], labels_test)
            res.append(acc_test.item())
            if verbose:
                print(
                    "Test full set results:",
                    "loss= {:.4f}".format(loss_test.item()),
                    "accuracy= {:.4f}".format(acc_test.item()),
                )

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
        return res

    def test_ori(self, nlayers, model_type, verbose=True):
        res = []
        args = self.args
        data, device = self.data, self.device

        features = data.feat_full
        adj = data.adj_full
        labels = data.labels_full
        idx_train = data.idx_train if args.dataset not in ["reddit", "flickr"] else None
        if args.dataset in ["reddit", "flickr"]:
            features = data.feat_train
            adj = data.adj_train
            labels = data.labels_train

        print("======= testing full graph on %s" % model_type)
        if model_type == "MLP":
            model_class = GCN
        else:
            model_class = eval(model_type)
        weight_decay = 5e-4
        dropout = 0.5 if args.dataset in ["reddit"] else 0

        model = model_class(
            nfeat=features.shape[1],
            nhid=args.hidden,
            dropout=dropout,
            weight_decay=weight_decay,
            nlayers=nlayers,
            nclass=data.nclass,
            device=device,
        ).to(device)

        if args.dataset in ["ogbn-arxiv", "arxiv"]:
            model = model_class(
                nfeat=features.shape[1],
                nhid=args.hidden,
                dropout=0.0,
                weight_decay=weight_decay,
                nlayers=nlayers,
                with_bn=False,
                nclass=data.nclass,
                device=device,
            ).to(device)

        noval = True if args.dataset in ["reddit", "flickr"] else False
        model.fit_with_val(
            features,
            adj,
            labels,
            data,
            train_iters=1000,
            normalize=True,
            verbose=False,
            noval=noval,
            idx_train=idx_train,
        )

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        if model_type == "MLP":
            output = model.predict_unnorm(data.feat_test, sp.eye(len(data.feat_test)))
        else:
            output = model.predict(data.feat_test, data.adj_test)

        if args.dataset in ["reddit", "flickr"]:
            loss_test = F.nll_loss(output, labels_test)
            acc_test = utils.accuracy(output, labels_test)
            res.append(acc_test.item())
            if verbose:
                print(
                    "Test set results:",
                    "loss= {:.4f}".format(loss_test.item()),
                    "accuracy= {:.4f}".format(acc_test.item()),
                )

        # if not args.dataset in ['reddit', 'flickr']:
        else:
            # Full graph
            output = model.predict(data.feat_full, data.adj_full)
            loss_test = F.nll_loss(output[data.idx_test], labels_test)
            acc_test = utils.accuracy(output[data.idx_test], labels_test)
            res.append(acc_test.item())
            if verbose:
                print(
                    "Test full set results:",
                    "loss= {:.4f}".format(loss_test.item()),
                    "accuracy= {:.4f}".format(acc_test.item()),
                )

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
        return res

    def test_ori_pymodel(self, nlayers, model_type, verbose=False):
        res = []
        args = self.args

        if args.dataset in ["cora", "citeseer"]:
            args.epsilon = (
                0.5  # Make the graph sparser as GAT does not work well on dense graph
            )
        else:
            args.epsilon = 0.01

        print("======= testing full graph %s" % model_type)
        data, device = self.data, self.device

        if args.dataset in ["reddit", "flickr"]:
            features = data.feat_train
            adj = data.adj_train
            labels = data.labels_train
        else:
            features = data.feat_full
            adj = data.adj_full
            labels = data.labels_full
        idx_train = data.idx_train if args.dataset not in ["reddit", "flickr"] else None

        # with_bn = True if self.args.dataset in ['ogbn-arxiv'] else False
        with_bn = False
        if model_type == "GAT":
            model = GAT(
                nfeat=features.shape[1],
                nhid=16,
                heads=16,
                dropout=0.0,
                weight_decay=0e-4,
                nlayers=self.args.nlayers,
                lr=0.001,
                nclass=data.nclass,
                device=device,
                dataset=self.args.dataset,
            ).to(device)
        noval = True if args.dataset in ["reddit", "flickr"] else False
        model.fit(
            features,
            adj,
            labels,
            np.arange(len(features)),
            noval=noval,
            data=data,
            train_iters=10000 if noval else 3000,
            normalize=True,
            verbose=verbose,
            idx_train=idx_train,
        )

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        if args.dataset in ["reddit", "flickr"]:
            output = model.predict(data.feat_test, data.adj_test)
            loss_test = F.nll_loss(output, labels_test)
            acc_test = utils.accuracy(output, labels_test)
            res.append(acc_test.item())
            if verbose:
                print(
                    "Test set results:",
                    "loss= {:.4f}".format(loss_test.item()),
                    "accuracy= {:.4f}".format(acc_test.item()),
                )

        else:
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
        return res

    def train(self, verbose=True):
        args = self.args
        data = self.data

        final_res = {}
        runs = self.args.nruns

        # for model_type in ['GCN', 'GraphSage', 'SGC1', 'MLP', 'APPNP1', 'Cheby', 'GT']:
        for model_type in ['GCN']:
            res = []
            nlayer = 2
            for i in range(runs):
                res.append(self.test_ori(nlayer, verbose=False, model_type=model_type))
            res = np.array(res)
            print('Test/Train Mean Accuracy:',
                    repr([res.mean(0), res.std(0)]))

        # for model_type in ['GAT']:
        #     print(f'=== testing {model_type}')
        #     res = []
        #     nlayer = 2
        #     for i in range(runs):
        #         res.append(self.test_ori_pymodel(verbose=False, nlayers=nlayer, model_type=model_type))
        #     res = np.array(res)
        #     print('Layer:', nlayer)
        #     print('Test/Full Test/Train Mean Accuracy:',
        #             repr([res.mean(0), res.std(0)]))

        # for model_type in ['GCN',  'GraphSage', 'SGC1', 'MLP', 'APPNP1', 'Cheby', 'GT']:
        models_name = [
            # "SGC",
            # "GCN",
            # "GraphSage",
            # "APPNP",
            # "Cheby",
            # "GT",
            # "MLP",
        ]
        results = []
        for model_type in models_name:
            res = []
            nlayer = 2
            for i in range(runs):
                res.append(self.test(nlayer, verbose=False, model_type=model_type))
            res = np.array(res)
            print("Test/Train Mean Accuracy:", repr([res.mean(0), res.std(0)]))
            final_res[model_type] = [res.mean(0), res.std(0)]
            results.append(f"{res.mean(0)[0]*100:.2f} ± {res.std(0)[0]*100:.2f}")

        # for model_type in ["GAT"]:
        #     print(f"=== testing {model_type}")
        #     res = []
        #     nlayer = 2
        #     for i in range(runs):
        #         res.append(
        #             self.test_pymodel(
        #                 verbose=False, nlayers=nlayer, model_type=model_type
        #             )
        #         )
        #     res = np.array(res)
        #     print("Layer:", nlayer)
        #     print(
        #         "Test/Full Test/Train Mean Accuracy:", repr([res.mean(0), res.std(0)])
        #     )
        #     final_res[model_type] = [res.mean(0), res.std(0)]

        print("Final result:", final_res)
        print("\t".join(models_name))
        print("\t".join(results))
