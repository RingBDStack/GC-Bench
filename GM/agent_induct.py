import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
from utils.utils import match_loss, regularization, row_normalize_tensor, init_feat
import deeprobust.graph.utils as utils
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from networks_nc.gcn import GCN
from networks_nc.sgc import SGC
from networks_nc.sgc_multi import SGC as SGC1
from networks_nc.parametrized_adj import PGE
from networks_nc.IGNR import GraphonLearner as IGNR
from utils.init_coreset import init_coreset_inductive as init_coreset
import scipy.sparse as sp
from torch_sparse import SparseTensor

class GCond:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device

        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)
        # n = int(len(data.idx_train) * args.reduction_rate)
        n = max(int(len(data.idx_train) * args.reduction_rate), self.labels_syn.shape[0])
        d = data.feat_train.shape[1]
        print(f'target reduced size:{int(len(data.idx_train) * args.reduction_rate)}')
        print(f'actual reduced size:{n}')
        self.nnodes_syn = n
        self.feat_syn = torch.FloatTensor(n, d).to(device)

        if args.method == "SGDD":
            self.pge = IGNR(node_feature=d, nfeat=128, nnodes=n, device=device, args=args
                    ).to(device)
        else:
            self.pge = PGE(nfeat=d, nnodes=n, device=device, args=args).to(device)

        self.reset_parameters()
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        print('adj_syn:', (n,n), 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def generate_labels_syn(self, data):
        from collections import Counter
        counter = Counter(data.labels_train)
        num_class_dict = {}
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x:x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}

        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn

    def test_with_val(self, verbose=True):
        res = []

        data, device = self.data, self.device
        feat_syn, pge, labels_syn = self.feat_syn.detach(), \
                                self.pge, self.labels_syn
        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        dropout = 0.5 if self.args.dataset in ['reddit'] else 0
        model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=dropout,
                    weight_decay=5e-4, nlayers=2,
                    nclass=data.nclass, device=device).to(device)

        adj_syn = pge.inference(feat_syn)
        args = self.args

        if args.save:
            torch.save(adj_syn, f'{args.save_dir}/{args.method}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
            torch.save(feat_syn, f'{args.save_dir}/{args.method}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')

        noval = True
        model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                     train_iters=600, normalize=True, verbose=False, noval=noval)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        output = model.predict(data.feat_test, data.adj_test)

        loss_test = F.nll_loss(output, labels_test)
        acc_test = utils.accuracy(output, labels_test)
        res.append(acc_test.item())
        if verbose:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
        print(adj_syn.sum(), adj_syn.sum()/(adj_syn.shape[0]**2))

        if False:
            if self.args.dataset == 'ogbn-arxiv':
                thresh = 0.6
            elif self.args.dataset == 'reddit':
                thresh = 0.91
            else:
                thresh = 0.7

            labels_train = torch.LongTensor(data.labels_train).cuda()
            output = model.predict(data.feat_train, data.adj_train)
            # loss_train = F.nll_loss(output, labels_train)
            # acc_train = utils.accuracy(output, labels_train)
            loss_train = torch.tensor(0)
            acc_train = torch.tensor(0)
            if verbose:
                print("Train set results:",
                      "loss= {:.4f}".format(loss_train.item()),
                      "accuracy= {:.4f}".format(acc_train.item()))
            res.append(acc_train.item())
        return res

    def train(self, verbose=True):
        args = self.args
        data = self.data
        if args.wandb:
            import wandb
            wandb.init(project="GCBM", name=f"{args.method}_{args.dataset}_{args.reduction_rate}_{args.seed}")
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn
        features, adj, labels = data.feat_train, data.adj_train, data.labels_train
        syn_class_indices = self.syn_class_indices
        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        feat_sub, adj_sub = self.get_sub_adj_feat(features)
        self.feat_syn.data.copy_(feat_sub)
        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)

        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                value=adj._values(), sparse_sizes=adj.size()).t()


        outer_loop, inner_loop = get_loops(args)
        if 'coreset' in args.init_way:
            idx_selected = init_coreset(data, args)
            feat_syn = features[idx_selected]
            self.labels_syn = labels[idx_selected]
        else:
            feat_syn, feature_init = init_feat(feat_syn, args, data, syn_class_indices, transductive=False)
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        loss_avg = 0
        # EXGC
        previous_loss = 1e10
        epochs_without_improvement = 0
        # train_acc_mean = 0
        test_acc_mean = 0

        stop_mask = torch.bernoulli(torch.ones(*feat_syn.shape)*self.args.prune).to(self.device)

        for it in range(args.epochs+1):
            wandb_log = {}
            loss_avg = 0
            if args.sgc==1:
                model = SGC(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout,
                            nlayers=args.nlayers, with_bn=False,
                            device=self.device).to(self.device)
            elif args.sgc==2:
                model = SGC1(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout,
                            nlayers=args.nlayers, with_bn=False,
                            device=self.device).to(self.device)

            else:
                model = GCN(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout, nlayers=args.nlayers,
                            device=self.device).to(self.device)

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
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    model.train() # for updating the mu, sigma of BatchNorm
                    output_real = model.forward(features, adj_norm)
                    for module in model.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer

                loss = torch.tensor(0.0).to(self.device)
                for c in range(data.nclass):
                    if c not in self.num_class_dict:
                        continue

                    batch_size, n_id, adjs = data.retrieve_class_sampler(
                            c, adj, transductive=False, args=args)

                    if args.nlayers == 1:
                        adjs = [adjs]
                    adjs = [adj.to(self.device) for adj in adjs]
                    output = model.forward_sampler(features[n_id], adjs)
                    loss_real = F.nll_loss(output, labels[n_id[:batch_size]])
                    gw_real = torch.autograd.grad(loss_real, model_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    ind = syn_class_indices[c]
                    if args.nlayers == 1:
                        adj_syn_norm_list = [adj_syn_norm[ind[0]: ind[1]]]
                    else:
                        adj_syn_norm_list = [adj_syn_norm]*(args.nlayers-1) + \
                                [adj_syn_norm[ind[0]: ind[1]]]

                    output_syn = model.forward_sampler_syn(feat_syn, adj_syn_norm_list)
                    loss_syn = F.nll_loss(output_syn, labels_syn[ind[0]: ind[1]])

                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                    loss += coeff  * match_loss(gw_syn, gw_real, args, device=self.device)

                loss_avg += loss.item()
                # TODO: regularize
                if args.alpha > 0:
                    loss_reg = args.alpha * regularization(adj_syn, utils.tensor2onehot(labels_syn))
                # else:
                else:
                    loss_reg = torch.tensor(0)

                loss = loss + loss_reg

                if args.opt_scale > 0:                          # if not SGDD, opt_scale is zero
                    loss_opt = args.opt_scale * opt_loss
                else:
                    loss_opt = torch.tensor(0)

                loss = loss + loss_opt

                wandb_log['loss_opt'] = loss_opt

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
                    if ol == outer_loop-1:
                        zero_indices = (stop_mask == 0).nonzero()
                        num_zeros = zero_indices.size(0)
                        if num_zeros !=0:
                            current_zeros_ratio = num_zeros / stop_mask.numel()

                            num_elements = stop_mask.numel()

                            num_to_convert=int(num_elements/self.args.circulation)
                            indices_to_convert = zero_indices[torch.randperm(num_zeros)[:num_to_convert]]

                            stop_mask[indices_to_convert[:, 0], indices_to_convert[:, 1]] = 1
                            print('it', it,'num_zeros',num_zeros, 'current_zeros_ratio', current_zeros_ratio,
                                'num_to_convert', num_to_convert,'nonzero',torch.nonzero(stop_mask).size(0))

                    feat_syn.data = feat_syn.data * stop_mask


                if args.debug and ol % 5 ==0:
                    print('Gradient matching loss:', loss.item())

                if ol == outer_loop - 1:
                    # print('loss_reg:', loss_reg.item())
                    # print('Gradient matching loss:', loss.item())
                    break


                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn)
                adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    optimizer_model.step() # update gnn param

            loss_avg /= (data.nclass*outer_loop)
            if it % 50 == 0:
                print('Epoch {}, loss_avg: {}'.format(it, loss_avg))

            eval_epochs = [50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1600, 2000, 3000, 4000, 5000]
            if verbose and it in eval_epochs:
            # if verbose and (it+1) % 500 == 0:
                res = []
                runs = 1 if args.dataset in ['ogbn-arxiv', 'reddit', 'flickr'] else 3
                for i in range(runs):
                    # self.test()
                    res.append(self.test_with_val())
                res = np.array(res)
                print('Test:',
                        repr([res.mean(0), res.std(0)]))
                # train_acc_mean = res.mean(0)[0]
                test_acc_mean = res.mean(0)[0]
            
            wandb_log['loss_avg'] = loss_avg
            # wandb_log['train_acc_mean'] = train_acc_mean
            wandb_log['test_acc_mean'] = test_acc_mean
            wandb.log(wandb_log)

            if loss_avg < previous_loss:
                previous_loss = loss_avg
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Check if the maximum number of epochs without improvement has been reached
            if args.early_stopping and epochs_without_improvement >= self.args.max_epochs_without_improvement:
                print(f"Training stopped as the loss did not decrease for {self.args.max_epochs_without_improvement} consecutive epochs.")
                break




    def get_sub_adj_feat(self, features):
        data = self.data
        args = self.args
        idx_selected = []

        from collections import Counter;
        counter = Counter(self.labels_syn.cpu().numpy())

        for c in range(data.nclass):
            tmp = data.retrieve_class(c, num=counter[c])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[idx_selected]

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
            sims[i, indices_argsort[: -k]] = 0
        adj_knn = torch.FloatTensor(sims).to(self.device)
        return features, adj_knn


def get_loops(args):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if args.one_step:
        return 10, 0

    if args.dataset in ['ogbn-arxiv']:
        return 20, 0
    if args.dataset in ['reddit']:
        return args.outer, args.inner
    if args.dataset in ['flickr']:
        return args.outer, args.inner
        # return 10, 1
    if args.dataset in ['cora']:
        return 20, 10
    if args.dataset in ['citeseer']:
        return 20, 5 # at least 200 epochs
    else:
        return 20, 5

