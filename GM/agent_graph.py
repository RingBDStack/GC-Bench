import os.path as osp
import os
from math import ceil
import torch
import torch.nn.functional as F
from networks_gc.gcn import GCN
from networks_gc.dense_gcn import DenseGCN
from networks_gc.dense_sgc import get_akx
from collections import Counter
import numpy as np
from utils.utils import match_loss, init_feat_c
from utils.utils_graphset import TensorDataset, SparseTensorDataset
from utils.utils_graphset import *
from copy import deepcopy
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score

cls_criterion = torch.nn.BCEWithLogitsLoss()

class GraphAgent:

    def __init__(self, data, args, device, nnodes_syn=75):
        self.data = data
        self.args = args
        self.device = device
        labels_train = [x.y.item() for x in data[0]]

        print('training size:', len(labels_train))
        nfeat = data[0].num_features
        nclass = data[0].num_classes

        self.prepare_train_indices()

        # parametrize syn data
        self.labels_syn = self.get_labels_syn(labels_train)
        if args.ipc == 0:
            n = int(len(labels_train) * args.reduction_rate)
        else:
            self.labels_syn = torch.LongTensor([[i]*args.ipc for i in range(nclass)]).to(device).view(-1)
            self.syn_class_indices = {i: [i*args.ipc, (i+1)*args.ipc] for i in range(nclass)}
            n = args.ipc * nclass

        self.adj_syn = torch.rand(size=(n, nnodes_syn, nnodes_syn), dtype=torch.float, requires_grad=True, device=device)
        self.feat_syn = torch.rand(size=(n, nnodes_syn, nfeat), dtype=torch.float, requires_grad=True, device=device)

        # init feat

        for c in range(nclass):
            ind = self.syn_class_indices[c]
            num_graph = ind[1] - ind[0]
            feat_real, adj_real = self.get_graphs(c, batch_size=num_graph, to_dense=True)
            for i in range(num_graph):
                init_feat = init_feat_c(nnodes_syn, feat_real[i], args.init_way)
                if init_feat is not None:
                    self.feat_syn.data[ind[0]+i] = init_feat

        # init adj
        if args.init == 'real':
            for c in range(nclass):
                ind = self.syn_class_indices[c]
                feat_real, adj_real = self.get_graphs(c, batch_size=ind[1]-ind[0], max_node_size=nnodes_syn, to_dense=True)
                # self.feat_syn.data[ind[0]: ind[1]] = feat_real[:, :nnodes_syn].detach().data
                self.adj_syn.data[ind[0]: ind[1]] = adj_real[:, :nnodes_syn, :nnodes_syn].detach().data
            self.sparsity = self.adj_syn.mean().item()
            if args.stru_discrete:
                self.adj_syn.data.copy_(self.adj_syn*10-5) # max:5; min:-5
        else:
            if args.stru_discrete:
                adj_init = torch.log(self.adj_syn) - torch.log(1-self.adj_syn)
                adj_init = adj_init.clamp(-10, 10)
                self.adj_syn.data.copy_(adj_init)

        print('adj.shape:', self.adj_syn.shape, 'feat.shape:', self.feat_syn.shape)
        self.optimizer_adj = torch.optim.Adam([self.adj_syn], lr=args.lr_adj)
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.weights = []

    def prepare_train_indices(self):
        dataset = self.data[0]
        indices_class = {}
        nnodes_all = []
        for ix, single in enumerate(dataset):
            c = single.y.item()
            if c not in indices_class:
                indices_class[c] = [ix]
            else:
                indices_class[c].append(ix)
            nnodes_all.append(single.num_nodes)

        self.nnodes_all = np.array(nnodes_all)
        self.real_indices_class = indices_class

    def get_labels_syn(self, labels_train):
        counter = Counter(labels_train)
        num_class_dict = {}
        n = len(labels_train)

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
        return torch.LongTensor(labels_syn).to(self.device)

    def get_graphs(self, c, batch_size, max_node_size=None, to_dense=False, idx_selected=None):
        """get random n images from class c"""
        if idx_selected is None:
            if max_node_size is None:
                idx_shuffle = np.random.permutation(self.real_indices_class[c])[:batch_size]
                sampled = self.data[4][idx_shuffle]
            else:
                indices = np.array(self.real_indices_class[c])[self.nnodes_all[self.real_indices_class[c]] <= max_node_size]
                idx_shuffle = np.random.permutation(indices)[:batch_size]
                sampled = self.data[4][idx_shuffle]
        else:
            sampled = self.data[4][idx_selected]
        data = Batch.from_data_list(sampled)
        if to_dense:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x, mask = to_dense_batch(x, batch=batch, max_num_nodes=max_node_size)
            adj = to_dense_adj(edge_index, batch=batch, max_num_nodes=max_node_size)
            return x.to(self.device), adj.to(self.device)
        else:
            return data.to(self.device)

    def get_graphs_multiclass(self, batch_size, max_node_size=None, idx_herding=None):
        """get random n graphs from classes"""
        if idx_herding is None:
            if max_node_size is None:
                idx_shuffle = []
                for c in range(self.data[0].num_classes):
                    idx_shuffle.append(np.random.permutation(self.real_indices_class[c])[:batch_size])
                idx_shuffle = np.hstack(idx_shuffle)
                sampled = self.data[4][idx_shuffle]
            else:
                idx_shuffle = []
                for c in range(self.data[0].num_classes):
                    indices = np.array(self.real_indices_class[c])[self.nnodes_all[self.real_indices_class[c]] <= max_node_size]
                    idx_shuffle.append(np.random.permutation(indices)[:batch_size])
                idx_shuffle = np.hstack(idx_shuffle)
                sampled = self.data[4][idx_shuffle]
        else:
            sampled = self.data[4][idx_herding]
        data = Batch.from_data_list(sampled)
        return data.to(self.device)

    def clip(self):
        self.adj_syn.data.clamp_(min=0, max=1)
        # self.feat_syn.data.clamp_(min=0, max=1)

    def train(self):
        dataset = self.data[0]
        train_loader = self.data[1]
        device = self.device
        args = self.args
        if args.wandb:
            import wandb
            wandb.init(project="GCBM", name=f"{args.method}_{args.dataset}_{args.ipc}_{args.seed}")

        args.outer_loop, args.inner_loop = args.outer, args.inner

        sparsity = self.sparsity

        # EXGC
        previous_loss = 1e10
        epochs_without_improvement = 0
        train_acc_mean = 0
        test_acc_mean = 0

        stop_mask = torch.bernoulli(torch.ones(*self.feat_syn.shape)*self.args.prune).to(self.device)

        import time; st=time.time()
        for it in range(args.epochs):
            wandb_log = {}
            runs = 3
            if it == 0 and args.lr_adj!=0 and args.eval_init:
                print('=== performance before optimizing:')
                res = []
                for _ in range(runs):
                    if args.dataset in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace' ]:
                        res.append(self.test(epochs=500))
                    elif args.dataset in ['DD']:
                        res.append(self.test(epochs=100))
                    else:
                        res.append(self.test(epochs=500))

                res = np.array(res)
                print('Mean Train/Val/TestAcc:', res.mean(0))
                print('Std Train/Val/TestAcc:', res.std(0))

            model_syn = DenseGCN(nfeat=dataset.num_features, nhid=args.hidden, net_norm=args.net_norm, pooling=args.pooling,
                        dropout=0.0, nclass=dataset.num_classes, nconvs=args.nconvs, args=args).to(self.device)
            model_real = GCN(nfeat=dataset.num_features, nhid=args.hidden, net_norm=args.net_norm, pooling=args.pooling,
                        dropout=0.0, nclass=dataset.num_classes, nconvs=args.nconvs, args=args).to(self.device)

            model_real.load_state_dict(model_syn.state_dict())
            model_real_parameters = list(model_real.parameters())
            model_syn_parameters = list(model_syn.parameters())
            optimizer = torch.optim.Adam(model_syn.parameters(), lr=args.lr_model)

            loss_avg = 0
            for ol in range(args.outer_loop):

                BN_flag = False
                bn_real_state = []
                for model in [model_real]:
                    for module in model.modules():
                        if 'BatchNorm' in module._get_name(): #BatchNorm
                            BN_flag = True
                    if BN_flag:
                        data_real = self.get_graphs_multiclass(batch_size=16)
                        model.train() # for updating the mu, sigma of BatchNorm
                        output_real = model(data_real)
                        for module in model.modules():
                            if 'BatchNorm' in module._get_name():  #BatchNorm
                                module.eval() # fix mu and sigma of every BatchNorm layer
                                bn_real_state.append(module.state_dict())

                if BN_flag:
                    model_syn.train() # for updating the mu, sigma of BatchNorm
                    for module in model_syn.modules():
                        ii = 0
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer
                            module.load_state_dict(bn_real_state[ii])
                            ii += 1

                feat_syn = self.feat_syn
                adj_syn = self.adj_syn

                if args.stru_discrete:
                    adj_syn = self.get_discrete_graphs(adj_syn, inference=False)
                loss = 0
                if args.dataset not in ['ogbg-molbace', 'CIFAR10']:
                    for c in range(dataset.num_classes):
                        data_real = self.get_graphs(c, batch_size=args.bs_cond)
                        ind = self.syn_class_indices[c]
                        feat_syn_c = feat_syn[ind[0]:ind[1]]
                        adj_syn_c = adj_syn[ind[0]: ind[1]]

                        labels_real = torch.ones((data_real.y.shape[0],), device=self.device, dtype=torch.long) * c

                        labels_syn = self.labels_syn[ind[0]:ind[1]]
                        output_real = model_real(data_real)
                        if args.dataset in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace']:
                            loss_real = cls_criterion(output_real, labels_real.view(-1, 1).float())
                        else:
                            loss_real = F.nll_loss(output_real, labels_real)
                        gw_real = torch.autograd.grad(loss_real, model_real_parameters)
                        gw_real = list((_.detach().clone() for _ in gw_real))

                        output_syn = model_syn(feat_syn_c, adj_syn_c)
                        if args.dataset in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace']:
                            loss_syn = cls_criterion(output_syn, labels_syn.view(-1, 1).float())
                        else:
                            loss_syn = F.nll_loss(output_syn, labels_syn)
                        gw_syn = torch.autograd.grad(loss_syn, model_syn_parameters, create_graph=True)

                        loss += match_loss(gw_syn, gw_real, args, self.device)
                else:
                    data_real = self.get_graphs_multiclass(batch_size=args.bs_cond)
                    selected = []
                    for c in range(dataset.num_classes):
                        ind = self.syn_class_indices[c]
                        ind = np.arange(ind[0], ind[1])
                        selected.append(ind)

                    selected = np.hstack(selected)
                    feat_syn_c = feat_syn[selected]
                    adj_syn_c = adj_syn[selected]

                    labels_real = data_real.y

                    labels_syn = self.labels_syn[selected]
                    output_real = model_real(data_real)
                    if args.dataset in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace']:
                        loss_real = cls_criterion(output_real, labels_real.view(-1, 1).float())
                    else:
                        loss_real = F.nll_loss(output_real, labels_real)
                    gw_real = torch.autograd.grad(loss_real, model_real_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = model_syn(feat_syn_c, adj_syn_c)
                    if args.dataset in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace']:
                        loss_syn = cls_criterion(output_syn, labels_syn.view(-1, 1).float())
                    else:
                        loss_syn = F.nll_loss(output_syn, labels_syn)
                    gw_syn = torch.autograd.grad(loss_syn, model_syn_parameters, create_graph=True)

                    loss += 1e-0*match_loss(gw_syn, gw_real, args, self.device)

                loss_reg = F.relu(torch.sigmoid(self.adj_syn).mean() - sparsity)
                if args.dataset in ['ogbg-molhiv']:
                    akx = get_akx(feat_syn, adj_syn, K=args.nconvs, pool=args.pooling)
                    nclass = dataset.num_classes
                    first = np.sqrt(2) * loss_avg * nclass
                    second = 3/2/np.sqrt(100) * (nclass-1)/nclass / adj_syn.shape[0] * akx
                    if it % 50==0:
                        print('first:', first , 'second:', second)
                    loss_avg += loss.item()
                    loss = loss + self.args.beta_reg*loss_reg + 1/np.sqrt(2)*second # + 1e-4* torch.norm(self.feat_syn)
                else:
                    loss_avg += loss.item()
                    loss = loss + self.args.beta_reg*loss_reg

                self.optimizer_adj.zero_grad()
                self.optimizer_feat.zero_grad()

                loss.backward()

                self.optimizer_adj.step()
                self.optimizer_feat.step()

                # EXGC
                if args.method == "EXGC":
                    if ol == args.outer_loop - 1:
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


                if not self.args.stru_discrete:
                    self.clip()

                

                if ol == args.outer_loop - 1:
                    break

                self.train_inner(model_syn, model_real, optimizer, epochs=args.inner_loop)

            loss_avg /= (dataset.num_classes*args.outer_loop)

            wandb_log['loss_avg'] = loss_avg
            wandb_log['loss_sparsity'] = loss_reg.item()

            if it % 20 == 0:
                print('Condensation - Iter:', it, 'loss:', loss_avg)
                print('sparsity loss', loss_reg.item())

            if it == 400:
                self.optimizer_adj = torch.optim.Adam([self.adj_syn], lr=0.1*args.lr_adj) # optimizer for synthetic data
                self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=0.1*args.lr_feat) # optimizer for

            if loss_avg < previous_loss:
                previous_loss = loss_avg
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Check if the maximum number of epochs without improvement has been reached
            if args.early_stopping and epochs_without_improvement >= self.args.max_epochs_without_improvement:
                print(f"Training stopped as the loss did not decrease for {self.args.max_epochs_without_improvement} consecutive epochs.")
                break

            print_freq = 10
            if (it+1) % print_freq == 0:
                print('time consumed:', time.time()-st)
                adj_syn2 = self.adj_syn.detach().clone()

                if args.save:
                    torch.save(adj_syn,f"{args.save_dir}/{args.method}/adj_{args.dataset}_{args.ipc}_{args.seed}.pt")
                    torch.save(feat_syn,f"{args.save_dir}/{args.method}/feat_{args.dataset}_{args.ipc}_{args.seed}.pt")

                res = []
                for _ in range(runs):
                    if args.dataset in ['ogbg-molhiv']:
                        res.append(self.test(epochs=100))
                    else:
                        res.append(self.test(epochs=500))
                res = np.array(res)
                wandb_log['train_acc_mean'] = res.mean(0)[0]
                wandb_log['val_acc_mean'] = res.mean(0)[1]
                wandb_log['test_acc_mean'] = res.mean(0)[2]
                if args.wandb:
                    wandb.log(wandb_log)
                print('Mean Train/Val/TestAcc', res.mean(0))
                print('Std Train/Val/TestAcc', res.std(0))


    def test(self, epochs=500, save=False, verbose=False, new_data=None):
        dataset = self.data[0]
        args = self.args
        model_syn = DenseGCN(nfeat=dataset.num_features, nhid=args.hidden, dropout=args.dropout, net_norm=args.net_norm,
                        nconvs=args.nconvs, nclass=dataset.num_classes, pooling=args.pooling, args=args).to(self.device)
        model_real = GCN(nfeat=dataset.num_features, dropout=0.0, net_norm=args.net_norm,
                        nconvs=args.nconvs, nhid=args.hidden, nclass=dataset.num_classes, pooling=args.pooling, args=args).to(self.device)

        if new_data is None:
            feat_syn = self.feat_syn.detach()
            adj_syn = self.adj_syn.detach()
            if args.stru_discrete:
                adj_syn = self.get_discrete_graphs(adj_syn, inference=True)
            # print('Mean sparsity:', (adj_syn.sum(1).sum(1) / adj_syn.size(1) / adj_syn.size(1)).mean().item())
        else:
            feat_syn, adj_syn = new_data
            feat_syn, adj_syn = feat_syn.detach(), adj_syn.detach()

        labels_syn = self.labels_syn

        # Convert adjancency matrix to edge_index stored as torch_geometric.data.Data
        sampled = []
        sampled = np.ndarray((adj_syn.size(0),), dtype=object)
        from torch_geometric.data import Data
        for i in range(adj_syn.size(0)):
            x = feat_syn[i]
            adj = adj_syn[i]
            g = adj.nonzero().T
            y = self.labels_syn[i]
            single_data = Data(x=x, edge_index=g, y=y)
            sampled[i] = (single_data)
        return self.test_pyg_data(sampled, epochs=epochs)

    def test_pyg_data(self, syn_data=None, epochs=500, save=False, verbose=False):
        dataset = self.data[0]
        args = self.args
        use_val = True
        model = GCN(nfeat=dataset.num_features, nconvs=args.nconvs, nhid=args.hidden, nclass=dataset.num_classes, net_norm=args.net_norm, pooling=args.pooling, dropout=args.dropout, args=args).to(self.device)
        lr = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if syn_data is None:
            data = self.adj_syn
        else:
            data = syn_data
        dst_syn_train = SparseTensorDataset(data)

        from torch_geometric.loader import DataLoader
        if args.dataset in ['CIFAR10']:
            train_loader = DataLoader(dst_syn_train, batch_size=512, shuffle=True, num_workers=0)
        else:
            train_loader = DataLoader(dst_syn_train, batch_size=128, shuffle=True, num_workers=0)

        @torch.no_grad()
        def test(loader, report_metric=False):
            model.eval()
            if self.args.dataset in ['ogbg-molhiv','ogbg-molbbbp', 'ogbg-molbace']:
                pred, y = [], []
                for data in loader:
                    data = data.to(self.device)
                    pred.append(model(data))
                    y.append(data.y.view(-1,1))
                from ogb.graphproppred import Evaluator;
                evaluator = Evaluator(self.args.dataset)
                return evaluator.eval({'y_pred': torch.cat(pred),
                             'y_true': torch.cat(y)})['rocauc']
            else:
                correct = 0
                for data in loader:
                    data = data.to(self.device)
                    pred = model(data).max(dim=1)[1]
                    correct += pred.eq(data.y.view(-1)).sum().item()
                    if report_metric:
                        nnodes_list = [(data.ptr[i]-data.ptr[i-1]).item() for i in range(1, len(data.ptr))]
                        low = np.quantile(nnodes_list, 0.2)
                        high = np.quantile(nnodes_list, 0.8)
                        correct_low = pred.eq(data.y.view(-1))[nnodes_list<=low].sum().item()
                        correct_medium = pred.eq(data.y.view(-1))[(nnodes_list>low)&(nnodes_list<high)].sum().item()
                        correct_high = pred.eq(data.y.view(-1))[nnodes_list>=high].sum().item()
                        print(correct_low/(nnodes_list<=low).sum(),
                              correct_medium/((nnodes_list>low) & (nnodes_list<high)).sum(),
                              correct_high/(nnodes_list>=high).sum())
                return correct / len(loader.dataset)

        res = []
        best_val_acc = 0

        for it in range(epochs):
            if it == epochs//2:
                optimizer = torch.optim.Adam(model.parameters(), lr=0.1*lr)

            model.train()
            loss_all = 0
            for data in train_loader:
                data = data.to(self.device)
                y = data.y
                optimizer.zero_grad()
                output = model(data)
                if args.dataset in ['ogbg-molhiv','ogbg-molbbbp', 'ogbg-molbace']:
                    loss = cls_criterion(output, y.view(-1, 1).float())
                else:
                    loss = F.nll_loss(output, y.view(-1))
                loss.backward()
                loss_all += y.size(0) * loss.item()
                optimizer.step()

            loss = loss_all / len(dst_syn_train)
            if verbose:
                if it % 100 == 0:
                    print('Evaluation Stage - loss:', loss)

            if use_val:
                acc_val = test(self.data[2])
                if acc_val > best_val_acc:
                    best_val_acc = acc_val
                    if verbose:
                        acc_train = test(self.data[1])
                        acc_test = test(self.data[3], report_metric=False)
                        print('acc_train:', acc_train, 'acc_val:', acc_val, 'acc_test:', acc_test)
                    if save:
                        torch.save(model.state_dict(), f'{args.save_dir}/{args.method}/{args.dataset}_{args.seed}.pt')
                    weights = deepcopy(model.state_dict())

        if use_val:
            model.load_state_dict(weights)
        else:
            best_val_acc = test(self.data[2])
        acc_train = test(self.data[1])
        acc_test = test(self.data[3], report_metric=False)
        # print([acc_train, best_val_acc, acc_test])
        return [acc_train, best_val_acc, acc_test]

    def train_inner(self, model_syn, model_real, optimizer, epochs=500, save=False, verbose=False):
        if epochs == 0:
            return
        dataset = self.data[0]
        args = self.args
        feat_syn = self.feat_syn.detach()
        adj_syn = self.adj_syn
        adj_syn = adj_syn.detach()
        labels_syn = self.labels_syn
        dst_syn_train = TensorDataset(feat_syn, adj_syn, labels_syn)
        train_loader = torch.utils.data.DataLoader(dst_syn_train, batch_size=128, shuffle=True, num_workers=0)

        for it in range(epochs):
            model_syn.train()
            loss_all = 0
            for data in train_loader:
                x, adj, y = data
                x, adj, y = x.to(self.device), adj.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = model_syn(x, adj, mask=None)
                if args.dataset in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace']:
                    loss = cls_criterion(output, y.view(-1, 1).float())
                else:
                    loss = F.nll_loss(output, y.view(-1))
                loss.backward()
                optimizer.step()
        model_real.load_state_dict(model_syn.state_dict())

    def test_full_train(self, epochs=500, save=False, verbose=False):
        dataset = self.data[0]
        use_val = True
        args = self.args
        model = GCN(nfeat=dataset.num_features, nconvs=args.nconvs, nhid=args.hidden, nclass=dataset.num_classes, net_norm=args.net_norm, pooling=args.pooling, dropout=args.dropout, args=args).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_loader = self.data[1]

        @torch.no_grad()
        def test(loader, report_metric=False):
            model.eval()
            if self.args.dataset in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace']:
                pred, y = [], []
                for data in loader:
                    data = data.to(self.device)
                    pred.append(model(data))
                    y.append(data.y.view(-1,1))
                from ogb.graphproppred import Evaluator;
                evaluator = Evaluator(self.args.dataset)
                return evaluator.eval({'y_pred': torch.cat(pred),
                             'y_true': torch.cat(y)})['rocauc']
            else:
                correct = 0
                for data in loader:
                    data = data.to(self.device)
                    pred = model(data).max(dim=1)[1]
                    correct += pred.eq(data.y.view(-1)).sum().item()
                    if report_metric:
                        nnodes_list = [(data.ptr[i]-data.ptr[i-1]).item() for i in range(1, len(data.ptr))]
                        low = np.quantile(nnodes_list, 0.2)
                        high = np.quantile(nnodes_list, 0.8)
                        correct_low = pred.eq(data.y.view(-1))[nnodes_list<=low].sum().item()
                        correct_medium = pred.eq(data.y.view(-1))[(nnodes_list>low)&(nnodes_list<high)].sum().item()
                        correct_high = pred.eq(data.y.view(-1))[nnodes_list>=high].sum().item()
                        print(correct_low/(nnodes_list<=low).sum(),
                              correct_medium/((nnodes_list>low) & (nnodes_list<high)).sum(),
                              correct_high/(nnodes_list>=high).sum())
                return correct / len(loader.dataset)

        res = []
        best_val_acc = 0

        for it in range(epochs):
            if it == epochs//2:
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            model.train()
            loss_all = 0
            for data in train_loader:
                data = data.to(self.device)
                y = data.y
                optimizer.zero_grad()
                if self.args.augment:
                    x = F.dropout(x)
                output = model(data)
                if args.dataset in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace']:
                    loss = cls_criterion(output, y.view(-1, 1).float())
                else:
                    loss = F.nll_loss(output, y.view(-1))
                loss.backward()
                loss_all += y.size(0) * loss.item()
                optimizer.step()

            loss = loss_all / len(self.data[0])
            if verbose:
                if it % 100 == 0:
                    print('Evaluation Stage - loss:', loss)

            if use_val:
                acc_val = test(self.data[2])
                if acc_val > best_val_acc:
                    best_val_acc = acc_val
                    if verbose:
                        acc_train = test(self.data[1])
                        acc_test = test(self.data[3], report_metric=True)
                        print('acc_train:', acc_train, 'acc_val:', acc_val, 'acc_test:', acc_test)
                    if save:
                        torch.save(model.state_dict(), f'{args.save_dir}/{args.method}/model_{args.dataset}_{args.seed}.pt')
                    weights = deepcopy(model.state_dict())

        if use_val:
            model.load_state_dict(weights)
            acc_train = test(self.data[1])
            acc_test = test(self.data[3], report_metric=True)

        @torch.no_grad()
        def get_embeds(loader):
            model.eval()
            all_emb = []
            for data in loader:
                data = data.to(self.device)
                emb = model.embed(data)
                all_emb.append(emb)
            return torch.cat(all_emb, dim=0)

        # don't shuffle training data
        new_train_loader = DataLoader(self.data[0], batch_size=1024, shuffle=False)
        embeds = get_embeds(new_train_loader)
        return embeds.cpu()

    def get_discrete_graphs(self, adj, inference):
        if not hasattr(self, 'cnt'):
            self.cnt = 0

        if self.args.dataset not in ['CIFAR10']:
            adj = (adj.transpose(1,2) + adj) / 2

        if not inference:
            N = adj.size()[1]
            vals = torch.rand(adj.size(0) * N * (N+1) // 2)
            vals = vals.view(adj.size(0), -1).to(self.device)
            i, j = torch.triu_indices(N, N)
            epsilon = torch.zeros_like(adj)
            epsilon[:, i, j] = vals
            epsilon.transpose(1,2)[:, i, j] = vals

            tmp = torch.log(epsilon) - torch.log(1-epsilon)
            self.tmp = tmp
            adj = tmp + adj
            t0 = 1
            tt = 0.01
            end_iter = 200
            t = t0*(tt/t0)**(self.cnt/end_iter)
            if self.cnt == end_iter:
                print('===reached the end of anealing...')
            self.cnt += 1

            t = max(t, tt)
            adj = torch.sigmoid(adj/t)
            adj = adj * (1-torch.eye(adj.size(1)).to(self.device))
        else:
            adj = torch.sigmoid(adj)
            adj = adj * (1-torch.eye(adj.size(1)).to(self.device))
            adj[adj> 0.5] = 1
            adj[adj<= 0.5] = 0
        return adj


