import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
import numpy as np
import random
import time
import argparse
import torch
import torch.nn.functional as F
import os
import datetime
import deeprobust.graph.utils as utils
from networks_gc.gcn import GCN
from networks_nc.sgc import SGC
from networks_nc.sgc_multi import SGC as SGC1
from networks_nc.myappnp import APPNP
from networks_nc.myappnp1 import APPNP1
from networks_nc.mycheby import Cheby
from networks_nc.mygraphsage import GraphSage
from networks_nc.gat import GAT
import scipy.sparse as sp 
from utils.utils_graph import DataGraph
from utils.utils import *
from utils.utils_graphset import SparseTensorDataset
from utils.utils_graphset import Dataset as CustomDataset
from gntk_cond import GNTK
import logging
from tensorboardX import SummaryWriter
from sklearn.neighbors import kneighbors_graph
import json
from copy import deepcopy
import wandb


# random seed setting
def main(args, writer):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(args.device)
    #logging.info('start!')
    if args.dataset in ['cora', 'citeseer']:
        args.epsilon = 0.05
    else:
        args.epsilon = 0.01

    data = CustomDataset(args)
    packed_data = data.packed_data

    res_val = []
    res_test = []
    nlayer = 2
    acc_final = 0
    for i in range(args.nruns):
        for it in range((args.ITER + 1)//50):
            acc_train, best_acc_val, best_acc_test = test(packed_data, args, it)
            if best_acc_test > acc_final:
                acc_final = best_acc_test
            res_val.append(best_acc_val)
            res_test.append(best_acc_test)
    res_val = np.array(res_val)
    res_test = np.array(res_test)
    logging.info('TEST: Full Graph Mean Accuracy: {:.6f}, STD: {:.6f}'.format(res_test.mean(), res_test.std()))
    logging.info('TEST: Valid Graph Mean Accuracy: {:.6f}, STD: {:.6f}'.format(res_val.mean(), res_val.std()))

    if args.wandb:
        wandb.log({'test_acc_mean': res_test.mean(), 'test_acc_std': res_test.std(),
               'val_acc_mean': res_val.mean(), 'val_acc_std': res_val.std()})
    print("BEST TEST:{acc_final:.6f}".format(acc_final=acc_final))
    return best_acc_val, best_acc_test, args

def test(real_data, args, it, epochs=500, save=False, verbose=False,new_labels=None):
    args = args

    feat_syn, labels_syn = get_syn_data(args,args.device, it)
    feat_syn, labels_syn = feat_syn.detach(), labels_syn.detach()

    adj_syn = torch.eye(feat_syn.size(1)).to(args.device).repeat(feat_syn.size(0),1,1)
    labels_syn = labels_syn

    # Convert adjancency matrix to edge_index stored as torch_geometric.data.Data
    sampled = []
    sampled = np.ndarray((adj_syn.size(0),), dtype=object)
    from torch_geometric.data import Data
    for i in range(adj_syn.size(0)):
        x = feat_syn[i]
        adj = adj_syn[i]
        g = adj.nonzero().T
        y = labels_syn[i]
        single_data = Data(x=x, edge_index=g, y=y)
        sampled[i] = (single_data)
    return test_pyg_data(real_data, args, sampled, device=args.device, epochs=epochs)


def test_pyg_data(real_data, args, syn_data, device, epochs=500, save=False, verbose=False):
    cls_criterion = torch.nn.BCEWithLogitsLoss()
    dataset = real_data[0]
    args = args
    use_val = True
    model = GCN(nfeat=dataset.num_features, nconvs=args.nconvs, nhid=args.hidden, nclass=dataset.num_classes, net_norm=args.net_norm, pooling=args.pooling, dropout=args.dropout, args=args).to(device)
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
        if args.dataset in ['ogbg-molhiv','ogbg-molbbbp', 'ogbg-molbace']:
            pred, y = [], []
            for data in loader:
                data = data.to(device)
                pred.append(model(data))
                y.append(data.y.view(-1,1))
            from ogb.graphproppred import Evaluator;
            evaluator = Evaluator(args.dataset)
            return evaluator.eval({'y_pred': torch.cat(pred),
                            'y_true': torch.cat(y)})['rocauc']
        else:
            correct = 0
            for data in loader:
                data = data.to(device)
                pred = model(data).max(dim=1)[1]
                correct += pred.eq(data.y.view(-1)).sum().item()
                if report_metric:
                    nnodes_list = [(data.ptr[i]-data.ptr[i-1]).item() for i in range(1, len(data.ptr))]
                    low = np.quantile(nnodes_list, 0.2)
                    high = np.quantile(nnodes_list, 0.8)
                    correct_low = pred.eq(data.y.view(-1))[nnodes_list<=low].sum().item()
                    correct_medium = pred.eq(data.y.view(-1))[(nnodes_list>low)&(nnodes_list<high)].sum().item()
                    correct_high = pred.eq(data.y.view(-1))[nnodes_list>=high].sum().item()
                    # print(100*correct_low/(nnodes_list<=low).sum(),
                    #         100*correct_medium/((nnodes_list>low) & (nnodes_list<high)).sum(),
                    #         100*correct_high/(nnodes_list>=high).sum())
            return 100*correct / len(loader.dataset)

    res = []
    best_val_acc = 0

    for it in range(epochs):
        if it == epochs//2:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1*lr)

        model.train()
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
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
            acc_val = test(real_data[2])
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                if verbose:
                    acc_train = test(real_data[1])
                    acc_test = test(real_data[3], report_metric=False)
                    print('acc_train:', acc_train, 'acc_val:', acc_val, 'acc_test:', acc_test)
                if save:
                    torch.save(model.state_dict(), f'saved/{args.dataset}_{args.seed}.pt')
                weights = deepcopy(model.state_dict())

    if use_val:
        model.load_state_dict(weights)
    else:
        best_val_acc = test(real_data[2])
    acc_train = test(real_data[1])
    acc_test = test(real_data[3], report_metric=False)
    # print([acc_train, best_val_acc, acc_test])
    return [acc_train, best_val_acc, acc_test]


def get_syn_data(args, device, it, model_type=None):
    # if args.best_ntk_score==1:
    feat_syn = torch.load(f'{args.log_dir}/feat_{args.dataset}_{args.reduction_rate}_{it*50}_{args.seed}.pt',
                            map_location='cpu')
    labels_syn = torch.load(
        f'{args.log_dir}/label_{args.dataset}_{args.reduction_rate}_{it*50}_{args.seed}.pt',
        map_location='cpu')

    feat_syn = feat_syn.to(device)
    labels_syn = labels_syn.to(device)

    return feat_syn, labels_syn


def test_graph(args):

    log_dir = './' + args.save_log + '/Test/{}-model_{}'.format(args.dataset,
                                                                             str(args.reduction_rate))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'test.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('This is the log_dir: {}'.format(log_dir))
    writer = SummaryWriter(log_dir + '/tbx_log')
    main(args, writer)
    logging.info(args)
    logging.info('Finish!, Log_dir: {}'.format(log_dir))
