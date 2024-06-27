import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils.utils import *
import torch.nn.functional as F
from utils.utils_graph import DataGraph
import logging
import datetime
from tensorboardX import SummaryWriter
import deeprobust.graph.utils as utils
from itertools import repeat
from networks_nc.gat import GAT,Dpr2Pyg
from networks_nc.gcn import GCN
from networks_nc.sgc import SGC
from utils.utils_graph import GraphData

def main(args):
    # random seed setting
    random.seed(args.seed_teacher)
    np.random.seed(args.seed_teacher)
    torch.manual_seed(args.seed_teacher)
    torch.cuda.manual_seed(args.seed_teacher)
    device = torch.device(args.device)
    logging.info('args = {}'.format(args))

    data_graphsaint = ['flickr', 'reddit']
    if args.dataset in data_graphsaint:
        data = DataGraph(args.dataset)
        data_full = data.data_full
    else:
        data_full = get_dataset(args.dataset,dataset_dir=args.data_dir)
        data = Transd2Ind(data_full)

    features, adj, labels = data.feat_train, data.adj_train, data.labels_train
    features_sort, adj_sort, labels_sort = data.feat_full, data.adj_full, data.labels_full

    adj, features, labels = utils.to_tensor(adj, features, labels, device=device)
    adj_sort, features_sort, labels_sort = utils.to_tensor(adj_sort, features_sort, labels_sort, device=device)

    feat_val, adj_val, labels_val = data.feat_val, data.adj_val, data.labels_val
    adj_val, feat_val, labels_val = utils.to_tensor(adj_val, feat_val, labels_val,device=device)

    feat_test, adj_test, labels_test = data.feat_test, data.adj_test, data.labels_test
    adj_test, feat_test, labels_test = utils.to_tensor(adj_test, feat_test, labels_test, device=device)

    if utils.is_sparse_tensor(adj):
        adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
    else:
        adj_norm = utils.normalize_adj_tensor(adj)
    
    if utils.is_sparse_tensor(adj_sort):
        adj_norm_sort = utils.normalize_adj_tensor(adj_sort, sparse=True)
    else:
        adj_norm_sort = utils.normalize_adj_tensor(adj_sort)

    adj = adj_norm
    adj_sort = adj_norm_sort

    trajectories = []

    model_type = args.buffer_model_type
    sorted_trainset = sort_training_nodes_in(data, adj, labels)

    for it in range(0, args.num_experts):
        logging.info(
            '======================== {} -th number of experts for {}-model_type=============================='.format(
                it, model_type))

        model_class = eval(model_type)

        model = model_class(nfeat=features.shape[1], nhid=args.teacher_hidden, dropout=args.teacher_dropout,
                            nlayers=args.teacher_nlayers,
                            nclass=data.nclass, device=device).to(device)

        model.initialize()

        model_parameters = list(model.parameters())

        if args.optim == 'Adam':
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_teacher, weight_decay=args.wd_teacher)
        elif args.optim == 'SGD':
            optimizer_model = torch.optim.SGD(model_parameters, lr=args.lr_teacher, momentum=args.mom_teacher,
                                              weight_decay=args.wd_teacher)

        timestamps = []

        timestamps.append([p.detach().cpu() for p in model.parameters()])

        best_val_acc = best_test_acc = 0

        lr_schedule = [args.teacher_epochs // 2 + 1]

        lr = args.lr_teacher
        lam = float(args.lam)
        T = float(args.T)
        scheduler = args.scheduler

        for e in range(args.teacher_epochs + 1):
            model.train()
            optimizer_model.zero_grad()
            output = model.forward(features, adj)
            size = training_scheduler(args.lam, e, T, scheduler)
            if args.method == "GEOM":
                print(size)
                training_subset = sorted_trainset[:int(size * sorted_trainset.shape[0])]
                loss_buffer = F.nll_loss(output[training_subset], labels[training_subset])
            else:
                loss_buffer = F.nll_loss(output, labels)
            acc_buffer = utils.accuracy(output, labels)
            writer.add_scalar('buffer_train_loss_curve', loss_buffer.item(), e)
            writer.add_scalar('buffer_train_acc_curve', acc_buffer.item(), e)
            logging.info("Epochs: {} : Full graph train set results: loss= {:.4f}, accuracy= {:.4f} ".format(e,
                                                                                                             loss_buffer.item(),
                                                                                                             acc_buffer.item()))
            loss_buffer.backward()
            optimizer_model.step()

            if e in lr_schedule and args.decay:
                lr = args.lr_teacher*args.decay_factor
                logging.info('NOTE! Decaying lr to :{}'.format(lr))
                if args.optim == 'SGD':
                    optimizer_model = torch.optim.SGD(model_parameters, lr=lr, momentum=args.mom_teacher,weight_decay=args.wd_teacher)
                elif args.optim == 'Adam':
                    optimizer_model = torch.optim.Adam(model_parameters, lr=lr,
                                                       weight_decay=args.wd_teacher)

                optimizer_model.zero_grad()

            if e % 1 == 0:
                #logging.info("Epochs: {} : Train set training:, loss= {:.4f}".format(e, loss_buffer.item()))
                model.eval()
                output_val = model.predict(feat_val, adj_val)
                loss_val = F.nll_loss(output_val, labels_val)
                acc_val = utils.accuracy(output_val, labels_val)

                writer.add_scalar('val_set_loss_curve', loss_val.item(), e)
                writer.add_scalar('val_set_acc_curve', acc_val.item(), e)

                # Full graph
                output_test = model.predict(feat_test, adj_test)
                loss_test = F.nll_loss(output_test, labels_test)
                acc_test = utils.accuracy(output_test, labels_test)
                logging.info('test_acc = {}'.format(acc_test))

                writer.add_scalar('test_set_loss_curve', loss_test.item(), e)
                writer.add_scalar('test_set_acc_curve', acc_test.item(), e)

                if acc_val > best_val_acc:
                    best_val_acc = acc_val
                    best_test_acc = acc_test
                    best_it  = e
                    logging.info('best_test_acc = {} within {}-th iteration'.format(best_test_acc,best_it))
                    # print('best_test_acc = {} within {}-th iteration'.format(best_test_acc,best_it))

            if e % args.param_save_interval == 0 and e>1:
                timestamps.append([p.detach().cpu() for p in model.parameters()])
                p_current = timestamps[-1]
                p_0 = timestamps[0]
                target_params = torch.cat([p_c.data.reshape(-1) for p_c in p_current], 0)
                starting_params = torch.cat([p0.data.reshape(-1) for p0 in p_0], 0)
                param_dist1 = torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
                writer.add_scalar('param_change', param_dist1.item(), e)
                logging.info(
                    '==============================={}-th iter with length of {}-th tsp'.format(e, len(timestamps)))

        logging.info("Valid set best results: accuracy= {:.4f}".format(best_val_acc.item()))
        logging.info("Test set best results: accuracy= {:.4f}".format(best_test_acc.item()))
        trajectories.append(timestamps)

        if len(trajectories) == args.traj_save_interval:
            n = 0
            while os.path.exists(os.path.join(log_dir, f"{args.method}_replay_buffer_{n}.pt")):
                n += 1
            logging.info("Saving {}".format(os.path.join(log_dir, f"{args.method}_replay_buffer_{n}.pt")))
            if args.save_trajectories:
                torch.save(trajectories, os.path.join(log_dir, f"{args.method}_replay_buffer_{n}.pt"))
            trajectories = []



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--method',type=str, default='SFGC')
    parser.add_argument('--dataset', type=str,default='flickr')
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument('--teacher_epochs', type=int, default=1000, help='training epochs')
    parser.add_argument('--teacher_nlayers', type=int, default=2)
    parser.add_argument('--teacher_hidden', type=int, default=256)
    parser.add_argument('--teacher_dropout', type=float, default=0.0)
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for buffer learning rate')
    parser.add_argument('--wd_teacher', type=float, default=0)
    parser.add_argument('--mom_teacher', type=float, default=0)
    parser.add_argument('--seed_teacher', type=int, default=15, help='Random seed.')
    parser.add_argument('--num_experts', type=int, default=200, help='training iterations')
    parser.add_argument('--param_save_interval', type=int, default=10)
    parser.add_argument('--traj_save_interval', type=int, default=10)
    parser.add_argument('--save_log', type=str, default='logs', help='path to save logs')
    parser.add_argument('--save_trajectories',type=float,default=True, help='whether to save trajectories')
    parser.add_argument('--buffer_model_type', type=str, default='GCN', help='Default buffer_model type')
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'], help='Default buffer_model type')
    parser.add_argument('--decay', type=int, default=0, choices=[1, 0], help='whether to decay lr at 1/2 training epochs')
    parser.add_argument('--decay_factor', type=float, default=0.1, help='decay factor of lr at 1/2 training epochs')
    parser.add_argument('--lam', type=float, default=0.70)
    parser.add_argument('--T', type=int, default=200)
    parser.add_argument('--scheduler', type=str, default='root')
    args = parser.parse_args()

    # log_dir = './' + args.save_log + '/Buffer/{}-{}'.format(args.dataset,
    #                                                         datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    log_dir = './' + args.save_log + '/Buffer/{}-buffer'.format(args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('This is the log_dir: {}'.format(log_dir))
    writer = SummaryWriter(log_dir + '/tbx_log')
    main(args)
    logging.info(args)
    logging.info('Finish!, Log_dir: {}'.format(log_dir))
