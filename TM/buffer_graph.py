import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0,ROOT_DIR)
from utils.utils_graphset import Dataset as CustomDataset
import numpy as np
import random
import argparse
import torch
from utils.utils import *
import torch.nn.functional as F
import logging
import sys
import datetime
from tensorboardX import SummaryWriter
from deeprobust.graph.utils import accuracy
from networks_gc.gcn import GCN
from torch_geometric.data import Batch
from copy import deepcopy

def get_graphs_multiclass(dataset, args, batch_size, max_node_size=None, idx_herding=None):
    indices_class = {}
    nnodes_all = []
    for ix, single in enumerate(dataset[0]):
        c = single.y.item()
        if c not in indices_class:
            indices_class[c] = [ix]
        else:
            indices_class[c].append(ix)
        nnodes_all.append(single.num_nodes)

    nnodes_all = np.array(nnodes_all)
    real_indices_class = indices_class 
    """get random n graphs from classes"""
    if idx_herding is None:
        if max_node_size is None:
            idx_shuffle = []
            for c in range(dataset[0].num_classes):
                idx_shuffle.append(np.random.permutation(real_indices_class[c])[:batch_size])
            idx_shuffle = np.hstack(idx_shuffle)
            sampled = dataset[4][idx_shuffle]
        else:
            idx_shuffle = []
            for c in range(dataset[0].num_classes):
                indices = np.array(real_indices_class[c])[nnodes_all[real_indices_class[c]] <= max_node_size]
                idx_shuffle.append(np.random.permutation(indices)[:batch_size])
            idx_shuffle = np.hstack(idx_shuffle)
            sampled = dataset[4][idx_shuffle]
    else:
        sampled = dataset[4][idx_herding]
    data = Batch.from_data_list(sampled)
    return data.to(args.device)


def test_pyg_data(data,args):
    dataset = data[0]
    model = GCN(nfeat=dataset.num_features, nconvs=args.nconvs, nhid=args.hidden, nclass=dataset.num_classes, net_norm=args.net_norm, pooling=args.pooling, dropout=args.dropout, args=args).to(args.device)
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    from torch_geometric.loader import DataLoader

    @torch.no_grad()
    def test(loader, report_metric=False):
        model.eval()
        if args.dataset in ['ogbg-molhiv','ogbg-molbbbp', 'ogbg-molbace']:
            pred, y = [], []
            for data in loader:
                data = data.to(args.device)
                pred.append(model(data))
                y.append(data.y.view(-1,1))
            from ogb.graphproppred import Evaluator;
            evaluator = Evaluator(args.dataset)
            return evaluator.eval({'y_pred': torch.cat(pred),
                            'y_true': torch.cat(y)})['rocauc']
        else:
            correct = 0
            for data in loader:
                data = data.to(args.device)
                pred = model(data).max(dim=1)[1]
                correct += pred.eq(data.y.view(-1)).sum().item()
                if report_metric:
                    nnodes_list = [(data.ptr[i]-data.ptr[i-1]).item() for i in range(1, len(data.ptr))]
                    low = np.quantile(nnodes_list, 0.2)
                    high = np.quantile(nnodes_list, 0.8)
                    correct_low = pred.eq(data.y.view(-1))[nnodes_list<=low].sum().item()
                    correct_medium = pred.eq(data.y.view(-1))[(nnodes_list>low)&(nnodes_list<high)].sum().item()
                    correct_high = pred.eq(data.y.view(-1))[nnodes_list>=high].sum().item()
                    print(100*correct_low/(nnodes_list<=low).sum(),
                            100*correct_medium/((nnodes_list>low) & (nnodes_list<high)).sum(),
                            100*correct_high/(nnodes_list>=high).sum())
            return 100*correct / len(loader.dataset)
    best_val_acc = 0
    cls_criterion = torch.nn.BCEWithLogitsLoss()    
    best_val_acc = test(data[2])
    acc_train = test(data[1])
    acc_test = test(data[3], report_metric=False)
    # print([acc_train, best_val_acc, acc_test])
    return [acc_train, best_val_acc, acc_test]

def main(args):
    # random seed setting
    random.seed(args.seed_teacher)
    np.random.seed(args.seed_teacher)
    torch.manual_seed(args.seed_teacher)
    torch.cuda.manual_seed(args.seed_teacher)
    device = torch.device(args.device)
    logging.info('args = {}'.format(args))
    data = CustomDataset(args).packed_data
    dataset = data[0]
    data_real = get_graphs_multiclass(data,args,batch_size=args.bs_cond)
    selected = []
    labels_real = data_real.y

    trajectories = []

    model_type = args.buffer_model_type

    for it in range(0, args.num_experts):
        logging.info(
            '======================== {} -th number of experts for {}-model_type=============================='.format(
                it, model_type))

        model = GCN(nfeat=dataset.num_features, nhid=args.hidden, net_norm=args.net_norm, pooling=args.pooling,
                        dropout=0.0, nclass=dataset.num_classes, nconvs=args.nconvs, args=args).to(args.device)
        # print(model)

        # model.initialize()

        model_parameters = list(model.parameters())

        if args.optim == 'Adam':
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_teacher, weight_decay=args.wd_teacher)
        elif args.optim == 'SGD':
            optimizer_model = torch.optim.SGD(model_parameters, lr=args.lr_teacher, momentum=args.mom_teacher,
                                              weight_decay=args.wd_teacher)

        timestamps = []

        timestamps.append([p.detach().cpu() for p in model.parameters()])

        best_val_acc = best_test_acc = best_it = 0

        if args.dataset!='citeseer':
            lr_schedule = [args.teacher_epochs // 2 + 1]
        else:
            lr_schedule = [600]

        #lr_schedule = [args.teacher_epochs // 2 + 1]
        lr = args.lr_teacher
        # lam = float(args.lam)
        # T = float(args.T)
        # args.lam = lam
        # args.T = T 
        # scheduler = args.scheduler
        for e in range(args.teacher_epochs + 1):
            model.train()
            optimizer_model.zero_grad()
            output = model(data_real)
            loss_buffer = F.nll_loss(output,labels_real)
            acc_buffer = accuracy(output, labels_real)
            writer.add_scalar('buffer_train_loss_curve', loss_buffer.item(), e)
            writer.add_scalar('buffer_train_acc_curve', acc_buffer.item(), e)
            logging.info("Epochs: {} : Full graph train set results: loss= {:.4f}, accuracy= {:.4f} ".format(e,
                                                                                                             loss_buffer.item(),
                                                                                                             acc_buffer.item()))
            loss_buffer.backward()
            optimizer_model.step()


            if e in lr_schedule and args.decay:
                lr = lr*args.decay_factor
                logging.info('NOTE! Decaying lr to :{}'.format(lr))
                if args.optim == 'SGD':
                    optimizer_model = torch.optim.SGD(model_parameters, lr=lr, momentum=args.mom_teacher,weight_decay=args.wd_teacher)
                elif args.optim == 'Adam':
                    optimizer_model = torch.optim.Adam(model_parameters, lr=lr,
                                                       weight_decay=args.wd_teacher)

                optimizer_model.zero_grad()

            if e % 20 == 0:
                logging.info("Epochs: {} : Train set training:, loss= {:.4f}".format(e, loss_buffer.item()))
                model.eval()
                [acc_train, best_val_acc,acc_test] = test_pyg_data(data,args)


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

        logging.info("Valid set best results: accuracy= {:.4f}".format(best_val_acc))
        logging.info("Test set best results: accuracy= {:.4f}".format(best_test_acc))
        # print("Test set best results: accuracy= {:.4f} within best iteration = {}".format(best_test_acc.item(),best_it))
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
    parser.add_argument('--dataset', type=str, default='MUTAG')
    parser.add_argument('--net_norm', type=str, default='none')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--nconvs', type=int, default=3)
    parser.add_argument("--dataset_dir", type=str, default="data", help="Data directory")
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
    parser.add_argument('--ipc', type=int, default=1, help='number of condensed samples per class')
    parser.add_argument('--traj_save_interval', type=int, default=10)
    parser.add_argument('--save_log', type=str, default='logs', help='path to save logs')
    parser.add_argument('--save_trajectories',type=float,default=True, help='whether to save trajectories')
    parser.add_argument('--buffer_model_type', type=str, default='GCN', help='Default buffer_model type')
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'], help='Default buffer_model type')
    parser.add_argument('--decay', type=int, default=0, choices=[1, 0], help='whether to decay lr at 1/2 training epochs')
    parser.add_argument('--decay_factor', type=float, default=0.1, help='decay factor of lr at 1/2 training epochs')
    parser.add_argument('--bs_cond', type=int, default=256, help='batch size for sampling graphs')
    args = parser.parse_args()

    if args.dataset == "ogbg-molhiv":
        args.pooling = "sum"
    if args.dataset == "CIFAR10":
        args.net_norm = "instancenorm"
    if args.dataset == "MUTAG" and args.ipc == 50:
        args.ipc = 20

    log_dir = args.save_log + '/Buffer/{}-buffer'.format(args.dataset)
    if args.method == 'GEOM':
        log_dir = './' + args.save_log + '/Buffer/{}-lam-{}-T-{}-scheduler-{}'.format(args.dataset,args.lam,args.T,args.scheduler)
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
