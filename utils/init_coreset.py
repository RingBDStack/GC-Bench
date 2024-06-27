import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
import random
import argparse
import logging
# from configs import load_config
from utils.utils import *
from utils.utils_graph import DataGraph
from networks_nc.gcn import GCN
from coreset import KCenter, Herding, Random, Center, KMeans
from tqdm import tqdm
import torch
import deeprobust.graph.utils as utils
import datetime
import numpy as np
import torch.nn.functional as F

def init_coreset(data,args):
    device = torch.device(args.device)
    log_dir = args.coreset_log + '/Coreset/{}-reduce'.format(args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'coreset.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('This is the log_dir: {}'.format(log_dir))

    features, adj, labels = data.feat_full, data.adj_full, data.labels_full
    adj, features, labels = utils.to_tensor(adj, features, labels, device=device)
    adj, features, labels = adj.to(device), features.to(device), labels.to(device)
    if utils.is_sparse_tensor(adj):
        adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
    else:
        adj_norm = utils.normalize_adj_tensor(adj)

    adj = adj_norm.to(device)
    idx_train = data.idx_train
    idx_val = data.idx_val
    idx_test = data.idx_test
    labels_test = labels[data.idx_test]

    # Setup GCN Model
    # device = 'cuda'
    model = GCN(nfeat=features.shape[1], nhid=args.coreset_hidden, nclass=data.nclass, device=device,
                weight_decay=args.coreset_init_weight_decay)

    model = model.to(device)
    if args.coreset_load_npy=='':
        optimizer_model = torch.optim.Adam(model.parameters(), lr=args.coreset_init_lr, weight_decay=args.coreset_init_weight_decay)
        for e in range(args.coreset_epochs + 1):
            model.train()
            optimizer_model.zero_grad()
            embed, output = model.forward(features, adj,get_embedding=True)
            loss = F.nll_loss(output[idx_train], labels[idx_train])
            acc = utils.accuracy(output[idx_train], labels[idx_train])

            logging.info('=========Train===============')
            logging.info(
                'Epochs={}: Full graph train set results: loss = {:.4f}, accuracy = {:.4f}'.format(e, loss.item(), acc.item()))
            loss.backward()
            optimizer_model.step()
            if e % 10 == 0:
                model.eval()
                output_test = model.forward(features, adj)
                loss_test = F.nll_loss(output_test[idx_test], labels_test)
                acc_test = utils.accuracy(output_test[idx_test], labels_test)
                logging.info('=========Testing===============')
                logging.info(
                    'Epochs={}: Test results: loss = {:.4f}, accuracy = {:.4f}'.format(e, loss_test.item(), acc_test.item()))

            embed_out = embed

        if args.coreset_method == 'kcenter':
            agent = KCenter(data, args, device=device)
        if args.coreset_method == 'herding':
            agent = Herding(data, args, device=device)
        if args.coreset_method == 'random':
            agent = Random(data, args, device=device)
        if args.coreset_method == "K-Center":
            agent = KCenter(data, args, device=device)
        elif args.coreset_method == "Center":
            agent = Center(data, args, device=device)
        elif args.coreset_method == "K-means":
            agent = KMeans(data, args, device=device)
        elif args.coreset_method == "herding":
            agent = Herding(data, args, device=device)
        elif args.coreset_method == "Random_real":
            agent = Random(data, args, device=device)
        idx_selected = agent.select(embed_out)
        feat_train = features[idx_selected]
        adj_train = data.adj_full[np.ix_(idx_selected, idx_selected)]
        labels_train = labels[idx_selected]
        if args.coreset_save:
            logging.info('Saving...')
            np.save(f'{log_dir}/idx_{args.dataset}_{args.reduction_rate}_{args.coreset_method}_{args.seed}.npy', idx_selected)
        logging.info(args)
        logging.info(log_dir)
    else:
        res = []
        #runs = 10
        logging.info('Loading from: {}'.format(args.coreset_load_npy))
        idx_selected_train = np.load(f'{args.coreset_load_npy}/idx_{args.dataset}_{args.reduction_rate}_{args.coreset_method}_{args.seed}.npy')
        idx_selected = idx_selected_train
        feat_train = features[idx_selected_train]
        adj_train = data.adj_full[np.ix_(idx_selected_train, idx_selected_train)]
        labels_train = labels[idx_selected_train]
        if sp.issparse(adj_train):
            adj_train = sparse_mx_to_torch_sparse_tensor(adj_train)
        else:
            adj_train = torch.FloatTensor(adj_train)
        adj_train, feat_train, labels_train = adj_train.to(device), feat_train.to(device), labels_train.to(device)

        if utils.is_sparse_tensor(adj_train):
            adj_train_norm = utils.normalize_adj_tensor(adj_train, sparse=True)
        else:
            adj_train_norm = utils.normalize_adj_tensor(adj_train)

        adj_train = adj_train_norm.to(device)
        if args.coreset_opt_type_train=='Adam':
            optimizer_model_coreset = torch.optim.Adam(model.parameters(), lr=args.lr_coreset, weight_decay=args.wd_coreset)
        elif args.coreset_opt_type_train=='SGD':
            optimizer_model_coreset = torch.optim.SGD(model.parameters(), lr=args.lr_coreset, momentum=0.9)
        for _ in tqdm(range(args.coreset_runs)):
            model.initialize()
            best_test_acc=0
            for e in range(args.coreset_epochs + 1):
                model.train()
                optimizer_model_coreset.zero_grad()
                output_train = model.forward(feat_train, adj_train)
                loss_train = F.nll_loss(output_train, labels_train)
                acc_train = utils.accuracy(output_train, labels_train)
                logging.info('=========Train coreset===============')
                logging.info('Epochs={}: coreset results: loss = {:.4f}, accuracy = {:.4f}'.format(e, loss_train.item(),
                                                                                                acc_train.item()))

                loss_train.backward()
                optimizer_model_coreset.step()
                if e % 10 == 0:
                    model.eval()
                    output_test = model.forward(features, adj)
                    loss_test = F.nll_loss(output_test[idx_test], labels_test)
                    acc_test = utils.accuracy(output_test[idx_test], labels_test)
                    if acc_test > best_test_acc:
                        best_test_acc = acc_test.item()
                    logging.info('=========Test coreset===============')
                    logging.info('Epochs={}: Test coreset results: loss = {:.4f}, accuracy = {:.4f}'.format(e, loss_test.item(),
                                                                                                    acc_test.item()))
            res.append(best_test_acc)

        res = np.array(res)
        logging.info(args)
        logging.info(log_dir)
        logging.info('Mean accuracy = {:.4f}, Std = {:.4f}'.format(res.mean(), res.std()))
    return idx_selected

def init_coreset_inductive(data,args):
    device = torch.device(args.device)
    # torch.cuda.set_device(args.gpu_id)
    # args = load_config(args)
    # print(args)
    log_dir = './' + args.coreset_log + '/Coreset/{}-reduce_{}-{}'.format(args.dataset, str(args.reduction_rate),
                                                                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'coreset.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('This is the log_dir: {}'.format(log_dir))

    features, adj, labels = data.feat_train, data.adj_train, data.labels_train
    adj, features, labels = utils.to_tensor(adj, features, labels, device=device)
    # labels = torch.FloatTensor(labels)
    adj, features, labels = adj.to(device), features.to(device), labels.to(device)
    feat_test, adj_test, labels_test = data.feat_test, data.adj_test, data.labels_test
    adj_test, feat_test, labels_test = utils.to_tensor(adj_test, feat_test, labels_test, device=device)

    feat_val, adj_val, labels_val = data.feat_val, data.adj_val, data.labels_val
    adj_val, feat_val, labels_val = utils.to_tensor(adj_val, feat_val, labels_val, device=device)

    if utils.is_sparse_tensor(adj):
        adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
    else:
        adj_norm = utils.normalize_adj_tensor(adj)

    adj = adj_norm.to(device)

    # Setup GCN Model
    # device = 'cuda'
    model = GCN(nfeat=features.shape[1], nhid=args.coreset_hidden, nclass=data.nclass, device=device,
                weight_decay=args.coreset_init_weight_decay)

    model = model.to(device)
    if args.coreset_load_npy == '':
        optimizer_model = torch.optim.Adam(model.parameters(), lr=args.coreset_init_lr, weight_decay=args.coreset_init_weight_decay)
        best_load_test = best_load_it = 0
        for e in range(args.coreset_epochs + 1):
            model.train()
            optimizer_model.zero_grad()
            embed, output = model.forward(features, adj, get_embedding=True)
            loss = F.nll_loss(output, labels)
            acc = utils.accuracy(output, labels)

            # print('Epochs:', e, 'Full graph train set results: loss = ',loss.item(), 'accuracy=',acc.item())
            if e % 10 == 0:
                logging.info('=========Train===============')
                logging.info(
                    'Epochs={}: Full graph train set results: loss = {:.4f}, accuracy = {:.4f}'.format(e, loss.item(),
                                                                                                    acc.item()))
            loss.backward()
            optimizer_model.step()
            if e % 1 == 0:
                model.eval()
                # You can use the inner function of model to test
                output_test = model.predict(feat_test, adj_test)
                loss_test = F.nll_loss(output_test, labels_test)
                acc_test = utils.accuracy(output_test, labels_test)
                if acc_test > best_load_test:
                    best_load_test = acc_test.item()
                    best_load_it = e
                    # print('=============Testing===============')
                    logging.info('=========Testing===============')
                    # print('Epochs:', e, 'Test results: loss = ',loss_test.item(), 'accuracy=',acc_test.item())

            embed_out = embed
        logging.info(
            'BEST Test results: accuracy = {:.4f} within {}-th iterations'.format(best_load_test, best_load_it))

        if args.coreset_method == 'kcenter':
            agent = KCenter(data, args, device=device)
        if args.coreset_method == 'herding':
            agent = Herding(data, args, device=device)
        if args.coreset_method == 'random':
            agent = Random(data, args, device=device)
        idx_selected = agent.select(embed_out, inductive=True)
        feat_train = features[idx_selected]
        adj_train = data.adj_train[np.ix_(idx_selected, idx_selected)]
        labels_train = labels[idx_selected]
        if args.coreset_save:
            logging.info('Saving...')
            np.save(f'{log_dir}/idx_{args.dataset}_{args.reduction_rate}_{args.coreset_method}_{args.seed}.npy', idx_selected)
        logging.info(args)
        logging.info(log_dir)
    else:
        res = []
        # runs = 10
        logging.info('Loading from: {}'.format(args.coreset_load_npy))
        idx_selected_train = np.load(
            f'{args.coreset_load_npy}/idx_{args.dataset}_{args.reduction_rate}_{args.coreset_method}_{args.seed}.npy')
        idx_selected = idx_selected_train
        feat_train = features[idx_selected_train]
        # feat_train = F.normalize(feat_train,p=1)
        adj_train = data.adj_train[np.ix_(idx_selected_train, idx_selected_train)]
        # adj_train = torch.ones((feat_train.shape[0],feat_train.shape[0]))
        # adj_train = torch.eye(feat_train.shape[0])
        labels_train = labels[idx_selected_train]
        if sp.issparse(adj_train):
            adj_train = sparse_mx_to_torch_sparse_tensor(adj_train)
        else:
            adj_train = torch.FloatTensor(adj_train)
        adj_train, feat_train, labels_train = adj_train.to(device), feat_train.to(device), labels_train.to(device)
        # assert False

        if utils.is_sparse_tensor(adj_train):
            adj_train_norm = utils.normalize_adj_tensor(adj_train, sparse=True)
        else:
            adj_train_norm = utils.normalize_adj_tensor(adj_train)

        adj_train = adj_train_norm.to(device)
        if args.coreset_opt_type_train == 'Adam':
            optimizer_model_coreset = torch.optim.Adam(model.parameters(), lr=args.lr_coreset, weight_decay=args.wd_coreset)
        elif args.coreset_opt_type_train == 'SGD':
            optimizer_model_coreset = torch.optim.SGD(model.parameters(), lr=args.lr_coreset, momentum=0.9)
        for _ in tqdm(range(args.coreset_runs)):
            model.initialize()
            best_test_acc = 0
            for e in range(args.coreset_epochs + 1):
                model.train()
                optimizer_model_coreset.zero_grad()
                output_train = model.forward(feat_train, adj_train)
                loss_train = F.nll_loss(output_train, labels_train)
                acc_train = utils.accuracy(output_train, labels_train)
                logging.info('=========Train coreset===============')
                logging.info('Epochs={}: coreset results: loss = {:.4f}, accuracy = {:.4f}'.format(e, loss_train.item(),
                                                                                                acc_train.item()))

                # print('Epochs:', e, 'Train graph set results: loss = ', loss_train.item(), 'accuracy=', acc_train.item())
                loss_train.backward()
                optimizer_model_coreset.step()
                if e % 10 == 0:
                    model.eval()
                    # You can use the inner function of model to test
                    output_test = model.predict(feat_test, adj_test)
                    loss_test = F.nll_loss(output_test, labels_test)
                    acc_test = utils.accuracy(output_test, labels_test)
                    if acc_test > best_test_acc:
                        best_test_acc = acc_test.item()
                        best_it = e
                    logging.info('=========Test coreset===============')
                    logging.info(
                        'Epochs={}: Test coreset results: loss = {:.4f}, accuracy = {:.4f} within the best_acc = {}, best-iter = {}'.format(
                            e, loss_test.item(),
                            acc_test.item(), best_test_acc, best_it))
            res.append(best_test_acc)

        #    model.fit_with_val(feat_train, adj_train, labels_train, data,
        #                 train_iters=600, normalize=True, verbose=False)
        #
        #    model.eval()
        #    labels_test = torch.LongTensor(data.labels_test).cuda()
        #
        #    # Full graph
        #    output = model.predict(data.feat_full, data.adj_full)
        #    loss_test = F.nll_loss(output[data.idx_test], labels_test)
        #    acc_test = utils.accuracy(output[data.idx_test], labels_test)
        #    res.append(acc_test.item())
        #
        res = np.array(res)
        logging.info(args)
        logging.info(log_dir)
        logging.info('Mean accuracy = {:.4f}, Std = {:.4f}'.format(res.mean(), res.std()))
    return idx_selected

def init_graphs(graph_list, num, max_nodes, initialize='Random'):
    if initialize == 'random':
        indices = random.sample(range(len(graph_list)), num)
    else:
        # padd features
        features = []
        for data in graph_list:
            pad_size = max_nodes - data.x.size(0)
            feature = torch.nn.functional.pad(data.x, (0, 0, 0, pad_size), "constant", 0)
            features.append(feature)
        flatten_feats = torch.stack([f.flatten() for f in features])
        if initialize == 'kcenter':
            n = len(features)
            indices = []
            current_idx = torch.randint(n, (1,)).item()  # Randomly select the initial center
            indices.append(current_idx)
            dists = torch.norm(flatten_feats - flatten_feats[current_idx], dim=1)  # Compute distances to the initial center
            for _ in range(1, num):
                current_idx = torch.argmax(dists).item()  # Find the furthest point from the current set of centers
                indices.append(current_idx)
                new_dists = torch.norm(flatten_feats - flatten_feats[current_idx], dim=1)
                dists = torch.minimum(dists, new_dists)  # Update distances
        if initialize == 'herding':
            mean_feature = torch.mean(flatten_feats, dim=0)
            indices = []
            idx_left = np.arange(len(features)).tolist()
            centroid = mean_feature.clone()
            for _ in range(num):
                distances = torch.norm(flatten_feats[idx_left] - centroid, dim=1)
                selected_idx = torch.argmin(distances).item()
                indices.append(idx_left[selected_idx])
                del idx_left[selected_idx]
                selected_features = flatten_feats[indices]
                centroid = mean_feature - torch.mean(selected_features, dim=0)
    return indices