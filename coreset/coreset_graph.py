import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from utils.init_coreset import init_graphs
from utils.utils_graphset import Dataset as GraphSetDataset
from utils.utils_graphset import get_max_nodes
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='NCI1',
                        choices=['PROTEINS', 'NCI1', 'DD', 'NCI109',
                        'ogbg-molbbbp', 'ogbg-molbace', 'ogbg-molhiv',
                        'MNIST', 'CIFAR10'],
                        help="The dataset to be used.")
    parser.add_argument("--method", type=str, default='Herding',
                        choices=['Herding', 'KCenter', 'Random'],
                        help="The initialization of the synthetic graphs")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--init_way", type=str, default="Random")
    parser.add_argument('--gpc', type=int, default=10,
                        help='Number of graphs per class to be synthetized.')
    parser.add_argument('--scale', type=str, default='uniform',
                        choices=['uniform', 'degree'],
                        help="The normalization method of GNTK")
    parser.add_argument("--save", type=int, default=1, help="Save Synthetic Graphs")
    parser.add_argument("--save_dir", type=str, default="save")

    args = parser.parse_args()
    name = args.dataset

    graph_per_class = args.gpc

    """ Loading dataset with different metrics """
    dataset = GraphSetDataset(args)
    training_set = dataset.train_dataset
    val_set = dataset.val_dataset
    test_set = dataset.test_dataset
    nclass = dataset.nclass
    nfeat = dataset.nfeat
    labels = dataset.labels

    """ Use Coreset Methods to select graphs """
    selected_idx = []
    max_nodes = get_max_nodes(args)
    for cla, label in enumerate(labels):
        graphs_cla = [(idx, data) for idx, data in enumerate(training_set) if data.y == label]
        init_idx = init_graphs([data for _, data in graphs_cla], graph_per_class, max_nodes, args.method)
        selected_idx.extend(graphs_cla[i][0] for i in init_idx)
    os.makedirs(f'{args.save_dir}/{args.method}/', exist_ok=True)
    if args.save:
        np.save(f'{args.save_dir}/{args.method}/idx_{args.dataset}_{args.gpc}.npy', selected_idx)