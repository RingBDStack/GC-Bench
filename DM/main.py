import sys
import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from utils.utils_graph import *
from utils.utils import *
from gcdm import GCDM


def main():

    parser = argparse.ArgumentParser(
        description="Parameters for GCBM-node classification"
    )
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to the config JSON file"
    )
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--section", type=str, default="")
    parser.add_argument("--wandb", type=int, default=1, help="Use wandb")
    parser.add_argument("--method", type=str, default="GCDM", help="Method")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--save_dir", type=str, default="save", help="Save directory")
    parser.add_argument("--keep_ratio", type=float, default=1.0)
    parser.add_argument("--sgc", type=int, default=1)
    parser.add_argument("--reduction_rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=15, help="Random seed")
    parser.add_argument("--alpha", type=float, default=0, help="Regularization term")
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--save", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs")
    parser.add_argument("--nlayers", type=int, default=2, help="Number of layers")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--lr_adj", type=float, default=1e-3)
    parser.add_argument("--lr_feat", type=float, default=1e-3)
    parser.add_argument("--lr_model", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--normalize_features", type=bool, default=True)
    parser.add_argument("--inner", type=int, default=0)
    parser.add_argument("--outer", type=int, default=20)
    parser.add_argument("--transductive", type=int, default=1)
    parser.add_argument("--one_step", type=int, default=0)
    parser.add_argument("--init_way", type=str, default="Random_real")
    parser.add_argument('--label_rate', type=float, default=1)

    args = parser.parse_args()
    if os.path.exists(args.config_dir + "/" + args.config):
        with open(args.config_dir + "/" + args.config, "r") as config_file:
            config = json.load(config_file)

        if args.section in config:
            section_config = config[args.section]

        for key, value in section_config.items():
            setattr(args, key, value)

    torch.cuda.set_device(args.gpu_id)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(args)

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    if not os.path.exists(f"{args.save_dir}/{args.method}"):
        os.makedirs(f"{args.save_dir}/{args.method}")

    data_pyg = [
        "cora",
        "citeseer",
        "pubmed",
        "cornell",
        "texas",
        "wisconsin",
        "chameleon",
        "squirrel",
    ]
    if args.dataset in data_pyg:
        data_full = get_dataset(args.dataset, args.normalize_features, args.data_dir)
        data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
    else:
        if args.transductive:
            data = DataGraph(args.dataset, data_dir=args.data_dir)
        else:
            data = DataGraph(
                args.dataset, label_rate=args.label_rate, data_dir=args.data_dir
            )
        data_full = data.data_full

    agent = GCDM(data, args, device="cuda")

    agent.train()


if __name__ == "__main__":
    main()
