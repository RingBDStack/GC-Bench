import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
import datetime
import json
import argparse
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from utils.utils_graph import DataGraph
from utils.utils import *
from utils.utils_graphset import Dataset
from tensorboardX import SummaryWriter
from test_condg_graph import test_graph

def main():

    parser = argparse.ArgumentParser(description="Parameters for GCBM")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")
    parser.add_argument("--wandb",type=int,default=0,help="Use wandb")
    parser.add_argument("--method",type=str, default="SFGC", help="Method")
    parser.add_argument("--dataset", type=str, default="NCI1", help="Dataset")
    parser.add_argument("--transductive", type=int, default=1, help="Transductive setting")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--save_dir", type=str, default="save", help="Save directory")
    parser.add_argument("--keep_ratio", type=float, default=1.0)
    parser.add_argument('--ipc', type=int, default=50, help='number of condensed samples per class')
    parser.add_argument("--reduction_rate", type=float, default=1)
    parser.add_argument("--seed", type=int, default=15, help="Random seed")
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--net_norm', type=str, default='none')
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--nconvs', type=int, default=3)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--save", type=int, default=1)
    
    # SFGC coreset init parameters
    parser.add_argument("--coreset_hidden", type=float, default=256)
    parser.add_argument("--normalize_features", type=bool, default=True)
    parser.add_argument("--coreset_init_weight_decay", type=float, default=5e-4)
    parser.add_argument("--coreset_init_lr", type=float, default=0.01)
    parser.add_argument("--lr_coreset", type=float, default=0.01)
    parser.add_argument("--wd_coreset", type=float, default=5e-4)
    parser.add_argument("--coreset_nlayers", type=int, default=2, help="Random seed.")
    parser.add_argument("--coreset_epochs", type=int, default=1)
    parser.add_argument("--coreset_save", type=int, default=1)
    parser.add_argument("--coreset_log", type=str, default="logs")
    parser.add_argument(
        "--coreset_method",
        type=str,
        default="kcenter",
        choices=["kcenter", "herding", "random"],
    )
    parser.add_argument("--coreset_load_npy", type=str, default="")
    parser.add_argument("--coreset_opt_type_train", type=str, default="Adam")
    parser.add_argument("--coreset_runs",type=int,default=10)
    # SFGC distill parameters
    # parser.add_argument(
    #     "--config", type=str, default="config.json", help="Path to the config JSON file"
    # )
    # parser.add_argument(
    #     "--section",
    #     type=str,
    #     default="runed exps name",
    #     help="the experiments needs to run",
    # )
    parser.add_argument("--nruns", type=int, default=1)
    parser.add_argument("--ITER", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--samp_iter", type=int, default=5)
    parser.add_argument("--samp_num_per_class", type=int, default=10)
    parser.add_argument("--ntk_reg",type=int,default=1)
    parser.add_argument("--syn_steps", type=int, default=200)
    parser.add_argument("--expert_epochs",type=int,default=800)
    parser.add_argument("--start_epoch", type=int, default=10)
    parser.add_argument("--lr_feat", type=float, default=0.01)
    parser.add_argument("--lr_student", type=float, default=0.01)
    parser.add_argument("--lr_lr", type=float, default=1e-6)
    parser.add_argument("--student_nlayers", type=int, default=2)
    parser.add_argument("--student_hidden", type=int, default=256)
    parser.add_argument("--student_dropout", type=float, default=0.0)
    parser.add_argument(
        "--save_log", type=str, default="logs", help="path to save logs"
    )
    parser.add_argument(
        "--load_all",
        action="store_true",
        help="only use if you can fit all expert trajectories into RAM",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="number of expert files to read (leave as None unless doing ablations)",
    )
    parser.add_argument(
        "--max_experts",
        type=int,
        default=None,
        help="number of experts to read per file (leave as None unless doing ablations)",
    )
    parser.add_argument(
        "--condense_model", type=str, default="GCN", help="Default condensation model"
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        default="SGC",
        help="evaluation model for saving best feat",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="S",
        help="eval_mode, check utils.py for more info",
    )
    parser.add_argument(
        "--initial_save", type=int, default=0, help="whether save initial feat and syn"
    )
    parser.add_argument(
        "--interval_buffer",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether use interval buffer",
    )
    parser.add_argument(
        "--rand_start",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether use random start",
    )
    parser.add_argument(
        "--optimizer_con",
        type=str,
        default="Adam",
        help="See choices",
        choices=["Adam", "SGD"],
    )
    parser.add_argument(
        "--optim_lr", type=int, default=1, help="whether use LR lr learning optimizer"
    )
    parser.add_argument(
        "--optimizer_lr",
        type=str,
        default="Adam",
        help="See choices",
        choices=["Adam", "SGD"],
    )


    args = parser.parse_args()

    # with open(args.config, "r") as config_file:
    #     config = json.load(config_file)

    # if args.section in config:
    #     section_config = config[args.section]

    # for key, value in section_config.items():
    #     setattr(args, key, value)

    torch.cuda.set_device(args.gpu_id)
    
    log_dir = './' + args.save_log + f'/Distill/{args.dataset}-reduce-{args.reduction_rate}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    buffer_path = os.path.join(args.save_log, "Buffer/{}-buffer".format(args.dataset))
    if not os.path.exists(buffer_path):
        os.makedirs(buffer_path)

    args.__setattr__("buffer_path", buffer_path)
    args.__setattr__("device","cuda:{}".format(args.gpu_id))
    args.__setattr__("log_dir", log_dir)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('This is the log_dir: {}'.format(log_dir))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logging.info("args = {}".format(args))

    print(args)

    if not os.path.exists(f"{args.save_dir}/{args.method}"):
        os.makedirs(f"{args.save_dir}/{args.method}")

    data = Dataset(args)
    packed_data = data.packed_data

    from baselines.TM.metagtt_graph import MetaGtt

    agent = MetaGtt(packed_data, args)
    writer = SummaryWriter()
    agent.distill(writer)
    test_graph(args)

if __name__ == "__main__":
    main()
