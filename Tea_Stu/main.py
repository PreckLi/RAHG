import argparse
import random
import torch
import numpy as np
from train_eval import train, evaluate
from data_deal import load_data
from model import AttentionModel
from utils import construct_hypergraph
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--train_ratio', type=float, default=0.8, help='train ratio')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--emb_type', type=str, default='attention', help='embedding type:graphwave/node2vec/attention')
parser.add_argument('--model_type', type=str, default="hgcn", help='model type')
parser.add_argument('--hgcn_construct_type', type=str, default='degree', help='model type')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--seed', type=int, default=140, help='seed')
parser.add_argument('--nhid', type=int, default=100, help='hidden size')
parser.add_argument('--epochs', type=int, default=400, help='maximum number of epochs')
parser.add_argument('--dropout', type=float, default=0.6, help='dropout')
parser.add_argument('--heads', type=int, default=8, help='attention heads')
parser.add_argument('--fusion_type', type=str, default='attention', help='attention or concat')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def main():
    acc_list = list()
    logger = get_logger(f'./log_file/{args.hgcn_construct_type}_{args.model_type}.log')
    logger.info('start training!')
    for i in range(100):
        print("range{i}----------------------------------------------------------".format(i + 1))
        data = load_data(args)
        if args.model_type == 'hgcn':
            H = construct_hypergraph(data.edge_index, data.num_nodes, args.hgcn_construct_type)
            data.H = H
        model = AttentionModel(data, args.nhid, args.model_type,args.heads,args.dropout)
        model = train(model, args, data)
        acc = evaluate(model, data,args)
        acc_list.append(acc)
        logger.info(f"{args.dropout},{args.nhid},{args.lr},NO{i}, acc:{acc}")
    mean_acc = np.mean(acc_list)
    max_acc = max(acc_list)
    min_acc = min(acc_list)
    logger.info(f"mean:{mean_acc}, max:{max_acc}, min:{min_acc}")
    print(acc_list)
    print('mean acc:', format(mean_acc, '.4f'))
    print('max acc:', format(max_acc, '.4f'))
    print('min acc:', format(min_acc, '.4f'))

    print('---------args-----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('----------End!----------')


if __name__ == '__main__':
    main()
