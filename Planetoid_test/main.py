import argparse
import os
import torch
import numpy as np
from train_eval import train, evaluate
from data_deal import load_data
from model import AttentionModel
from utils import construct_hypergraph


parser = argparse.ArgumentParser()
parser.add_argument('--train_ratio', type=float, default=0.8, help='train ratio')
parser.add_argument('--dataset', type=str, default='Pubmed', help='Cora/Citeseer/Pubmed')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--emb_type', type=str, default='attention', help='embedding type:graphwave/node2vec')
parser.add_argument('--model_type', type=str, default="hgcn", help='model type')
parser.add_argument('--hgcn_construct_type', type=str, default='degree', help='model type')
parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
parser.add_argument('--seed', type=int, default=777, help='seed')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--epochs', type=int, default=400, help='maximum number of epochs')
parser.add_argument('--dropout', type=float, default=0.6, help='dropout')
parser.add_argument('--heads', type=int, default=8, help='attention heads')
parser.add_argument('--residual', type=bool, default=True, help='residual')
parser.add_argument('--fusion_type', type=str, default='attention', help='attention or concat')
args = parser.parse_args()


def main():
    acc_list = list()
    for i in range(10):
        print("range{}----------------------------------------------------------".format(i + 1))
        data = load_data(args)
        if args.model_type == 'hgcn':
            if os.path.exists(f'hyperedge_index_{args.dataset}_{args.hgcn_construct_type}.pt'):
                H=torch.load(f'hyperedge_index_{args.dataset}_{args.hgcn_construct_type}.pt')
            else:
                H = construct_hypergraph(data.edge_index, data.num_nodes, args.hgcn_construct_type,args)
            data.H = H
        model = AttentionModel(data, args.nhid, args.model_type, args.heads, dropout=args.dropout)
        model, loss = train(model, args, data)
        acc = evaluate(model, data, args)
        acc_list.append(acc)
    mean_acc = np.mean(acc_list)
    max_acc = max(acc_list)
    min_acc = min(acc_list)
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
