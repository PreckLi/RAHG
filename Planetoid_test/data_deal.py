import torch
from utils import get_X_matrix
from torch_geometric.utils import  to_dense_adj
from torch_geometric.datasets import Planetoid


def load_data(args):
    dataset = Planetoid(root='datasets', name=args.dataset)
    data = dataset.data
    data.num_classes = dataset.num_classes

    emb_role_file = f'datasets/{args.dataset}/graphwave_emb/{args.dataset}.csv'
    emb_stru_file = f'datasets/{args.dataset}/node2vec_emb/{args.dataset}.emb'
    x_role, num_nodes, num_node_features_r = get_X_matrix(emb_role_file, 'graphwave')
    x_stru, _, num_node_features_s = get_X_matrix(emb_stru_file, 'node2vec')
    data.x_role = x_role
    data.x_stru = x_stru
    data.num_node_features_r = num_node_features_r
    data.num_node_features_s = num_node_features_s
    return data
