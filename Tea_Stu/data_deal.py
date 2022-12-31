import torch
from utils import get_X_matrix, get_edge_index, get_label, shuffle_data, shuffle_attention_data, set_mask
from torch_geometric.data import Data
import numpy as np
import networkx as nx


def load_data(args):
    edge_file = f'datasets/Tea_Stu/tea_stu_repeat.edgelist'
    label_file = f'datasets/Tea_Stu/tea_stu_label.txt'
    edge_index = get_edge_index(edge_file)
    label = get_label(label_file)
    if args.emb_type != 'attention':
        if args.emb_type == 'graphwave':
            x_file = f'datasets/graphwave_emb/tea_stu.csv'
        if args.emb_type == 'node2vec':
            x_file = f'datasets/node2vec_emb/tea_stu.emb'
        x, num_nodes, num_node_features = get_X_matrix(x_file, emb_type=args.emb_type)
        x, label = shuffle_data(x, label)
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes, num_node_features=num_node_features, y=label,
                    num_classes=2)
    if args.emb_type == 'attention':
        emb_role_file = f'datasets/graphwave_emb/tea_stu.csv'
        emb_stru_file = f'datasets/node2vec_emb/tea_stu.emb'
        x_role, num_nodes_r, num_node_features_r = get_X_matrix(emb_role_file, 'graphwave')
        x_stru, _, num_node_features_s = get_X_matrix(emb_stru_file, 'node2vec')
        x_role, x_stru, label = shuffle_attention_data(x_role, x_stru, label)
        data = Data(x_role=x_role, x_stru=x_stru, edge_index=edge_index, num_nodes=num_nodes_r,
                    num_node_features_r=num_node_features_r,num_node_features_s=num_node_features_s, y=label, num_classes=2)

    data = set_mask(data, args)
    return data
