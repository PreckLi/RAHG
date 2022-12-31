import torch
from utils import get_X_matrix, get_edge_index, get_label, shuffle_data, shuffle_attention_data, set_mask
from torch_geometric.data import Data
import networkx as nx
import numpy as np


def load_data(args):
    edge_file = f'datasets/internet_industry/internet-industry-partnerships.edgelist'
    label_file = f'datasets/internet_industry/label.txt'
    edge_index = get_edge_index(edge_file)
    label = get_label(label_file)
    if args.emb_type != 'attention':
        if args.emb_type == 'graphwave':
            x_file = f'datasets/graphwave_emb/internet_industry.csv'
        if args.emb_type == 'node2vec':
            x_file = f'datasets/node2vec_emb/internet_industry.csv'
        x, num_nodes, num_node_features = get_X_matrix(x_file, emb_type=args.emb_type)
        x, label = shuffle_data(x, label)
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes, num_node_features=num_node_features, y=label,
                    num_classes=3)
    if args.emb_type == 'attention':
        emb_role_file = f'datasets/graphwave_emb/internet_industry.csv'
        emb_stru_file = f'datasets/node2vec_emb/internet_industry.csv'
        x_role, num_nodes_r, num_node_features_r = get_X_matrix(emb_role_file, 'graphwave')
        x_stru, _, num_node_features_s = get_X_matrix(emb_stru_file, 'node2vec')
        x_role, x_stru, label = shuffle_attention_data(x_role, x_stru, label)
        data = Data(x_role=x_role, x_stru=x_stru, edge_index=edge_index, num_nodes=num_nodes_r,
                    num_node_features_r=num_node_features_r,num_node_features_s=num_node_features_s, y=label, num_classes=3)

    from torch_geometric.utils import to_networkx, degree
    degree_t = degree(edge_index[0])
    mean_degree = torch.mean(degree_t)
    G = to_networkx(data)
    k_num = nx.core_number(G)
    mean_k_num = np.mean(list(k_num.values()))
    transitivity = nx.transitivity(G)

    data = set_mask(data, args)
    return data
