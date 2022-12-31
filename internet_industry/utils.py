import torch
import pandas as pd
import random
from torch_geometric.data import Data
from torch_geometric.utils import degree


def get_X_matrix(file, emb_type):
    if emb_type == 'graphwave':
        data = pd.read_csv(file, delimiter=',', dtype=float)
        data = data.drop(labels='reals_0', axis=1)
        data = data.drop(labels='imags_0', axis=1)
    if emb_type == 'node2vec':
        data = pd.read_csv(file, skiprows=1, delimiter=' ', header=None, dtype=float)
        data = data.sort_values(by=0)
    x = data.iloc[:, 1:]
    row_num = x.shape[0]
    col_num = x.shape[1]
    x = torch.from_numpy(x.values)
    x = torch.as_tensor(x, dtype=torch.float32)
    return x, row_num, col_num


def get_edge_index(file):
    edge_list_file = open(file, encoding="utf-8")
    edge_list = list()
    for reader in edge_list_file.readlines():
        row = str(reader).strip("\n").split(" ")
        temprow = list()
        for i in row:
            temprow.append(int(i))
        edge_list.append(temprow)
    edge_list = sorted(sorted(edge_list, key=(lambda x: x[1])), key=(lambda x: x[0]))
    edge_list = list(map(list, zip(*edge_list)))
    edge_list = torch.tensor(edge_list, dtype=torch.long)
    return edge_list


def get_label(file):
    node_labels_list = list()
    label_list = list()
    cnt = 0
    file = open(file, encoding="utf-8")
    for reader in file.readlines():
        if cnt == 0:
            cnt = 1
            continue
        row = str(reader).strip("\n").split(" ")
        temprow = list()
        for i in row:
            temprow.append(int(i))
        node_labels_list.append(temprow)
        cnt += 1
    for i in node_labels_list:
        label_list.append(i[1])
    label = torch.as_tensor(label_list, dtype=torch.long)
    return label


def shuffle_data(x: torch.Tensor, label: torch.Tensor):
    if torch.is_tensor(x) == True:
        x = x.numpy()
    if torch.is_tensor(label) == True:
        label = label.numpy()
    for i in range(len(x)):
        j = int(random.random() * (i + 1))
        if j <= len(x) - 1:
            x[i], x[j] = x[j], x[i]
            label[i], label[j] = label[j], label[i]
    x = torch.from_numpy(x)
    label = torch.from_numpy(label)

    return x, label


def shuffle_attention_data(x_r: torch.Tensor, x_s: torch.Tensor, label: torch.Tensor):
    if torch.is_tensor(x_r) == True and torch.is_tensor(x_s) == True:
        x_r = x_r.numpy()
        x_s = x_s.numpy()
    if torch.is_tensor(label) == True:
        label = label.numpy()
    for i in range(len(x_r)):
        j = int(random.random() * (i + 1))
        if j <= len(x_r) - 1:
            x_r[i], x_r[j] = x_r[j], x_r[i]
            x_s[i], x_s[j] = x_s[j], x_s[i]
            label[i], label[j] = label[j], label[i]
    x_r = torch.from_numpy(x_r)
    x_s = torch.from_numpy(x_s)
    label = torch.from_numpy(label)

    return x_r, x_s, label


def set_mask(data: Data, args):
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    train_mask[:int(args.train_ratio * data.y.size(0))] = True

    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask[int(args.train_ratio * data.y.size(0)):] = True

    data.train_mask = train_mask
    data.test_mask = test_mask

    return data


def construct_hypergraph(edge_index, num_nodes, construct_type):
    if construct_type == 'node_connection':
        hypergraph_list = list()
        now_src_node = -1
        for i in range(len(edge_index[0])):
            src_node = edge_index[0][i]
            if src_node != now_src_node:
                hypergraph_list.append([0] * num_nodes)
                now_src_node = src_node
                now_dst_node = edge_index[1][i]
                hypergraph_list[-1][now_dst_node] = 1
            else:
                now_dst_node = edge_index[1][i]
                hypergraph_list[-1][now_dst_node] = 1
        hypergraph_matrix = torch.Tensor(hypergraph_list).t()
        node_set = list()
        edge_set = list()
        for i in range(hypergraph_matrix.shape[1]):
            for j in range(hypergraph_matrix.shape[0]):
                if hypergraph_matrix[j][i] == 1:
                    node_set.append(j)
                    edge_set.append(i)
        hyperedge_index = [node_set, edge_set]
    if construct_type == 'degree':
        degree_dict = dict()
        degree_list = degree(edge_index[0]).tolist()
        for idx, value in enumerate(degree_list):
            value = int(value)
            if not degree_dict.get(value):
                degree_dict[value] = [idx]
            else:
                degree_dict[value].append(idx)
        node_set = list()
        edge_set = list()
        for idx, key in enumerate(degree_dict):
            node_set.extend(degree_dict[key])
            edge_set.extend([idx] * len(degree_dict[key]))
        hyperedge_index = [node_set, edge_set]
    hyperedge_index = torch.as_tensor(hyperedge_index, dtype=torch.long)
    return hyperedge_index
