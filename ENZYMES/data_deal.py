from utils import get_X_matrix,get_edge_index,get_label,shuffle_data,shuffle_attention_data, set_mask
from torch_geometric.data import Data

def load_data(args):
    edge_file = f'datasets/ENZYMES/ENZYMES{args.dataset[7:]}.edgelist'
    label_file = f'datasets/ENZYMES/ENZYMES{args.dataset[7:]}_label.txt'
    edge_index = get_edge_index(edge_file)
    label = get_label(label_file)
    if args.emb_type != 'attention':
        if args.emb_type == 'graphwave':
            x_file = f'datasets/graphwave_emb/ENZYMES{args.dataset[7:]}.csv'
        if args.emb_type == 'node2vec':
            x_file = f'datasets/node2vec_emb/ENZYMES{args.dataset[7:]}.emb'
        x, num_nodes, num_node_features = get_X_matrix(x_file, emb_type=args.emb_type)
        x, label = shuffle_data(x, label)
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes, num_node_features=num_node_features, y=label,
                    num_classes=2)

    if args.emb_type == 'attention':
        emb_role_file = f'datasets/graphwave_emb/ENZYMES{args.dataset[7:]}.csv'
        emb_stru_file = f'datasets/node2vec_emb/ENZYMES{args.dataset[7:]}.emb'
        x_role, num_nodes_r, num_node_features_r = get_X_matrix(emb_role_file, 'graphwave')
        x_stru, _, num_node_features_s = get_X_matrix(emb_stru_file, 'node2vec')
        x_role, x_stru, label = shuffle_attention_data(x_role, x_stru, label)
        data = Data(x_role=x_role, x_stru=x_stru, edge_index=edge_index, num_nodes=num_nodes_r,
                    num_node_features_r=num_node_features_r, num_node_features_s=num_node_features_s, y=label,
                    num_classes=2)

    data = set_mask(data, args)
    return data