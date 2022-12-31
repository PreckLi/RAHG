import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, HypergraphConv


class AttentionModel(nn.Module):
    def __init__(self, data, nhid, gnn_type, heads, dropout):
        super().__init__()
        r_feats_num = data.num_node_features_r
        s_feats_num = data.num_node_features_s
        self.heads = heads
        self.linear_r = nn.Linear(r_feats_num, nhid)
        self.q_r = nn.Parameter(torch.rand(nhid, self.heads))
        self.linear_s = nn.Linear(s_feats_num, nhid)
        self.q_s = nn.Parameter(torch.rand(nhid, self.heads))
        self.att_fusion = nn.Linear(self.heads * nhid, nhid)

        self.sim_fusion = nn.Linear(r_feats_num + s_feats_num, nhid)
        if gnn_type == 'gcn':
            self.model = GCN(data, nhid)
        if gnn_type == 'hgcn':
            self.model = HyperGCN(data, nhid, dropout=dropout)

    def forward(self, data, args):
        r_feats, s_feats = data.x_role, data.x_stru
        if args.fusion_type == 'attention':
            r_feats = torch.tanh(self.linear_r(r_feats))
            r_alpha = torch.matmul(r_feats, self.q_r)

            s_feats = torch.tanh(self.linear_s(s_feats))
            s_alpha = torch.matmul(s_feats, self.q_s)

            alpha = torch.exp(r_alpha) + torch.exp(s_alpha)
            r_alpha = torch.exp(r_alpha) / alpha
            s_alpha = torch.exp(s_alpha) / alpha
            fusion_x = torch.cat(
                [r_alpha[:, i].view(-1, 1) * r_feats + s_alpha[:, i].view(-1, 1) * s_feats for i in range(self.heads)],
                dim=1)
            fusion_x = self.att_fusion(fusion_x)

        if args.fusion_type == 'concat':
            fusion_x = torch.cat((r_feats, s_feats), dim=1)
            fusion_x = self.sim_fusion(fusion_x)
        data.x = fusion_x
        result = self.model.forward(data,args)
        return result

    def predict(self, data: Data, args):
        r_feats, s_feats = data.x_role, data.x_stru
        if args.fusion_type == 'attention':
            r_feats = torch.tanh(self.linear_r(r_feats))
            r_alpha = torch.matmul(r_feats, self.q_r)

            s_feats = torch.tanh(self.linear_s(s_feats))
            s_alpha = torch.matmul(s_feats, self.q_s)

            alpha = torch.exp(r_alpha) + torch.exp(s_alpha)
            r_alpha = torch.exp(r_alpha) / alpha
            s_alpha = torch.exp(s_alpha) / alpha
            fusion_x = torch.cat(
                [r_alpha[:, i].view(-1, 1) * r_feats + s_alpha[:, i].view(-1, 1) * s_feats for i in range(self.heads)],
                dim=1)
            fusion_x = self.att_fusion(fusion_x)

        if args.fusion_type == 'concat':
            fusion_x = torch.cat((r_feats, s_feats), dim=1)
            fusion_x = self.sim_fusion(fusion_x)
        data.x = fusion_x
        hid = self.model.predict(data)
        return hid

    def predict_att_weights(self, data: Data, args):
        r_feats, s_feats = data.x_role, data.x_stru
        if args.fusion_type == 'attention':
            r_feats = torch.tanh(self.linear_r(r_feats))
            r_alpha = torch.matmul(r_feats, self.q_r)

            s_feats = torch.tanh(self.linear_s(s_feats))
            s_alpha = torch.matmul(s_feats, self.q_s)

            alpha = torch.exp(r_alpha) + torch.exp(s_alpha)
            r_alpha = torch.exp(r_alpha) / alpha
            s_alpha = torch.exp(s_alpha) / alpha

            return r_alpha, s_alpha
        else:
            return

    def reset_parameters(self):
        self.linear_r.reset_parameters()
        self.linear_s.reset_parameters()


class GCN(nn.Module):
    def __init__(self, data, nhid):
        super().__init__()
        self.gcn1 = GCNConv(data.num_node_features, nhid)
        self.gcn2 = GCNConv(nhid, nhid)
        self.lin = nn.Linear(nhid, data.num_classes)
        self.down_sample = nn.Linear(data.num_node_features, nhid)

    def forward(self, data: Data):
        x, adj = data.x, data.edge_index
        x = self.gcn1(x, adj)
        x = torch.tanh(x)
        x = F.dropout(x, training=self.training)
        x = self.gcn2(x, adj)
        x = torch.tanh(x)
        x = F.dropout(x, training=self.training)
        # x = x + origin_x
        x = self.lin(x)

        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    def __init__(self, data, nhid):
        super().__init__()
        self.gat1 = GATConv(data.num_node_features, nhid, heads=8)
        self.gat2 = GATConv(8 * nhid, nhid, heads=1)
        self.lin = nn.Linear(nhid, data.num_classes)
        self.down_sample = nn.Linear(nhid, nhid)

    def forward(self, data: Data):
        x, adj = data.x, data.edge_index
        x = self.gat1(x, adj)
        x = torch.tanh(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, adj)
        x = torch.tanh(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)


class HyperGCN(nn.Module):
    def __init__(self, data, nhid, dropout=0.5):
        super(HyperGCN, self).__init__()
        self.dropout = dropout
        # self.HGC1 = HypergraphConv(data.num_node_features+nhid, nhid)
        self.HGC1 = HypergraphConv(nhid, nhid)
        self.HGC2 = HypergraphConv(nhid, nhid)
        self.lin = nn.Linear(nhid, data.num_classes)
        # self.down_sample = nn.Linear(data.num_node_features+nhid, nhid)
        self.down_sample = nn.Linear(nhid, nhid)

    def forward(self, data: Data,args):
        x,H = data.x,data.H
        # x=torch.cat((data.x,data.fx),dim=1)
        origin_x = self.down_sample(x)
        x = self.HGC1(x, H)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.HGC2(x, H)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)
        if args.residual:
            x = x + origin_x
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)

    def predict(self, data: Data):
        x, H = data.x, data.H
        x = self.HGC1(x, H)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.HGC2(x, H)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x
