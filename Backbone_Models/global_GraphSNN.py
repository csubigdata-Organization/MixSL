import torch
import torch.nn.functional as F
import torch.nn as nn
from Backbone_Models.utils import GraphSNNConv


class GraphSNN(torch.nn.Module):
    

    def __init__(self, args, gnn_layers=2):

        super(GraphSNN, self).__init__()

        self.args = args
        self.gnn_layers = gnn_layers

        self.in_features = self.args.num_features
        self.num_classes = self.args.num_classes
        self.hidden_dim = self.args.hidden_dim

        self.N_nodes_max = self.args.N_nodes_max

        self.convs = nn.ModuleList()

        conv = GraphSNNConv(self.in_features, self.hidden_dim, self.N_nodes_max, self.args.dropout)
        self.convs.append(conv)

        for i in range(self.gnn_layers-1):
            conv = GraphSNNConv(self.hidden_dim, self.hidden_dim, self.N_nodes_max, self.args.dropout)
            self.convs.append(conv)

        self.out_proj = nn.Linear((self.in_features + self.hidden_dim * (self.gnn_layers)), self.num_classes)


    def forward(self, data):
        shapes = [len(adj) for adj in data.GraphSNN_adj]
        device = data.x.device

        X, start = [], 0
        for sha in shapes:
            X.append(data.x[start:start + sha])
            start += sha
        X = self.padding(X, self.N_nodes_max, device, dim=0)
        X = torch.cat(X).view(data.num_graphs, self.N_nodes_max, self.in_features)

        A = [torch.tensor(adj).to(device) for adj in data.GraphSNN_adj]
        A = self.padding(A, self.N_nodes_max, device, dim=0)
        A = self.padding(A, self.N_nodes_max, device, dim=1)
        A = torch.cat(A).view(data.num_graphs, self.N_nodes_max, self.N_nodes_max)


        hidden_states= [X]

        for layer in self.convs:
            X = F.dropout(layer(A, X), p=self.args.dropout)
            hidden_states.append(X)

        X = torch.cat(hidden_states, dim=2).sum(dim=1)
        X = self.out_proj(X)

        logits = F.log_softmax(X, dim=-1)
        return logits


    def padding(self, mtx, desired_dim, device, dim=0):
        mtx_padding = []
        for m in mtx:
            if dim == 0:
                dim1 = desired_dim-m.shape[0]
                dim2 = m.shape[1]
            elif dim == 1:
                dim1 = m.shape[0]
                dim2 = desired_dim-m.shape[1]
            else:
                raise Exception("Wrong dim parameter")
            empty_vec = torch.zeros((dim1, dim2)).to(device)
            m = torch.cat((m, empty_vec), dim=dim)
            mtx_padding.append(m)

        return mtx_padding
