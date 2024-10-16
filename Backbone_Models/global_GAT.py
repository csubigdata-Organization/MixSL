import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_add_pool


class GAT(torch.nn.Module):
    

    def __init__(self, args, gnn_layers=2):

        super(GAT, self).__init__()

        self.args = args
        self.gnn_layers = gnn_layers

        self.in_features = self.args.num_features
        self.num_classes = self.args.num_classes
        self.hidden_dim = self.args.hidden_dim

        self.pre_processing = nn.Linear(self.in_features, self.hidden_dim)
        self.init_global_pooling = global_add_pool

        self.convs = nn.ModuleList()
        self.global_poolings = []

        for i in range(self.gnn_layers):
            heads = 2
            conv = GATConv(self.hidden_dim, self.hidden_dim // heads, heads=heads, aggr='add')
            self.convs.append(conv)
            pooling = global_add_pool
            self.global_poolings.append(pooling)

        self.post_processing = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(self.hidden_dim, self.num_classes))

    def forward(self, data):
        reprs = []

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.pre_processing(x))
        reprs.append(self.init_global_pooling(x, batch))
        x = F.dropout(x, p=self.args.dropout, training=self.training)

        for i in range(self.gnn_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.args.dropout, training=self.training)

            graph_emb = self.global_poolings[i](x, batch)
            reprs.append(graph_emb)

        reprs_tensor = torch.stack(reprs, dim=-1)
        embed = reprs_tensor.sum(dim=-1)

        prediction = self.post_processing(embed)
        logits = F.log_softmax(prediction, dim=-1)
        return logits
