import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import global_add_pool


class I2GNN(torch.nn.Module):
    def __init__(self, args, gnn_layers=2):
        super(I2GNN, self).__init__()

        self.args = args
        self.gnn_layers = gnn_layers

        self.in_features = self.args.num_features
        self.num_classes = self.args.num_classes
        self.hidden_dim = self.args.hidden_dim

        self.pre_processing = torch.nn.Embedding(100, self.hidden_dim)
        self.pre_processing_x = nn.Linear(self.in_features, self.hidden_dim)
        self.init_edge_pooling_nn = Sequential(Linear(self.hidden_dim, 2*self.hidden_dim), ReLU(), Linear(2*self.hidden_dim, self.hidden_dim))
        self.init_node_pooling_nn = Sequential(Linear(self.hidden_dim, 2*self.hidden_dim), ReLU(), Linear(2*self.hidden_dim, self.hidden_dim))
        self.init_global_pooling = global_add_pool

        self.convs = nn.ModuleList()
        self.edge_pooling_nns = nn.ModuleList()
        self.node_pooling_nns = nn.ModuleList()
        self.global_poolings = []

        for i in range(self.gnn_layers):
            conv = GatedGraphConv(self.hidden_dim, 1)
            self.convs.append(conv)

            edge_pooling_nn = Sequential(Linear(self.hidden_dim, 2*self.hidden_dim), ReLU(), Linear(2*self.hidden_dim, self.hidden_dim))
            self.edge_pooling_nns.append(edge_pooling_nn)
            node_pooling_nn = Sequential(Linear(self.hidden_dim, 2*self.hidden_dim), ReLU(), Linear(2*self.hidden_dim, self.hidden_dim))
            self.node_pooling_nns.append(node_pooling_nn)
            pooling = global_add_pool
            self.global_poolings.append(pooling)

        self.post_processing = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(self.hidden_dim, self.num_classes))

    def forward(self, data):
        reprs = []

        z, x, edge_index, batch = data.z, data.x, data.edge_index, data.batch

        z = F.relu(self.pre_processing(z).view(-1, self.hidden_dim))
        x = F.relu(self.pre_processing_x(x))

        node_emb = self.init_global_pooling(z, data.node_to_subgraph2)
        node_emb = self.init_edge_pooling_nn(node_emb)
        sub_emb = self.init_global_pooling(node_emb, data.subgraph2_to_subgraph)
        sub_emb = self.init_node_pooling_nn(sub_emb)
        graph_emb = self.init_global_pooling(sub_emb * x, data.subgraph_to_graph)
        reprs.append(graph_emb)

        z = F.dropout(z, p=self.args.dropout, training=self.training)

        for i in range(self.gnn_layers):
            z = self.convs[i](z, edge_index)
            z = F.relu(z)
            z = F.dropout(z, p=self.args.dropout, training=self.training)

            node_emb = self.global_poolings[i](z, data.node_to_subgraph2)
            node_emb = self.edge_pooling_nns[i](node_emb)
            sub_emb = self.global_poolings[i](node_emb, data.subgraph2_to_subgraph)
            sub_emb = self.node_pooling_nns[i](sub_emb)
            graph_emb = self.global_poolings[i](sub_emb * x, data.subgraph_to_graph)
            reprs.append(graph_emb)

        reprs_tensor = torch.stack(reprs, dim=-1)
        embed = reprs_tensor.sum(dim=-1)

        prediction = self.post_processing(embed)
        logits = F.log_softmax(prediction, dim=-1)
        return logits
