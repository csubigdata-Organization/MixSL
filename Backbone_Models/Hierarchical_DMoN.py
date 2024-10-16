import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import DenseGINConv
from Backbone_Models.utils import DMoNPooling


class DMoN(torch.nn.Module):
    

    def __init__(self, args, gnn_layers=2):

        super(DMoN, self).__init__()

        self.args = args
        self.gnn_layers = gnn_layers

        self.in_features = self.args.num_features
        self.num_classes = self.args.num_classes
        self.hidden_dim = self.args.hidden_dim

        self.pre_processing = nn.Linear(self.in_features, self.hidden_dim)

        self.dense_convs = nn.ModuleList()
        self.dense_hierarchical_poolings = nn.ModuleList()

        for i in range(self.gnn_layers):
            nn1 = Sequential(Linear(self.hidden_dim, self.hidden_dim), ReLU(), Linear(self.hidden_dim, self.hidden_dim))
            conv = DenseGINConv(nn=nn1, train_eps=False)
            self.dense_convs.append(conv)

            hierarchical = DMoNPooling(self.hidden_dim, k=16)
            self.dense_hierarchical_poolings.append(hierarchical)

        self.post_processing = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(self.hidden_dim, self.num_classes))


    def forward(self, data):
        reprs = []

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.pre_processing(x))
        x = F.dropout(x, p=self.args.dropout, training=self.training)

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)
        aux_losses = []
        for i in range(self.gnn_layers):

            x = self.dense_convs[i](x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.args.dropout, training=self.training)

            mask = mask if i == 0 else None
            _, x, adj, l1, l2, l3 = self.dense_hierarchical_poolings[i](x, adj, mask)

            reprs.append(x)

            aux_loss = l1 + l2 + l3
            aux_losses.append(aux_loss)

        reprs_tensor = torch.stack(reprs, dim=-1)
        embed = reprs_tensor.sum(dim=-1)
        embed = embed.sum(dim=1)

        prediction = self.post_processing(embed)
        logits = F.log_softmax(prediction, dim=-1)
        return (logits, aux_losses)
