from torch_geometric.data import Data
import re
import dgl
import torch
from scipy import sparse as sp


class WalkData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        num_nodes = self.num_nodes
        num_edges = self.edge_index.size(-1)
        self.g = None
        if bool(re.search('(combined_subgraphs)', key)):
            return getattr(self, key[:-len('combined_subgraphs')]+'subgraphs_nodes_mapper').size(0)
        elif bool(re.search('(subgraphs_batch)', key)):
            # should use number of subgraphs or number of supernodes.
            return 1+getattr(self, key)[-1]
        elif bool(re.search('(nodes_mapper)|(selected_supernodes)', key)):
            return num_nodes
        elif bool(re.search('(edges_mapper)', key)):
            # batched_edge_attr[subgraphs_edges_mapper] shoud be batched_combined_subgraphs_edge_attr
            return num_edges
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if bool(re.search('(combined_subgraphs)', key)):
            return -1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

class WalkTransform(object):
    def __init__(self, walk_step=50):
        super().__init__()
        self.walk_step = walk_step

    def __call__(self, data):
        data = WalkData(**{k: v for k, v in data})

        graph_cpu = dgl.graph((data.edge_index[0], data.edge_index[1]))
        if graph_cpu.num_nodes() < data.num_nodes:
            offset = data.num_nodes - graph_cpu.num_nodes()
            for i in range(offset):
                graph_cpu.add_nodes(1)

        each_rand_feat, each_adj_matrix = self.walk_encoding(graph_cpu, self.walk_step) # obtain transition probability matrix

        return each_rand_feat

    def walk_encoding(self, g, domain_size):

        adj_matrix_dense = g.adjacency_matrix().to_dense()
        A = adj_matrix_csr = sp.csr_matrix(adj_matrix_dense)

        Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)  # D^-1 a matrix with the reciprocal of each node's indegree as the diagonal element
        R = A * Dinv  # each element RW[i][j] represents the transition probability from node i to node j

        # Iterate
        P = [torch.from_numpy(R.diagonal()).float()]
        RM = R
        for _ in range(domain_size - 1):
            RM = RM * R
            P.append(torch.from_numpy(RM.diagonal()).float())
        P = torch.stack(P, dim=-1)  # the probability of nodes returning to themselves after multiple random walks

        return P, adj_matrix_dense
