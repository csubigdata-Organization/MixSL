import os.path as osp
import torch
from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import networkx as nx
from torch_geometric.utils import to_dense_adj
from Mixed_Functions.I2GNN_utils import create_subgraphs2
from Mixed_Functions.I2GNN_dataloader import I2GNN_DataLoader


def load_data(dataset_name='MUTAG'):
    path = osp.join('/'.join(osp.dirname(osp.realpath(__file__)).split('/')[:-1]), 'data')

    dataset, num_classes, num_nodes = torch.load(osp.join(path, dataset_name, 'data_mix.pt'))
    
    print('the graphs in {} dataset is: {}'.format(dataset_name, len(dataset)))

    return dataset, num_classes, num_nodes


def generate_GraphSNN_adj(dataset):
    print('generating the adjacency matrix in GraphSNN paper')

    dataset_adj_list = []

    dataset_length = len(dataset)
    for itr in np.arange(dataset_length):
        A_array = np.array(to_dense_adj(dataset[itr].edge_index)[0])
        G = nx.from_numpy_matrix(A_array)

        sub_graphs = []
        subgraph_nodes_list = []
        sub_graphs_adj = []
        sub_graph_edges = []
        new_adj = torch.zeros(A_array.shape[0], A_array.shape[0])

        for i in np.arange(len(A_array)):
            s_indexes = []
            for j in np.arange(len(A_array)):
                s_indexes.append(i)
                if (A_array[i][j] == 1):
                    s_indexes.append(j)
            sub_graphs.append(G.subgraph(s_indexes))

        for i in np.arange(len(sub_graphs)):
            subgraph_nodes_list.append(list(sub_graphs[i].nodes))

        for index in np.arange(len(sub_graphs)):
            sub_graphs_adj.append(nx.adjacency_matrix(sub_graphs[index]).toarray())

        for index in np.arange(len(sub_graphs)):
            sub_graph_edges.append(sub_graphs[index].number_of_edges())

        for node in np.arange(len(subgraph_nodes_list)):
            sub_adj = sub_graphs_adj[node]
            for neighbors in np.arange(len(subgraph_nodes_list[node])):
                index = subgraph_nodes_list[node][neighbors]
                count = torch.tensor(0).float()
                if (index == node):
                    continue
                else:
                    c_neighbors = set(subgraph_nodes_list[node]).intersection(subgraph_nodes_list[index])
                    if index in c_neighbors:
                        nodes_list = subgraph_nodes_list[node]
                        sub_graph_index = nodes_list.index(index)
                        c_neighbors_list = list(c_neighbors)
                        for i, item1 in enumerate(nodes_list):
                            if (item1 in c_neighbors):
                                for item2 in c_neighbors_list:
                                    j = nodes_list.index(item2)
                                    count += sub_adj[i][j]

                    new_adj[node][index] = count / 2
                    new_adj[node][index] = new_adj[node][index] / (len(c_neighbors) * (len(c_neighbors) - 1))
                    new_adj[node][index] = new_adj[node][index] * (len(c_neighbors) ** 2)

        weight = torch.FloatTensor(new_adj)
        weight = weight / weight.sum(1, keepdim=True)

        weight = weight + torch.FloatTensor(A_array)

        coeff = weight.sum(1, keepdim=True)
        coeff = torch.diag((coeff.T)[0])

        weight = weight + coeff

        weight = weight.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        dataset_adj_list.append(weight)

    shapes = [len(adj) for adj in dataset_adj_list]
    N_nodes_max = np.max(shapes)
    for i, data in enumerate(dataset):
        setattr(data, 'GraphSNN_adj', dataset_adj_list[i])

    return dataset, N_nodes_max


def generate_I2GNN_data(dataset):
    print('generating the data in I2GNN paper')

    dataset_I2GNN_list = []

    for data in dataset:
        I2GNN_data = create_subgraphs2(data)
        dataset_I2GNN_list.append(I2GNN_data)

    dataset = dataset_I2GNN_list
    return dataset


def load_k_fold(dataset, folds, batch_size, backbone):
    print('{}-fold split'.format(folds))

    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), torch.zeros(len(dataset))):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    data_kfold = []
    for i in range(folds):
        data_ith = [0, 0, 0, 0]  # align with above split process.
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0

        train_mask = train_mask.nonzero().view(-1)

        train_dataset = [dataset[j] for j in train_mask]
        val_dataset = [dataset[j] for j in val_indices[i]]
        test_dataset = [dataset[j] for j in test_indices[i]]

        Right_DataLoader = I2GNN_DataLoader if backbone == 'I2GNN' else DataLoader
        data_ith.append(Right_DataLoader(train_dataset, batch_size, shuffle=True))
        data_ith.append(Right_DataLoader(val_dataset, batch_size, shuffle=True))
        data_ith.append(Right_DataLoader(test_dataset, batch_size, shuffle=True))
        data_kfold.append(data_ith)

    return data_kfold
