import os.path as osp
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from Encoded_Utils.encoded_function import set_feature_into_data, RW_feature, eigenvector_centrality, clustering_coefficient, count_node_maximal_cliques


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def save_encoded_feature(dataset_name='MUTAG', cleaned=False):
    path = osp.join('/'.join(osp.dirname(osp.realpath(__file__)).split('/')[:-1]), 'data')

    dataset = TUDataset(path, dataset_name, cleaned=cleaned)

    dataset.data.edge_attr = None
    # load and process
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    # for diffpool method: remove latge graphs
    num_nodes = max_num_nodes = 0
    for data in dataset:
        num_nodes += data.num_nodes
        max_num_nodes = max(data.num_nodes, max_num_nodes)

    # # Filter out a few really large graphs in order to apply DiffPool.
    if dataset_name == 'REDDIT-BINARY':
        num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
    else:
        num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)
    num_classes = dataset.num_classes


    ### obtain all data into list for modifing the data
    graphs_ptg = []
    for data in dataset:
        graphs_ptg.append(data)
    dataset = graphs_ptg


    ### obtain the corresponding feature
    dataset_RW_feature = RW_feature(dataset)
    dataset_evc_feature = eigenvector_centrality(dataset)
    dataset_clu_feature = clustering_coefficient(dataset)
    dataset_cliq_feature = count_node_maximal_cliques(dataset)

    ### add the feature into data
    dataset = set_feature_into_data(dataset, 'RW_feature', dataset_RW_feature)
    dataset = set_feature_into_data(dataset, 'evc_feature', dataset_evc_feature)
    dataset = set_feature_into_data(dataset, 'clu_feature', dataset_clu_feature)
    dataset = set_feature_into_data(dataset, 'cliq_feature', dataset_cliq_feature)


    torch.save((dataset, num_classes, num_nodes), osp.join(path, dataset_name, 'data_mix.pt'))
    print('the dataset is saved.')
