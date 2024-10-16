import torch
from torch_geometric.utils import to_networkx
import networkx as nx
from Encoded_Utils.RW_transform import WalkTransform



def norm_feature(dataset_feature, device):
    
    dataset_feature_indicator = []
    for feature in dataset_feature:
        graph_indicator = feature.shape[0]
        dataset_feature_indicator.append(graph_indicator)

    dataset_feature_cat = torch.cat(dataset_feature, dim=0).to(device)
    dataset_feature_cat = dataset_feature_cat.log()
    dataset_feature_norm = (dataset_feature_cat.max() - dataset_feature_cat) / (dataset_feature_cat.max() - dataset_feature_cat.mean())

    dataset_feature_norm_final = []
    start = 0
    for indicator in dataset_feature_indicator:
        feature_final = dataset_feature_norm[start:start + indicator]
        dataset_feature_norm_final.append(feature_final)
        start += indicator

    return dataset_feature_norm_final


def set_feature_into_data(dataset, set_name, set_feature):
    
    for i, data in enumerate(dataset):
        setattr(data, set_name, set_feature[i])

    return dataset


def RW_feature(dataset):
    
    RW_transform = WalkTransform()
    dataset_RW_feature = []
    device = dataset[0].x.device
    print('get random walk feature, all {} graphs'.format(len(dataset)))

    for i, data in enumerate(dataset):
        ii = i + 1
        if ii % 100 == 0:
            print('have got {} graphs'.format(ii))

        rand_feat = RW_transform(data).to(device)
        dataset_RW_feature.append(rand_feat)

    return dataset_RW_feature


def eigenvector_centrality(dataset):
    
    dataset_evc_feature = []
    device = dataset[0].x.device
    print('get eigenvector centrality feature, all {} graphs'.format(len(dataset)))
    for i, data in enumerate(dataset):
        ii = i + 1
        if ii % 100 == 0:
            print('have got {} graphs'.format(ii))

        graph = to_networkx(data)
        evc_feat = nx.eigenvector_centrality_numpy(graph)
        evc_feat = [evc_feat[i] for i in range(data.num_nodes)]
        evc_feat = torch.tensor(evc_feat, dtype=torch.float32).view(-1, 1).to(device)
        dataset_evc_feature.append(evc_feat)

    return dataset_evc_feature


def clustering_coefficient(dataset):
    
    dataset_clu_feature = []
    device = dataset[0].x.device
    print('get clustering coefficient feature, all {} graphs'.format(len(dataset)))
    for i, data in enumerate(dataset):
        ii = i + 1
        if ii % 100 == 0:
            print('have got {} graphs'.format(ii))

        graph = to_networkx(data)
        clu_feat = nx.clustering(graph)
        clu_feat = [clu_feat[i] for i in range(data.num_nodes)]
        clu_feat = torch.tensor(clu_feat, dtype=torch.float32).view(-1, 1).to(device)
        dataset_clu_feature.append(clu_feat)

    return dataset_clu_feature


def count_node_maximal_cliques(dataset):
    
    dataset_cliq_feature = []
    device = dataset[0].x.device
    print('get node maximal cliques feature, all {} graphs'.format(len(dataset)))
    for i, data in enumerate(dataset):
        ii = i + 1
        if ii % 100 == 0:
            print('have got {} graphs'.format(ii))

        graph = to_networkx(data, to_undirected=True)
        cliq_feat = {node: sum(1 for cliqs in nx.find_cliques(graph) if node in cliqs) for node in graph}
        cliq_feat = [cliq_feat[i] for i in range(data.num_nodes)]
        cliq_feat = torch.tensor(cliq_feat, dtype=torch.float32).view(-1, 1).to(device)
        dataset_cliq_feature.append(cliq_feat)

    dataset_cliq_feature = norm_feature(dataset_cliq_feature, device)

    return dataset_cliq_feature
