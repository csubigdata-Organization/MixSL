import torch
import torch_geometric
from torch_geometric.data import Data
import pdb


class Batch(Data):
    
    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

        self.batch = batch
        self.__data_class__ = Data
        self.__slices__ = None

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        
        
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = Batch()
        batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key: [0] for key in keys}

        for key in keys:
            batch[key] = []

        for key in follow_batch:
            batch['{}_batch'.format(key)] = []

        cumsum = {key: 0 for key in keys}
        if 'assignment_index_2' in keys:
            cumsum['assignment_index_2'] = torch.LongTensor([[0], [0]])
        if 'assignment_index_3' in keys:
            cumsum['assignment_index_3'] = torch.LongTensor([[0], [0]])
        batch.batch = []
        for i, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                if torch.is_tensor(item) and item.dtype != torch.bool:
                    item = item + cumsum[key]
                if torch.is_tensor(item):
                    size = item.size(data.__cat_dim__(key, data[key]))
                else:
                    size = 1
                batch.__slices__[key].append(size + batch.__slices__[key][-1])
                if key == 'node_to_subgraph':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'subgraph_to_graph':
                    cumsum[key] = cumsum[key] + 1
                elif key == 'original_edge_index':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'tree_edge_index':
                    cumsum[key] = cumsum[key] + data.num_cliques
                elif key == 'atom2clique_index':
                    cumsum[key] = cumsum[key] + torch.tensor([[data.num_atoms], [data.num_cliques]])
                elif key == 'edge_index_2':
                    cumsum[key] = cumsum[key] + data.iso_type_2.shape[0]
                elif key == 'edge_index_3':
                    cumsum[key] = cumsum[key] + data.iso_type_3.shape[0]
                elif key == 'batch_2':
                    cumsum[key] = cumsum[key] + 1
                elif key == 'batch_3':
                    cumsum[key] = cumsum[key] + 1
                elif key == 'assignment2_to_subgraph':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'assignment3_to_subgraph':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'assignment_index_2':
                    cumsum[key] = cumsum[key] + torch.LongTensor([[data.num_nodes], [data.iso_type_2.shape[0]]])
                elif key == 'assignment_index_3':
                    inc = data.iso_type_2.shape[0] if 'assignment_index_2' in data else data.num_nodes
                    cumsum[key] = cumsum[key] + torch.LongTensor([[inc], [data.iso_type_3.shape[0]]])
                elif key == 'node_to_subgraph_node':
                    cumsum[key] = cumsum[key] + torch.max(data.node_to_subgraph_node) + 1
                elif key == 'subgraph_node_to_subgraph':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'subgraph_edge_index':
                    cumsum[key] = cumsum[key] + torch.max(data.node_to_subgraph_node) + 1
                elif key == 'node_to_subgraph2':
                    cumsum[key] = cumsum[key] + data.node_to_subgraph2[-1] + 1
                elif key == 'subgraph2_to_subgraph':
                    cumsum[key] = cumsum[key] + data.subgraph2_to_subgraph[-1] + 1
                elif key == 'subgraph2_to_graph':
                    cumsum[key] = cumsum[key] + 1
                elif key == 'center_idx':
                    cumsum[key] = cumsum[key] + data.num_nodes
                elif key == 'node_to_original_node':
                    cumsum[key] = cumsum[key] + data.num_original_nodes
                else:
                    cumsum[key] = cumsum[key] + data.__inc__(key, item)
                batch[key].append(item)

                if key in follow_batch:
                    item = torch.full((size, ), i, dtype=torch.long)
                    batch['{}_batch'.format(key)].append(item)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes, ), i, dtype=torch.long)
                batch.batch.append(item)

        if num_nodes is None:
            batch.batch = None

        for key in batch.keys:
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(batch[key],
                                       dim=data_list[0].__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])

        # Copy custom data functions to batch (does not work yet):
        # if data_list.__class__ != Data:
        #     org_funcs = set(Data.__dict__.keys())
        #     funcs = set(data_list[0].__class__.__dict__.keys())
        #     batch.__custom_funcs__ = funcs.difference(org_funcs)
        #     for func in funcs.difference(org_funcs):
        #         setattr(batch, func, getattr(data_list[0], func))

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()

    def to_data_list(self):
        
        
        if self.__slices__ is None:
            raise RuntimeError(
                ('Cannot reconstruct data list from batch because the batch '
                 'object was not created using Batch.from_data_list()'))

        keys = [key for key in self.keys if key[-5:] != 'batch']
        cumsum = {key: 0 for key in keys}
        if 'assignment_index_2' in keys:
            cumsum['assignment_index_2'] = torch.LongTensor([[0], [0]])
        if 'assignment_index_3' in keys:
            cumsum['assignment_index_3'] = torch.LongTensor([[0], [0]])
        data_list = []
        for i in range(len(self.__slices__[keys[0]]) - 1):
            data = self.__data_class__()
            for key in keys:
                if torch.is_tensor(self[key]):
                    data[key] = self[key].narrow(
                        data.__cat_dim__(key,
                                         self[key]), self.__slices__[key][i],
                        self.__slices__[key][i + 1] - self.__slices__[key][i])
                    if self[key].dtype != torch.bool:
                        data[key] = data[key] - cumsum[key]
                else:
                    data[key] = self[key][self.__slices__[key][i]:self.
                                          __slices__[key][i + 1]]
                if key == 'node_to_subgraph':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'subgraph_to_graph':
                    cumsum[key] = cumsum[key] + 1
                elif key == 'original_edge_index':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'tree_edge_index':
                    cumsum[key] = cumsum[key] + data.num_cliques
                elif key == 'atom2clique_index':
                    cumsum[key] = cumsum[key] + torch.tensor([[data.num_atoms], [data.num_cliques]])
                elif key == 'edge_index_2':
                    cumsum[key] = cumsum[key] + data.iso_type_2.shape[0]
                elif key == 'edge_index_3':
                    cumsum[key] = cumsum[key] + data.iso_type_3.shape[0]
                elif key == 'batch_2':
                    cumsum[key] = cumsum[key] + 1
                elif key == 'batch_3':
                    cumsum[key] = cumsum[key] + 1
                elif key == 'assignment2_to_subgraph':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'assignment3_to_subgraph':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'assignment_index_2':
                    cumsum[key] = cumsum[key] + torch.LongTensor([[data.num_nodes], [data.iso_type_2.shape[0]]])
                elif key == 'assignment_index_3':
                    cumsum[key] = cumsum[key] + torch.LongTensor([[data.iso_type_2.shape[0]], [data.iso_type_3.shape[0]]])
                else:
                    cumsum[key] = cumsum[key] + data.__inc__(key, data[key])
            data_list.append(data)

        return data_list

    @property
    def num_graphs(self):
        
        return self.batch[-1].item() + 1
