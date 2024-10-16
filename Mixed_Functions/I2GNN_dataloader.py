import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data
from Mixed_Functions.I2GNN_batch import Batch 
import collections.abc as container_abcs
int_classes = int
string_classes = (str, bytes)


class I2GNN_DataLoader(torch.utils.data.DataLoader):
    
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 **kwargs):
        def collate(batch):
            elem = batch[0]
            if isinstance(elem, Data):
                return Batch.from_data_list(batch, follow_batch)
            elif isinstance(elem, float):
                return torch.tensor(batch, dtype=torch.float)
            elif isinstance(elem, int_classes):
                return torch.tensor(batch)
            elif isinstance(elem, string_classes):
                return batch
            elif isinstance(elem, container_abcs.Mapping):
                return {key: collate([d[key] for d in batch]) for key in elem}
            elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
                return type(elem)(*(collate(s) for s in zip(*batch)))
            elif isinstance(elem, container_abcs.Sequence):
                return [collate(s) for s in zip(*batch)]

            raise TypeError('I2GNN_DataLoader found invalid type: {}'.format(
                type(elem)))

        super(I2GNN_DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=lambda batch: collate(batch), **kwargs)


class DataListLoader(torch.utils.data.DataLoader):
    
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DataListLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=lambda data_list: data_list, **kwargs)


class DenseDataLoader(torch.utils.data.DataLoader):
    
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        def dense_collate(data_list):
            batch = Batch()
            for key in data_list[0].keys:
                batch[key] = default_collate([d[key] for d in data_list])
            return batch

        super(DenseDataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=dense_collate, **kwargs)
