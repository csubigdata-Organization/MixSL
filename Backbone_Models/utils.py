from typing import Callable, Optional, Tuple, Union
import torch
from torch.nn import Module
from torch_scatter import scatter, scatter_add, scatter_min, scatter_max
from torch_sparse import SparseTensor, remove_diag
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.typing import Adj, OptTensor, PairTensor, Tensor

import torch.nn as nn
from torch.nn import ReLU, Parameter, Sequential, BatchNorm1d, Dropout
import torch.nn.functional as F
import math

from typing import List
from torch import Tensor
from torch_geometric.nn.dense.mincut_pool import _rank3_trace


class GraphSNNConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, batchnorm_dim, dropout):
        super().__init__()

        self.mlp = Sequential(Linear(input_dim, hidden_dim), Dropout(dropout),
                              ReLU(), BatchNorm1d(batchnorm_dim),
                              Linear(hidden_dim, hidden_dim), Dropout(dropout),
                              ReLU(), BatchNorm1d(batchnorm_dim))

        self.linear = Linear(hidden_dim, hidden_dim)

        self.eps = Parameter(torch.FloatTensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv_eps = 0.1 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)

    def forward(self, A, X):
        
        
        batch, N = A.shape[:2]
        mask = torch.eye(N).unsqueeze(0).to(A.device)
        batch_diagonal = torch.diagonal(A, 0, 1, 2)
        batch_diagonal = self.eps * batch_diagonal
        A = mask * torch.diag_embed(batch_diagonal) + (1. - mask) * A

        X = self.mlp(A @ X)
        X = self.linear(X)
        X = F.relu(X)

        return X


EPS = 1e-15
class DMoNPooling(torch.nn.Module):
    
    
    def __init__(self, channels: Union[int, List[int]], k: int,
                 dropout: float = 0.0):
        super().__init__()

        if isinstance(channels, int):
            channels = [channels]

        from torch_geometric.nn.models.mlp import MLP
        self.mlp = MLP(channels + [k], act=None, norm=None)

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        
        self.mlp.reset_parameters()

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        s = self.mlp(x)
        s = F.dropout(s, self.dropout, training=self.training)
        s = torch.softmax(s, dim=-1)

        (batch_size, num_nodes, _), C = x.size(), s.size(-1)

        if mask is None:
            mask = torch.ones(batch_size, num_nodes, dtype=torch.bool,
                              device=x.device)

        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

        out = F.selu(torch.matmul(s.transpose(1, 2), x))
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        # Spectral loss:
        degrees = torch.einsum('ijk->ij', adj)  # B X N
        degrees = degrees.unsqueeze(-1) * mask  # B x N x 1
        degrees_t = degrees.transpose(1, 2)  # B x 1 x N

        m = torch.einsum('ijk->i', degrees) / 2  # B
        m_expand = m.view(-1, 1, 1).expand(-1, C, C)  # B x C x C

        ca = torch.matmul(s.transpose(1, 2), degrees)  # B x C x 1
        cb = torch.matmul(degrees_t, s)  # B x 1 x C

        normalizer = torch.matmul(ca, cb) / 2 / m_expand
        decompose = out_adj - normalizer
        spectral_loss = -_rank3_trace(decompose) / 2 / m
        spectral_loss = spectral_loss.mean()

        # Orthogonality regularization:
        ss = torch.matmul(s.transpose(1, 2), s)
        i_s = torch.eye(C).type_as(ss)
        ortho_loss = torch.norm(
            ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
            i_s / torch.norm(i_s), dim=(-1, -2))
        ortho_loss = ortho_loss.mean()

        # Cluster loss:
        cluster_size = torch.einsum('ijk->ik', s)  # B x C
        cluster_loss = torch.norm(input=cluster_size, dim=1)
        cluster_loss = cluster_loss / mask.sum(dim=1) * torch.norm(i_s) - 1
        cluster_loss = cluster_loss.mean()

        # Fix and normalize coarsened adjacency matrix:
        ind = torch.arange(C, device=out_adj.device)
        out_adj[:, ind, ind] = 0
        d = torch.einsum('ijk->ij', out_adj)
        d = torch.sqrt(d)[:, None] + EPS
        out_adj = (out_adj / d) / d.transpose(1, 2)

        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.mlp.in_channels}, '
                f'num_clusters={self.mlp.out_channels})')


def maximal_independent_set(edge_index: Adj, k: int = 1,
                            perm: OptTensor = None) -> Tensor:
    
    
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        device = edge_index.device()
        n = edge_index.size(0)
    else:
        row, col = edge_index[0], edge_index[1]
        device = row.device
        n = edge_index.max().item() + 1

    if perm is None:
        rank = torch.arange(n, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(n, dtype=torch.long, device=device)

    mis = torch.zeros(n, dtype=torch.bool, device=device)
    mask = mis.clone()
    min_rank = rank.clone()

    while not mask.all():
        for _ in range(k):
            min_neigh = torch.full_like(min_rank, fill_value=n)
            scatter_min(min_rank[row], col, out=min_neigh)
            torch.minimum(min_neigh, min_rank, out=min_rank)  # self-loops

        mis = mis | torch.eq(rank, min_rank)
        mask = mis.clone().byte()

        for _ in range(k):
            max_neigh = torch.full_like(mask, fill_value=0)
            scatter_max(mask[row], col, out=max_neigh)
            torch.maximum(max_neigh, mask, out=mask)  # self-loops

        mask = mask.to(dtype=torch.bool)
        min_rank = rank.clone()
        min_rank[mask] = n

    return mis


def maximal_independent_set_cluster(edge_index: Adj, k: int = 1,
                                    perm: OptTensor = None) -> PairTensor:
    
    
    mis = maximal_independent_set(edge_index=edge_index, k=k, perm=perm)
    n, device = mis.size(0), mis.device

    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index[0], edge_index[1]

    if perm is None:
        rank = torch.arange(n, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(n, dtype=torch.long, device=device)

    min_rank = torch.full((n,), fill_value=n, dtype=torch.long, device=device)
    rank_mis = rank[mis]
    min_rank[mis] = rank_mis

    for _ in range(k):
        min_neigh = torch.full_like(min_rank, fill_value=n)
        scatter_min(min_rank[row], col, out=min_neigh)
        torch.minimum(min_neigh, min_rank, out=min_rank)

    _, clusters = torch.unique(min_rank, return_inverse=True)
    perm = torch.argsort(rank_mis)
    return mis, perm[clusters]


Scorer = Callable[[Tensor, Adj, OptTensor, OptTensor], Tensor]
class KMISPooling(Module):
    
    
    _heuristics = {None, 'greedy', 'w-greedy'}
    _passthroughs = {None, 'before', 'after'}
    _scorers = {
        'linear',
        'random',
        'constant',
        'canonical',
        'first',
        'last',
    }

    def __init__(self, in_channels: Optional[int] = None, k: int = 1,
                 scorer: Union[Scorer, str] = 'linear',
                 score_heuristic: Optional[str] = 'greedy',
                 score_passthrough: Optional[str] = 'before',
                 aggr_x: Optional[Union[str, Aggregation]] = None,
                 aggr_edge: str = 'sum',
                 aggr_score: Callable[[Tensor, Tensor], Tensor] = torch.mul,
                 remove_self_loops: bool = True) -> None:
        super(KMISPooling, self).__init__()
        assert score_heuristic in self._heuristics, \
            "Unrecognized `score_heuristic` value."
        assert score_passthrough in self._passthroughs, \
            "Unrecognized `score_passthrough` value."

        if not callable(scorer):
            assert scorer in self._scorers, \
                "Unrecognized `scorer` value."

        self.k = k
        self.scorer = scorer
        self.score_heuristic = score_heuristic
        self.score_passthrough = score_passthrough

        self.aggr_x = aggr_x
        self.aggr_edge = aggr_edge
        self.aggr_score = aggr_score
        self.remove_self_loops = remove_self_loops

        if scorer == 'linear':
            assert self.score_passthrough is not None, \
                "`'score_passthrough'` must not be `None`" \
                " when using `'linear'` scorer"

            self.lin = Linear(in_channels=in_channels, out_channels=1,
                              weight_initializer='uniform')

    def _apply_heuristic(self, x: Tensor, adj: SparseTensor) -> Tensor:
        if self.score_heuristic is None:
            return x

        row, col, _ = adj.coo()
        x = x.view(-1)

        if self.score_heuristic == 'greedy':
            k_sums = torch.ones_like(x)
        else:
            k_sums = x.clone()

        for _ in range(self.k):
            scatter_add(k_sums[row], col, out=k_sums)

        return x / k_sums

    def _scorer(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                batch: OptTensor = None) -> Tensor:
        if self.scorer == 'linear':
            return self.lin(x).sigmoid()

        if self.scorer == 'random':
            return torch.rand((x.size(0), 1), device=x.device)

        if self.scorer == 'constant':
            return torch.ones((x.size(0), 1), device=x.device)

        if self.scorer == 'canonical':
            return -torch.arange(x.size(0), device=x.device).view(-1, 1)

        if self.scorer == 'first':
            return x[..., [0]]

        if self.scorer == 'last':
            return x[..., [-1]]

        return self.scorer(x, edge_index, edge_attr, batch)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None,
                batch: OptTensor = None) \
            -> Tuple[Tensor, Adj, OptTensor, OptTensor, Tensor, Tensor]:
        
        adj, n = edge_index, x.size(0)

        if not isinstance(edge_index, SparseTensor):
            adj = SparseTensor.from_edge_index(edge_index, edge_attr, (n, n))

        score = self._scorer(x, edge_index, edge_attr, batch)
        updated_score = self._apply_heuristic(score, adj)
        perm = torch.argsort(updated_score.view(-1), 0, descending=True)

        mis, cluster = maximal_independent_set_cluster(adj, self.k, perm)

        row, col, val = adj.coo()
        c = mis.sum()

        if val is None:
            val = torch.ones_like(row, dtype=torch.float)

        adj = SparseTensor(row=cluster[row], col=cluster[col], value=val,
                           is_sorted=False,
                           sparse_sizes=(c, c)).coalesce(self.aggr_edge)

        if self.remove_self_loops:
            adj = remove_diag(adj)

        if self.score_passthrough == 'before':
            x = self.aggr_score(x, score)

        if self.aggr_x is None:
            x = x[mis]
        elif isinstance(self.aggr_x, str):
            x = scatter(x, cluster, dim=0, dim_size=mis.sum(),
                        reduce=self.aggr_x)
        else:
            x = self.aggr_x(x, cluster, dim_size=c)

        if self.score_passthrough == 'after':
            x = self.aggr_score(x, score[mis])

        if isinstance(edge_index, SparseTensor):
            edge_index, edge_attr = adj, None

        else:
            row, col, edge_attr = adj.coo()
            edge_index = torch.stack([row, col])

        if batch is not None:
            batch = batch[mis]

        return x, edge_index, edge_attr, batch, mis, cluster

    def __repr__(self):
        if self.scorer == 'linear':
            channels = f"in_channels={self.lin.in_channels}, "
        else:
            channels = ""

        return f'{self.__class__.__name__}({channels}k={self.k})'
