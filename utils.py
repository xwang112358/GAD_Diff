import random
from dgl.data.utils import load_graphs
import os
import json
import torch
from torch_geometric.data import Data
from typing import Optional, Tuple, Union
from torch import Tensor

import copy
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_torch_csc_tensor


class RootedSubgraphData(Data):
    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == 'sub_edge_index':
            return self.n_id.size(0)
        if key in ['n_sub_batch', 'e_sub_batch']:
            return 1 + int(self.n_sub_batch[-1])
        elif key == 'n_id':
            return self.num_nodes
        elif key == 'e_id':
            assert self.edge_index is not None
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)

    def map_data(self) -> Data:
        data = copy.copy(self)

        for key, value in self.items():
            if key in ['sub_edge_index', 'n_id', 'e_id', 'e_sub_batch']:
                del data[key]
            elif key == 'n_sub_batch':
                continue
            elif key == 'num_nodes':
                data.num_nodes = self.n_id.size(0)
            elif key == 'edge_index':
                data.edge_index = self.sub_edge_index
            elif self.is_node_attr(key):
                dim = self.__cat_dim__(key, value)
                data[key] = value.index_select(dim, self.n_id)
            elif self.is_edge_attr(key):
                dim = self.__cat_dim__(key, value)
                data[key] = value.index_select(dim, self.e_id)

        return data


class RootedSubgraph(BaseTransform, ABC):
    @abstractmethod
    def extract(
        self,
        data: Data,
        node_id: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        pass

    def map(
        self,
        data: Data,
        n_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        assert data.edge_index is not None
        num_nodes = data.num_nodes
        assert num_nodes is not None

        n_sub_batch, n_id = n_mask.nonzero().t()
        e_mask = n_mask[:, data.edge_index[0]] & n_mask[:, data.edge_index[1]]
        e_sub_batch, e_id = e_mask.nonzero().t()

        sub_edge_index = data.edge_index[:, e_id]
        arange = torch.arange(n_id.size(0), device=data.edge_index.device)
        node_map = data.edge_index.new_ones(num_nodes, num_nodes)
        node_map[n_sub_batch, n_id] = arange
        sub_edge_index += (arange * data.num_nodes)[e_sub_batch]
        sub_edge_index = node_map.view(-1)[sub_edge_index]

        return sub_edge_index, n_id, e_id, n_sub_batch, e_sub_batch

    def forward(self, data: Data) -> List[Data]:
        subgraphs = []
        for node_id in range(data.num_nodes):
            out = self.extract(data, node_id)
            d = RootedSubgraphData.from_dict(data.to_dict())
            d.sub_edge_index, d.n_id, d.e_id, d.n_sub_batch, d.e_sub_batch = out
            subgraphs.append(d.map_data())
        return subgraphs


class RootedRWSubgraph(RootedSubgraph):
    def __init__(self, walk_length: int, repeat: int = 1):
        super().__init__()
        self.walk_length = walk_length
        self.repeat = repeat

    def extract(
        self,
        data: Data,
        node_id: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        from torch_cluster import random_walk

        assert data.edge_index is not None
        num_nodes = data.num_nodes
        assert num_nodes is not None

        start = torch.tensor([node_id], device=data.edge_index.device)
        start = start.view(-1, 1).repeat(1, self.repeat).view(-1)
        walk = random_walk(data.edge_index[0], data.edge_index[1], start,
                           self.walk_length, num_nodes=data.num_nodes)

        n_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool,
                             device=walk.device)
        start = start.view(-1, 1).repeat(1, (self.walk_length + 1)).view(-1)
        n_mask[start, walk.view(-1)] = True

        return self.map(data, n_mask)

    def forward(self, data: Data, node_ids: List[int]) -> List[Data]:
        subgraphs = []
        for node_id in node_ids:
            out = self.extract(data, node_id)
            d = RootedSubgraphData.from_dict(data.to_dict())
            d.sub_edge_index, d.n_id, d.e_id, d.n_sub_batch, d.e_sub_batch = out
            subgraphs.append(d.map_data())
        return subgraphs

# Example usage
# data = ...  # Your PyG Data object
# node_ids = [0, 1, 2]  # List of node IDs for which to extract subgraphs
# transform = RootedRWSubgraph(walk_length=3, repeat=2)
# subgraphs = transform.forward(data, node_ids)
# Now `subgraphs` is a list of Data objects representing the subgraphs for each node in node_ids









class GADDataset:
    def __init__(self, name='tfinance', prefix='datasets/'):
        graph = load_graphs(prefix + name)[0][0]
        self.name = name
        self.graph = graph

    def split(self, semi_supervised=True, trial_id=0):
        if semi_supervised:
            trial_id += 10
        self.graph.ndata['train_mask'] = self.graph.ndata['train_masks'][:,trial_id]
        self.graph.ndata['val_mask'] = self.graph.ndata['val_masks'][:,trial_id]
        self.graph.ndata['test_mask'] = self.graph.ndata['test_masks'][:,trial_id]
        print(self.graph.ndata['train_mask'].sum(), self.graph.ndata['val_mask'].sum(), self.graph.ndata['test_mask'].sum())





def random_walk_until_maxN(
    row: Tensor,
    col: Tensor,
    start: Tensor,
    maxN: int,
    walk_length: int,
    p: float = 1,
    q: float = 1,
    coalesced: bool = True,
    num_nodes: Optional[int] = None,
    return_edge_indices: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Samples random walks until the number of unique nodes reaches `maxN`.
    Args:
        row (LongTensor): Source nodes.
        col (LongTensor): Target nodes.
        start (LongTensor): Nodes from where random walks start.
        maxN (int): The number of unique nodes to reach.
        walk_length (int): The walk length.
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
        coalesced (bool, optional): If set to :obj:`True`, will coalesce/sort
            the graph given by :obj:`(row, col)` according to :obj:`row`.
            (default: :obj:`True`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)
        return_edge_indices (bool, optional): Whether to additionally return
            the indices of edges traversed during the random walk.
            (default: :obj:`False`)

    :rtype: :class:`LongTensor`
    """
    if num_nodes is None:
        num_nodes = max(int(row.max()), int(col.max()), int(start.max())) + 1

    if coalesced:
        perm = torch.argsort(row * num_nodes + col)
        row, col = row[perm], col[perm]

    deg = row.new_zeros(num_nodes)
    deg.scatter_add_(0, row, torch.ones_like(row))
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])

    all_nodes = []
    all_edges = []
    unique_nodes = set()

    while len(unique_nodes) < maxN:
        node_seq, edge_seq = torch.ops.torch_cluster.random_walk(
            rowptr, col, start, walk_length, p, q)
        
        all_nodes.append(node_seq)
        all_edges.append(edge_seq)
        
        unique_nodes.update(node_seq.view(-1).tolist())
        
        if len(unique_nodes) >= maxN:
            break

    all_nodes = torch.cat(all_nodes, dim=1)
    if return_edge_indices:
        all_edges = torch.cat(all_edges, dim=1)
        return all_nodes, all_edges

    return all_nodes



















# archived code

def get_khop_subgraph_with_random_walk(pyg_data, node_idx, maxN, p_1hop=0.7):
    # Initialize the sampled node set with the central node
    sampled_nodes = set([node_idx])
    current_node = node_idx
    
    while len(sampled_nodes) < maxN:
        if random.random() < p_1hop:
            # Perform a 1-hop random walk (current node is the destination)
            neighbors = pyg_data.edge_index[0][pyg_data.edge_index[1] == current_node].tolist()
            if neighbors:
                next_node = random.choice(neighbors)
                sampled_nodes.add(next_node)
                current_node = next_node
        else:
            # Perform a 2-hop random walk (current node is the destination)
            neighbors = pyg_data.edge_index[0][pyg_data.edge_index[1] == current_node].tolist()
            if neighbors:
                intermediate_node = random.choice(neighbors)
                second_neighbors = pyg_data.edge_index[0][pyg_data.edge_index[1] == intermediate_node].tolist()
                if second_neighbors:
                    next_node = random.choice(second_neighbors)
                    sampled_nodes.add(intermediate_node)
                    sampled_nodes.add(next_node)
                    current_node = next_node
        
        # Stop if we have reached maxN nodes
        if len(sampled_nodes) >= maxN:
            sampled_nodes = set(list(sampled_nodes)[:maxN])
            break
    
    # Convert sampled nodes to a list
    sampled_nodes = list(sampled_nodes)
    
    # Create a mask for the sampled nodes
    node_mask = torch.zeros(pyg_data.num_nodes, dtype=torch.bool)
    node_mask[sampled_nodes] = True
    
    # Filter edges to keep only those that connect sampled nodes
    edge_index = pyg_data.edge_index[:, node_mask[pyg_data.edge_index[0]] & node_mask[pyg_data.edge_index[1]]]
    
    # Remap the node indices in edge_index
    node_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sampled_nodes)}
    edge_index = torch.tensor(
        [[node_idx_map[idx] for idx in edge] for edge in edge_index.t().tolist()],
        dtype=torch.long
    ).t()
    
    # Create a new Data object for the subsampled subgraph
    subgraph_data = Data(
        x=pyg_data.x[sampled_nodes],
        edge_index=edge_index,
        y=pyg_data.y[sampled_nodes] if pyg_data.y is not None else None,
        edge_attr=pyg_data.edge_attr[node_mask[pyg_data.edge_index[0]] & node_mask[pyg_data.edge_index[1]]] if pyg_data.edge_attr is not None else None
    )
    
    return subgraph_data