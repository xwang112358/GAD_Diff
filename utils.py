import random
from dgl.data.utils import load_graphs
import os
import json
import torch
from torch_geometric.data import Data
from typing import Optional, Tuple, Union
from torch import Tensor



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