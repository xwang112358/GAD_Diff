import random
from dgl.data.utils import load_graphs
import os
import json
import torch
from torch_geometric.data import Data

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