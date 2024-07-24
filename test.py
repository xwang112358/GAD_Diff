import torch
from torch_geometric.utils import from_dgl, to_networkx, k_hop_subgraph, subgraph
import random
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected
from torch import Tensor
from typing import Optional, Tuple, Union
from collections import Counter
import numpy as np


reddit = torch.load('pyg_dataset/reddit.pt')
print(reddit)

def random_walk_subgraph(pyg_graph, start_node, walk_length, max_nodes, onlyE=False):
    edge_index = to_undirected(pyg_graph.edge_index)

    # Extract a 2-hop subgraph around the start_node
    hop2_subset, hop2_edge_index, mapping, _ = k_hop_subgraph(start_node, num_hops=2, edge_index=edge_index, relabel_nodes=True)
    node_mapping = {i: hop2_subset[i] for i in range(len(hop2_subset))}
    if len(hop2_subset) > max_nodes:
        walks = []
        while len(set(walks)) < max_nodes:
            walk = random_walk(pyg_graph, start_node, walk_length)
            walks.extend(walk)
            
        subset = [item[0] for item in Counter(walks).most_common(max_nodes)]
        subg_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True)
        node_mapping = {i: subset[i] for i in range(len(subset))}
    else:
        subset = hop2_subset
        subg_edge_index = hop2_edge_index

    x = pyg_graph.y[subset]
    x = torch.nn.functional.one_hot(x, num_classes=2).float()
    edge_attr = torch.tensor([[0, 1] for _ in range(subg_edge_index.shape[1])])
    extra_x = pyg_graph.x[subset]
    node_mapping = torch.tensor(list(node_mapping.values()))
    y = torch.empty(1, 0)
    # remove self-loops or not 
    if onlyE:
        x = torch.ones((len(subset), 1))
        
    # Create a new data object for the subgraph
    d = Data(x=x, edge_index=subg_edge_index, edge_attr = edge_attr, extra_x = extra_x,
             num_nodes=len(subset), node_mapping=node_mapping, y = y)
    return d

def random_walk(pyg_graph, start_node, walk_length=3):
    walk = [start_node]
    edge_index = pyg_graph.edge_index
    for _ in range(walk_length):
        neighbors = edge_index[1][edge_index[0] == walk[-1]]
        if len(neighbors) == 0:  # If no neighbors, stop the walk
            break
        next_node = np.random.choice(neighbors.cpu().numpy())
        walk.append(next_node)
    return walk

random_walk_subgraph(reddit, 0, 3, 10, onlyE=True)