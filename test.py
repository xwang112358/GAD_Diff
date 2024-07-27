# import torch
# from torch_geometric.utils import from_dgl, to_networkx, k_hop_subgraph, subgraph
# import random
# from torch_geometric.data import Data
# from torch_geometric.utils import subgraph, to_undirected
# from torch import Tensor
# from typing import Optional, Tuple, Union
# from collections import Counter
# import numpy as np


# reddit = torch.load('pyg_dataset/reddit.pt')
# print(reddit)

# def random_walk_subgraph(pyg_graph, start_node, walk_length, max_nodes, onlyE=False):
#     edge_index = to_undirected(pyg_graph.edge_index)

#     # Extract a 2-hop subgraph around the start_node
#     hop2_subset, hop2_edge_index, mapping, _ = k_hop_subgraph(start_node, num_hops=2, edge_index=edge_index, relabel_nodes=True)
#     node_mapping = {i: hop2_subset[i] for i in range(len(hop2_subset))}
#     if len(hop2_subset) > max_nodes:
#         walks = []
#         while len(set(walks)) < max_nodes:
#             walk = random_walk(pyg_graph, start_node, walk_length)
#             walks.extend(walk)
            
#         subset = [item[0] for item in Counter(walks).most_common(max_nodes)]
#         subg_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True)
#         node_mapping = {i: subset[i] for i in range(len(subset))}
#     else:
#         subset = hop2_subset
#         subg_edge_index = hop2_edge_index

#     x = pyg_graph.y[subset]
#     x = torch.nn.functional.one_hot(x, num_classes=2).float()
#     edge_attr = torch.tensor([[0, 1] for _ in range(subg_edge_index.shape[1])])
#     extra_x = pyg_graph.x[subset]
#     node_mapping = torch.tensor(list(node_mapping.values()))
#     y = torch.empty(1, 0)
#     # remove self-loops or not 
#     if onlyE:
#         x = torch.ones((len(subset), 1))
        
#     # Create a new data object for the subgraph
#     d = Data(x=x, edge_index=subg_edge_index, edge_attr = edge_attr, extra_x = extra_x,
#              num_nodes=len(subset), node_mapping=node_mapping, y = y)
#     return d

# def random_walk(pyg_graph, start_node, walk_length=3):
#     walk = [start_node]
#     edge_index = pyg_graph.edge_index
#     for _ in range(walk_length):
#         neighbors = edge_index[1][edge_index[0] == walk[-1]]
#         if len(neighbors) == 0:  # If no neighbors, stop the walk
#             break
#         next_node = np.random.choice(neighbors.cpu().numpy())
#         walk.append(next_node)
#     return walk

# random_walk_subgraph(reddit, 0, 3, 10, onlyE=True)



####################################################################################################


# get sampled local subgraph
import torch
from augment import augmentation
from torch_geometric.loader import DataLoader 
import yaml
from utils import GADDataset

from omegaconf import DictConfig
import networkx as nx
with open('configs/config.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg = DictConfig(cfg)
# print(cfg)

data = GADDataset('reddit')
data.split(semi_supervised=False, trial_id=1)
dgl_graph = data.graph
# train_mask = dgl_graph.ndata['train_mask'].bool()
label = dgl_graph.ndata['label']
# print(dgl_graph)
# print(111, train_mask.shape)

nx_graph = dgl_graph.to_networkx()
nx_undirected_graph = nx_graph.to_undirected()

# Find the number of disconnected subgraphs
num_disconnected_subgraphs = nx.number_connected_components(nx_undirected_graph)

print("Number of disconnected subgraphs:", num_disconnected_subgraphs) 


local_subgraph = torch.load('local_subgraphs/reddit_0.pt')
print(local_subgraph[0])
reddit = torch.load('pyg_dataset/reddit.pt')


augment_subgraphs = augmentation(cfg, reddit, 'reddit', local_subgraph)
augment_subgraph = augment_subgraphs[0]
print(augment_subgraph)

# update dgl_graph

for subgraph in augment_subgraphs:
    new_x = subgraph.x
    new_edges = subgraph.edge_index
    new_labels = subgraph.label

    assert new_x.size(0) == new_labels.size(0)
    assert new_x.size(0) == new_edges.max().item() + 1

    num_existing_nodes = dgl_graph.num_nodes()
    new_edges = new_edges + num_existing_nodes
    new_train_mask = torch.cat([torch.tensor([1], dtype=torch.uint8), torch.tensor([0]*(subgraph.x.size(0)-1), dtype=torch.uint8)])
    new_valid_mask = torch.tensor([0]*subgraph.x.size(0), dtype=torch.uint8)
    new_test_mask = torch.tensor([0]*subgraph.x.size(0), dtype=torch.uint8)

    dgl_graph.add_nodes(new_x.size(0), {'feature': new_x, 'label': new_labels, 'train_mask': new_train_mask, 'valid_mask': new_valid_mask, 'test_mask': new_test_mask})
    dgl_graph.add_edges(new_edges[0], new_edges[1])


# calculate the number disconnected subgraphs in the augmented graph

# Convert the augmented graph to a NetworkX graph
nx_graph = dgl_graph.to_networkx()
nx_undirected_graph = nx_graph.to_undirected()

# Find the number of disconnected subgraphs
num_disconnected_subgraphs = nx.number_connected_components(nx_undirected_graph)

print("Number of disconnected subgraphs:", num_disconnected_subgraphs) 

# print(dgl_graph)
# print(111, dgl_graph.ndata['train_mask'].bool().shape)






# todo: 1. check the newly added graphs are disconnected from the original graph