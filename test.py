import torch
from torch_geometric.utils import from_dgl, to_networkx, k_hop_subgraph, subgraph
print(subgraph)
import random
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected
from torch import Tensor
from typing import Optional, Tuple, Union
from collections import Counter
import numpy as np
from tqdm import tqdm
import torch_geometric.utils as utils
from collections import deque

from utils import GADDataset

def bfs_subgraph_sampling(pyg_graph, start_node, max_nodes, onlyE=False):
    edge_index = to_undirected(pyg_graph.edge_index)
    center_node = start_node

    # Extract a 2-hop subgraph around the start_node
    hop2_subset, hop2_edge_index, mapping, _ = k_hop_subgraph(start_node, num_hops=2, edge_index=edge_index, relabel_nodes=True)
    node_mapping = {i: hop2_subset[i].item() for i in range(len(hop2_subset))}

    if len(hop2_subset) > max_nodes:
        # BFS to downsample the subgraph while maintaining connectivity
        visited = set()
        queue = deque([start_node])
        subset = []

        while queue and len(subset) < max_nodes:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                subset.append(node)
                neighbors = edge_index[1][edge_index[0] == node].cpu().numpy()
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)

        subg_edge_index, _ = utils.subgraph(subset, edge_index, relabel_nodes=True)
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

    if onlyE:
        x = torch.ones((len(subset), 1))
        
    d = Data(x=x, edge_index=subg_edge_index, edge_attr=edge_attr, extra_x=extra_x,
             num_nodes=len(subset), node_mapping=node_mapping, y=y, center_node_idx=center_node)
    return d



data = GADDataset('tolokers')
pyg_graph = data.get_pyg_graph(save=False)

train_masks = pyg_graph.train_masks
train_mask = train_masks[:, 0]
print(train_mask.shape)
print(train_mask)


pyg_graph = data.get_pyg_graph(save=False)

anomaly_indices = anomaly_indices = torch.nonzero(pyg_graph.y, as_tuple=False).squeeze().tolist()


anomaly_subgraphs = []
for i in tqdm(range(2000)):
    node_idx = random.choice(anomaly_indices)
    subgraph = bfs_subgraph_sampling(pyg_graph, node_idx, 50, onlyE=True)
    anomaly_subgraphs.append(subgraph)

