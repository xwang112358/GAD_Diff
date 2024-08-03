import argparse
import numpy as np
import torch
from collections import Counter
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.data import Data
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_networkx, k_hop_subgraph, to_dense_adj, from_dgl
import torch.nn.functional as F
import networkx as nx
from tqdm import tqdm
from utils import GADDataset
import os
import random
from utils import bfs_subgraph_sampling

# from dgl.data.utils import load_graphs

# questions: 48921 nodes, 153540 edges, 301 feats 4.6% --> 2250 anomaly
# tolokers: 11758 nodes, 519000 edges, 10 feats, 21.8% --> 2563 anomaly
# reddit: 10984 nodes, 168016 edges, 64 feats, 3.3% --> 362.472 anomaly 


# to do
# 1. get subgraph dataset for questions, tolokers, reddit 
# 2. randomly check 200 subgraphs's cc, density, diameter, and average degree
# 3. fix the randomness mechanism for the subgraph sampling 

# The GAD graph is not undirected, but the graph diffusion model requires an undirected graph.
# since our dataset is not hetereophic, we can convert the graph to undirected graph 

# 1. fix the k to 2  (finished)
# 2. get 2-hop subgraph for each node (finished)
# 3. random walk from the center node to sample the nodes (finished)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create subgraphs from a PyG dataset.')
    parser.add_argument('--name', type=str, default='questions  ', help='Name of the dataset')
    parser.add_argument('--maxN', type=int, default=150, help='Largest number of nodes allowed in the subgraph')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_subgraphs', type=int, default=1000, help='Number of subgraphs to create')
    parser.add_argument('--onehop', action='store_true', help='Use one-hop subgraph sampling')    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    
    try:
        pyg_data = torch.load(f'./pyg_dataset/{args.name}.pt')

        print(f"Loaded {args.name} PyG data")
        print(pyg_data.num_nodes)

    except FileNotFoundError:
        data = GADDataset(args.name)
        pyg_data = torch.load(f'./pyg_dataset/{args.name}.pt')

    # Ensure edges are undirected

    nx_graph = to_networkx(pyg_data, to_undirected=True)
    components = list(nx.connected_components(nx_graph))
    if len(components) > 1:
        print(f"Graph contains {len(components)} connected components.")
    else:
        print("Graph is connected.")
    
    anomaly_indices = torch.nonzero(pyg_data.y, as_tuple=False).squeeze().tolist()
    # print(anomaly_indices)
    train_indices = torch.nonzero(pyg_data.train_masks[:, 0], as_tuple=False).squeeze().tolist()

    valid_indices = torch.nonzero(pyg_data.val_masks[:, 0], as_tuple=False).squeeze().tolist()
    train_anomaly_indices = list(set(anomaly_indices).intersection(set(train_indices)))
    valid_anomaly_indices = list(set(anomaly_indices).intersection(set(valid_indices)))

    train_valid_anomaly_indices = train_anomaly_indices + valid_anomaly_indices

    print(len(train_valid_anomaly_indices))
    
    num_nodes_1hop = []
    num_nodes_2hop = []
    anomaly_subgraphs = []
    
    for i in tqdm(range(len(train_valid_anomaly_indices))):
        subset, _, _, _= k_hop_subgraph(train_valid_anomaly_indices[i], 1, pyg_data.edge_index)
        num_nodes_1hop.append(len(subset))
        subset, _, _, _= k_hop_subgraph(train_valid_anomaly_indices[i], 2, pyg_data.edge_index)
        num_nodes_2hop.append(len(subset))
        # extract 
    # print mean and std of the subgraph sizes
    mean = np.mean(num_nodes_1hop)
    std = np.std(num_nodes_1hop)
    print(f"One hop Mean # nodes: {mean}, std: {std}")
    mean = np.mean(num_nodes_2hop)
    std = np.std(num_nodes_2hop)
    print(f"Two hop Mean # nodes: {mean}, std: {std}")
    
    if args.onehop:
        for node_idx in tqdm(train_valid_anomaly_indices):
            subgraph_data = bfs_subgraph_sampling(pyg_data, node_idx, 35, onlyE=True)
            anomaly_subgraphs.append(subgraph_data)
        
        os.makedirs(f'./pyg_dataset/{args.name}_anomaly', exist_ok=True)
        
        torch.save(anomaly_subgraphs, f'./pyg_dataset/{args.name}_anomaly/{args.name}_1hop.pt')
    

    anomaly_subgraphs = []
    
    for i in tqdm(range(args.num_subgraphs)):
        node_idx = random.choice(train_valid_anomaly_indices)
        subgraph_data = bfs_subgraph_sampling(pyg_data, node_idx, args.maxN, onlyE=True)
        anomaly_subgraphs.append(subgraph_data)
    
    os.makedirs(f'./pyg_dataset/{args.name}_anomaly', exist_ok=True)
    
    torch.save(anomaly_subgraphs, f'./pyg_dataset/{args.name}_anomaly/{args.name}_anomaly.pt')
    



        


    
