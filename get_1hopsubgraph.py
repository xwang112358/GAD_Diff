import argparse
import numpy as np
import torch
from collections import Counter
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.data import Data
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_dgl, to_networkx, k_hop_subgraph, to_dense_adj
import networkx as nx
from tqdm import tqdm
from utils import GADDataset, random_walk_until_maxN
import os
import random
from torch_geometric.transforms import NormalizeFeatures, SVDFeatureReduction


def get_1hop_subgraph(pyg_data, node_idx):
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, 1, pyg_data.edge_index, relabel_nodes=True, flow='source_to_target')
    
    return Data(x=pyg_data.x[subset], edge_index=edge_index, 
                y=pyg_data.y[subset], train_masks=pyg_data.train_masks[subset], 
                val_masks=pyg_data.val_masks[subset], 
                test_masks=pyg_data.test_masks[subset])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create k-hop subgraphs from a PyG dataset.')
    parser.add_argument('--name', type=str, default='reddit', help='Name of the dataset')

    args = parser.parse_args()
    name = args.name

    try:
        pyg_data = torch.load(f'./pyg_dataset/{name}.pt')
            
        print(f"Loaded {name} PyG data")
        print(pyg_data.num_nodes)

    except FileNotFoundError:
        data = GADDataset(name)
        print(data.graph.number_of_nodes())
        train_mask = data.graph.ndata['train_masks'][:,0]

        pyg_data = from_dgl(data.graph)
        pyg_data.edge_index = pyg_data.edge_index.long()
        
        if hasattr(pyg_data, 'train_masks'):
            pyg_data.train_masks = data.graph.ndata['train_masks'][:,0]
        if hasattr(pyg_data, 'val_masks'):
            pyg_data.val_masks = data.graph.ndata['val_masks'][:,0]
        if hasattr(pyg_data, 'test_masks'):
            pyg_data.test_masks = data.graph.ndata['test_masks'][:,0]
        if hasattr(pyg_data, 'count'):
            pyg_data.num_nodes = data.graph.number_of_nodes()
            del pyg_data.count 
        if hasattr(pyg_data, 'label'):
            pyg_data.y = pyg_data.label
            del pyg_data.label
        if hasattr(pyg_data, 'feature'):
            pyg_data.x = pyg_data.feature
            del pyg_data.feature
        
        torch.save(pyg_data, f'./pyg_dataset/{name}.pt')
        print(pyg_data)

    nx_graph = to_networkx(pyg_data, to_undirected=True)
    components = list(nx.connected_components(nx_graph))
    if len(components) > 1:
        print(f"Graph contains {len(components)} connected components.")
    else:
        print("Graph is connected.")
    
    anomaly_indices = torch.nonzero(pyg_data.y, as_tuple=False).squeeze().tolist()

    train_indices = torch.nonzero(pyg_data.train_masks, as_tuple=False).squeeze().tolist()
    valid_indices = torch.nonzero(pyg_data.val_masks, as_tuple=False).squeeze().tolist()
    test_indices = torch.nonzero(pyg_data.test_masks, as_tuple=False).squeeze().tolist()

    train_anomaly_indices = list(set(anomaly_indices).intersection(set(train_indices)))
    valid_anomaly_indices = list(set(anomaly_indices).intersection(set(valid_indices)))
    test_anomaly_indices = list(set(anomaly_indices).intersection(set(test_indices)))
    
    print(f"Number of anomalies in training set: {len(train_anomaly_indices)}")
    print(f"Number of anomalies in validation set: {len(valid_anomaly_indices)}")
    print(f"Number of anomalies in test set: {len(test_anomaly_indices)}")

    # Get 1-hop subgraph of each anomaly
    train_subgraphs = []
    for idx in train_anomaly_indices:
        train_subgraphs.append(get_1hop_subgraph(pyg_data, idx))

    valid_subgraphs = []
    for idx in valid_anomaly_indices:
        valid_subgraphs.append(get_1hop_subgraph(pyg_data, idx))

    train_valid_subgraphs = train_subgraphs + valid_subgraphs

    train_data = [(to_dense_adj(subgraph.edge_index).squeeze(0), subgraph.x) for subgraph in train_valid_subgraphs]

    train_clustering = [nx.average_clustering(to_networkx(subgraph)) for subgraph in train_valid_subgraphs]
    print(f"Average clustering coefficient of train subgraphs: {np.mean(train_clustering)}")


    if not os.path.exists(f'./subgraphs/{name}'):
        os.makedirs(f'./subgraphs/{name}')
    
    torch.save(train_data, f'./subgraphs/{name}/{name}_1hop.pt')