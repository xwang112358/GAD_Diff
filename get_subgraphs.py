import argparse
import numpy as np
import torch
from collections import Counter
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.data import Data
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_dgl, to_networkx, k_hop_subgraph
import networkx as nx
from tqdm import tqdm
from utils import GADDataset
import os
import random
from torch_geometric.transforms import NormalizeFeatures, SVDFeatureReduction
# from dgl.data.utils import load_graphs

# to do
# 1. get subgraph dataset for questions, tolokers, reddit 
# questions: 48921 nodes, 153540 edges, 301 feats 4.6% --> 2250 anomaly
# tolokers: 11758 nodes, 519000 edges, 10 feats, 21.8% --> 2563 anomaly
# reddit: 10984 nodes, 168016 edges, 64 feats, 3.3% --> 362.472 anomaly 

# 2. count number of nodes for 2-hop subgraphs 


class SubgraphDataset(InMemoryDataset):
    # to do: make it load the dataset by dataset = SubgraphDataset instead of torch.load() 
    def __init__(self, root, dataset_name, subgraph_data_list: List[Data], transform=None, pre_transform=None):
        self.subgraph_data_list = subgraph_data_list
        self.dataset_name = dataset_name
        super(SubgraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = self.collate(self.subgraph_data_list)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'{self.dataset_name}_2hop.pt']

    def download(self):
        pass

    def process(self):
        pass


# 1. fix the k to 2
# 2. how to randomly subsample the nodes to maxN while weighting the nodes by their hop distance?
# 3. can we use random walk to sample the nodes?  have more 1-hop random walk samples than 2-hop random walk samples

def get_khop_subgraph_with_random_walk(pyg_data, node_idx, maxN, p_1hop=0.7):
    # Initialize the sampled node set with the central node
    sampled_nodes = set([node_idx])
    current_node = node_idx
    
    while len(sampled_nodes) < maxN:
        if random.random() < p_1hop:
            # Perform a 1-hop random walk
            neighbors = pyg_data.edge_index[1][pyg_data.edge_index[0] == current_node].tolist()
            if neighbors:
                next_node = random.choice(neighbors)
                sampled_nodes.add(next_node)
                current_node = next_node
        else:
            # Perform a 2-hop random walk
            neighbors = pyg_data.edge_index[1][pyg_data.edge_index[0] == current_node].tolist()
            if neighbors:
                intermediate_node = random.choice(neighbors)
                second_neighbors = pyg_data.edge_index[1][pyg_data.edge_index[0] == intermediate_node].tolist()
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
        dtype=torch.long).t()
    
    # Create a new Data object for the subsampled subgraph
    subgraph_data = Data(
        x=pyg_data.x[sampled_nodes],
        edge_index=edge_index,
        y=pyg_data.y[sampled_nodes] if pyg_data.y is not None else None,
        edge_attr=pyg_data.edge_attr[node_mask[pyg_data.edge_index[0]] & node_mask[pyg_data.edge_index[1]]] if pyg_data.edge_attr is not None else None
    )
    
    return subgraph_data

def get_khop_subgraph(pyg_data, node_idx, k, maxN):
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx, k, pyg_data.edge_index, relabel_nodes=True)
    hop_dists = torch.full((pyg_data.num_nodes,), -1, dtype=torch.long)
    hop_dists[subset] = mapping

    if len(subset) > maxN:
        # Sort nodes by hop distance in descending order
        sorted_nodes = sorted(subset.tolist(), key=lambda n: -hop_dists[n])
        subset = torch.tensor(sorted_nodes[:maxN])
        edge_index, edge_mask = subgraph(subset, pyg_data.edge_index, relabel_nodes=True, num_nodes=pyg_data.num_nodes)

    # Apply NormalizeFeatures and SVDFeatureReduction
    subgraph_data = Data(x=pyg_data.x[subset], edge_index=edge_index, num_nodes=len(subset))

    label = pyg_data.y[node_idx].item()
    subgraph_data.y = torch.tensor([label], dtype=torch.long)
    subgraph_data.center_node_idx = mapping[0].item()
    return subgraph_data


def create_subgraph(args):
    pyg_data, node_idx, k, split, maxN = args
    subgraph_data = get_khop_subgraph(pyg_data, node_idx, k, maxN)
    subgraph_data.split = split
    return subgraph_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create k-hop subgraphs from a PyG dataset.')
    parser.add_argument('--name', type=str, default='questions  ', help='Name of the dataset')
    parser.add_argument('--khops', type=int, default=2, help='Number of hops for subgraphs')
    parser.add_argument('--maxN', type=int, default=50, help='Largest number of nodes allowed in the subgraph')
    parser.add_argument('--svd_out_channels', type=int, default=64, help='Number of output channels of SVD')
    parser.add_argument('--use_svd', action='store_true', help='whether to use svd to conduct dimension reduction')
    parser.add_argument('--use_norm', action='store_true', help='whether to use normalize the node features')
    
    args = parser.parse_args()

    name = args.name
    k = args.khops
    maxN = args.maxN
    use_svd = args.use_svd
    svd_out_channels = args.svd_out_channels
    use_norm = args.use_norm
    
    try:
        pyg_data = torch.load(f'./pyg_dataset/raw/{name}.pt')
        
        if use_norm:
            normalize = NormalizeFeatures()
            pyg_data = normalize(pyg_data)
        if use_svd:
            svd = SVDFeatureReduction(svd_out_channels)
            pyg_data = svd(pyg_data)
            
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

    # Ensure edges are undirected
    pyg_data.edge_index = to_undirected(pyg_data.edge_index)

    nx_graph = to_networkx(pyg_data, to_undirected=True)
    components = list(nx.connected_components(nx_graph))
    if len(components) > 1:
        print(f"Graph contains {len(components)} connected components.")
    else:
        print("Graph is connected.")
        
    train_indices = torch.nonzero(pyg_data.train_masks, as_tuple=False).squeeze().tolist()
    vali_indices = torch.nonzero(pyg_data.val_masks, as_tuple=False).squeeze().tolist()
    test_indices = torch.nonzero(pyg_data.test_masks, as_tuple=False).squeeze().tolist()

    # Create subgraphs with progress bar
    train_subgraphs = [create_subgraph((pyg_data, idx, k, 0, maxN)) for idx in tqdm(train_indices, desc='Processing train subgraphs')]
    valid_subgraphs = [create_subgraph((pyg_data, idx, k, 1, maxN)) for idx in tqdm(vali_indices, desc='Processing validation subgraphs')]
    test_subgraphs = [create_subgraph((pyg_data, idx, k, 2, maxN)) for idx in tqdm(test_indices, desc='Processing test subgraphs')]


    print(f"Created {len(train_subgraphs)} training subgraphs.")
    print(f"Created {len(valid_subgraphs)} validation subgraphs.")
    print(f"Created {len(test_subgraphs)} test subgraphs.")

    subgraph_dataset = train_subgraphs + valid_subgraphs + test_subgraphs
        
    # Create the InMemoryDataset
    dataset = SubgraphDataset(root='./pyg_dataset', dataset_name=name, subgraph_data_list=subgraph_dataset)
    
    if use_norm and use_svd:
        torch.save(dataset, f'./pyg_dataset/{name}_{k}hop_norm_svd{svd_out_channels}.pt')
    elif use_norm:
        torch.save(dataset, f'./pyg_dataset/{name}_{k}hop_norm.pt')
    elif use_svd:
        torch.save(dataset, f'./pyg_dataset/{name}_{k}hop_svd{svd_out_channels}.pt')
    else:
        torch.save(dataset, f'./pyg_dataset/{name}_{k}hop.pt')

    
