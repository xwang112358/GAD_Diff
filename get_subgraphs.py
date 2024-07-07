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
from utils import GADDataset, random_walk_until_maxN
import os
import random
from torch_geometric.transforms import NormalizeFeatures, SVDFeatureReduction

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

def get_khop_subgraph(pyg_data, node_idx, maxN):
    # Extract 2-hop subgraph
    hop2_subset, hop2_edge_index, hop2_mapping, hop2_edge_mask = k_hop_subgraph(
        node_idx, 2, pyg_data.edge_index, relabel_nodes=True, flow='source_to_target')

    # Convert the 2-hop subgraph to undirected
    hop2_edge_index = to_undirected(hop2_edge_index)

    # Perform random walk to sample nodes until maxN unique nodes are reached
    if len(hop2_subset) > maxN:
        walk_start = hop2_mapping[0].item() # center node
        walks = random_walk_until_maxN(hop2_edge_index[0], hop2_edge_index[1], torch.tensor([walk_start]), maxN, walk_length=5)
        subsample_subset = torch.unique(walks.flatten())
        
        while len(subsample_subset) < maxN:
            # keep random walking until we have maxN unique nodes
            walks = random_walk_until_maxN(hop2_edge_index[0], hop2_edge_index[1], torch.tensor([walk_start]), maxN, walk_length=6)
            # concatenate the new walks with the previous walks
            subsample_subset = torch.unique(torch.cat((subsample_subset, torch.unique(walks.flatten()))))

        if len(subsample_subset) > maxN:
            subsample_subset = subsample_subset[:maxN]

        subsample_edge_index, subsample_edge_mask = subgraph(subsample_subset, hop2_edge_index, relabel_nodes=True)
    else:
        subsample_subset = hop2_subset
        subsample_edge_index = hop2_edge_index

    # Create subgraph data object
    subgraph_data = Data(x=pyg_data.x[subsample_subset], edge_index=subsample_edge_index, y = pyg_data.y[subsample_subset],
                          num_nodes=len(subsample_subset))

    # Set the label for the subgraph based on the original node's label
    # label = pyg_data.y[node_idx].item()
    # subgraph_data.y = torch.tensor([label], dtype=torch.long)
    subgraph_data.center_node_idx = node_idx

    return subgraph_data


def create_subgraph(args):
    # record the size of the original k-hop subgraph
    # if the size is larger than maxN, then sample multiple subgraphs with random walk
    pyg_data, node_idx, split, maxN = args
    subgraph_data = get_khop_subgraph(pyg_data, node_idx, maxN)
    subgraph_data.split = split
    return subgraph_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create k-hop subgraphs from a PyG dataset.')
    parser.add_argument('--name', type=str, default='questions  ', help='Name of the dataset')
    # parser.add_argument('--khops', type=int, default=2, help='Number of hops for subgraphs')
    parser.add_argument('--maxN', type=int, default=150, help='Largest number of nodes allowed in the subgraph')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_subgraphs', type=int, default=1000, help='Number of subgraphs to create')
    parser.add_argument('--add_labels', action='store_true', help='Whether to add node labels to the subgraphs')
    parser.add_argument('--add_continousX', action='store_true', help='Whether to add continuous node features to the subgraphs')
    # parser.add_argument('--svd_out_channels', type=int, default=64, help='Number of output channels of SVD')
    # parser.add_argument('--use_svd', action='store_true', help='whether to use svd to conduct dimension reduction')
    # parser.add_argument('--use_norm', action='store_true', help='whether to use normalize the node features')
    
    args = parser.parse_args()
    # k = args.khops
    # use_svd = args.use_svd
    # svd_out_channels = args.svd_out_channels
    # use_norm = args.use_norm
    
    try:
        pyg_data = torch.load(f'./pyg_dataset/{args.name}.pt')
        
        # if use_norm:
        #     normalize = NormalizeFeatures()
        #     pyg_data = normalize(pyg_data)
        # if use_svd:
        #     svd = SVDFeatureReduction(svd_out_channels)
        #     pyg_data = svd(pyg_data)
            
        print(f"Loaded {args.name} PyG data")
        print(pyg_data.num_nodes)

    except FileNotFoundError:
        data = GADDataset(args.name)
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
        
        torch.save(pyg_data, f'./pyg_dataset/{args.name}.pt')
        print(pyg_data)

    # Ensure edges are undirected
    pyg_data.edge_index = to_undirected(pyg_data.edge_index)

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
    # test_anomaly_indices = list(set(anomaly_indices).intersection(set(test_indices)))
    
    print(111)
    print(pyg_data.y)

    # Create subgraphs with progress bar
    train_subgraphs = [create_subgraph((pyg_data, idx, 0, args.maxN)) for idx in tqdm(train_anomaly_indices, desc='Processing train subgraphs')]
    valid_subgraphs = [create_subgraph((pyg_data, idx, 1, args.maxN)) for idx in tqdm(valid_anomaly_indices, desc='Processing validation subgraphs')]
    # test_subgraphs = [create_subgraph((pyg_data, idx, 2, args.maxN)) for idx in tqdm(valid_anomaly_indices, desc='Processing test subgraphs')]

    # calculate the average clustering coefficient of the subgraphs
    # train_clustering = [nx.average_clustering(to_networkx(subgraph)) for subgraph in train_subgraphs]
    # print(f"Average clustering coefficient of train subgraphs: {np.mean(train_clustering)}")
    
    print(f"Created {len(train_subgraphs)} training subgraphs.")
    print(f"Created {len(valid_subgraphs)} validation subgraphs.")
    # print(f"Created {len(test_subgraphs)} test subgraphs.")

    subgraph_dataset = train_subgraphs + valid_subgraphs # + test_subgraphs


    for graph in subgraph_dataset:
        print(graph.x)
        print(graph.y)
        break

    if not os.path.exists(f'./pyg_dataset/{args.name}'):
        os.makedirs(f'./pyg_dataset/{args.name}')

    if args.add_labels:
        train_data = [(to_dense_adj(subgraph.edge_index).squeeze(0), F.one_hot(subgraph.y, num_classes=2)) for subgraph in subgraph_dataset]
        torch.save(train_data, f'./pyg_dataset/{args.name}/{args.name}_onehot.pt')
        print(f"Saved {args.name}_onehot.pt")

    elif args.add_continousX:
        train_data = [(to_dense_adj(subgraph.edge_index).squeeze(0), torch.cat([subgraph.x, subgraph.y.unsqueeze(1)], dim=1)) for subgraph in subgraph_dataset]
        torch.save(train_data, f'./pyg_dataset/{args.name}/{args.name}_continuous.pt')
        print(f"Saved {args.name}_continuous.pt")

    else:
        train_data = [to_dense_adj(subgraph.edge_index).squeeze(0) for subgraph in subgraph_dataset]
        torch.save(train_data, f'./pyg_dataset/{args.name}/{args.name}_adj.pt')
        print(f"Saved {args.name}_adj.pt")
        
        
    # print the size of each train_adj
    for adj, x in train_data:
        print(adj.shape, x.shape)
        print(x)
        break


        


    
