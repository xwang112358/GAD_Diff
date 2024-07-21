import random
from dgl.data.utils import load_graphs
import os
import json
import torch
from torch_geometric.data import Data
from typing import Optional, Tuple, Union
from torch import Tensor
from torch_geometric.data import InMemoryDataset
import copy
from abc import ABC, abstractmethod
from typing import Any, List, Tuple
import torch
from torch import Tensor
from torch_geometric.utils import subgraph, to_undirected, remove_self_loops
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_torch_csc_tensor
from torch_geometric.utils import to_networkx, k_hop_subgraph, to_dense_adj, from_dgl

### edge mapping 

def add_edges_to_edge_index(edge_index, new_edges):
    """
    Adds new edges to an existing edge_index tensor.

    Parameters:
    - edge_index (torch.Tensor): The original edge_index tensor of shape [2, num_edges].
    - new_edges (torch.Tensor): The new edges to be added, also of shape [2, num_new_edges].

    Returns:
    - torch.Tensor: Updated edge_index tensor containing the original and new edges.
    """
    # Concatenate the original edge_index with the new edges
    updated_edge_index = torch.cat([edge_index, new_edges], dim=1)

    # Optional: Remove duplicates to clean up the edge list
    updated_edge_index = remove_duplicates_from_edge_index(updated_edge_index)

    return updated_edge_index

def remove_duplicates_from_edge_index(edge_index):
    """
    Remove duplicate edges from edge_index.

    Parameters:
    - edge_index (torch.Tensor): The edge_index tensor of shape [2, num_edges].

    Returns:
    - torch.Tensor: Edge_index tensor with duplicates removed.
    """
    # Sort edge pairs and then find unique columns
    sorted_edge_index, _ = torch.sort(edge_index, dim=0)
    unique_edge_index = torch.unique(sorted_edge_index.t().contiguous(), dim=0)

    return unique_edge_index.t()


def assert_edges_exist(orig_edge_index, edge_index):
    # Function to convert [2, num_edges] to a unique identifier for each edge
    def edge_to_unique_id(edge_index):
        max_node = max(orig_edge_index.max(), edge_index.max()) + 1
        return edge_index[0] * max_node + edge_index[1]

    # Convert edge indices to unique identifiers
    orig_edge_ids = edge_to_unique_id(orig_edge_index)
    edge_ids = edge_to_unique_id(edge_index)

    # Check for each edge in edge_index if it exists in orig_edge_index
    exists = torch.isin(edge_ids, orig_edge_ids)

    # Assert all edges in edge_index exist in orig_edge_index
    assert exists.all(), "Some edges in edge_index do not exist in orig_edge_index."


def remove_edges(orig_edge_index, edge_index):
    # Function to convert [2, num_edges] to unique identifier format for each edge
    def edge_to_tuple_tensor(edge_index):
        return torch.cat((edge_index[0].unsqueeze(1), edge_index[1].unsqueeze(1)), dim=1)

    # Convert edge indices to tuple format
    orig_edges = edge_to_tuple_tensor(orig_edge_index)
    remove_edges = edge_to_tuple_tensor(edge_index)

    # Use a unique representation of each edge for easy comparison
    # Convert each pair to a large number (assuming max node index is reasonably small)
    max_node = max(orig_edge_index.max(), edge_index.max()) + 1
    orig_edges_flat = orig_edges[:, 0] * max_node + orig_edges[:, 1]
    remove_edges_flat = remove_edges[:, 0] * max_node + remove_edges[:, 1]

    # Create a mask for original edges not in remove_edges
    remaining_edges_mask = ~(orig_edges_flat.unsqueeze(1) == remove_edges_flat).any(dim=1)

    # Filter the original edges using the mask
    remaining_edges = orig_edge_index[:, remaining_edges_mask]

    return remaining_edges

def count_duplicate_edges(edge_index):
    # Function to convert [2, num_edges] to a unique identifier for each edge
    def edge_to_unique_id(edge_index):
        max_node = edge_index.max() + 1
        return edge_index[0] * max_node + edge_index[1]

    # Convert edge indices to unique identifiers
    edge_ids = edge_to_unique_id(edge_index)

    # Sort edge identifiers to group duplicates together
    sorted_edge_ids = torch.sort(edge_ids).values

    # Find duplicates: where the shifted values are equal, there is a duplicate
    duplicates = sorted_edge_ids[1:] == sorted_edge_ids[:-1]

    # Count the number of true values, which represent duplicates
    num_duplicates = duplicates.sum().item()

    return num_duplicates


### subgraph sampling

class GADDataset:
    def __init__(self, name='tfinance', prefix='datasets/'):
        graph = load_graphs(prefix + name)[0][0]
        self.name = name
        self.graph = graph
        # self.khop_subgraph = []
        self.pyg_graph = self.get_pyg_graph()

    def split(self, semi_supervised=True, trial_id=0):
        
        self.trial_id = trial_id
        if semi_supervised:
            trial_id += 10
        self.graph.ndata['train_mask'] = self.graph.ndata['train_masks'][:,trial_id]
        self.graph.ndata['val_mask'] = self.graph.ndata['val_masks'][:,trial_id]
        self.graph.ndata['test_mask'] = self.graph.ndata['test_masks'][:,trial_id]
        print(self.graph.ndata['train_mask'].sum(), self.graph.ndata['val_mask'].sum(), self.graph.ndata['test_mask'].sum())

    def get_pyg_graph(self):
        pyg_graph = from_dgl(self.graph)
        pyg_graph.edge_index = pyg_graph.edge_index.long()
        
        if hasattr(pyg_graph, 'train_masks'):
            pyg_graph.train_masks = self.graph.ndata['train_masks']
        if hasattr(pyg_graph, 'val_masks'):
            pyg_graph.val_masks = self.graph.ndata['val_masks']
        if hasattr(pyg_graph, 'test_masks'):
            pyg_graph.test_masks = self.graph.ndata['test_masks']
        if hasattr(pyg_graph, 'count'):
            pyg_graph.num_nodes = self.graph.number_of_nodes()
            del pyg_graph.count 
        if hasattr(pyg_graph, 'label'):
            pyg_graph.y = pyg_graph.label
            del pyg_graph.label
        if hasattr(pyg_graph, 'feature'):
            pyg_graph.x = pyg_graph.feature
            del pyg_graph.feature
            
        torch.save(pyg_graph, f'./pyg_dataset/{self.name}.pt')
        print(pyg_graph)
        return pyg_graph
        
    def get_local_subgraphs(self, maxNodes, maxN):
        assert hasattr(self, 'trial_id'), "call the split function first"
        anomaly_indices = torch.nonzero(self.pyg_graph.y, as_tuple=False).squeeze().tolist()
        train_indices = torch.nonzero(self.pyg_graph.train_masks[:, self.trial_id], as_tuple=False).squeeze().tolist()  
        train_anomaly_indices = list(set(anomaly_indices).intersection(set(train_indices)))
        
        i = 0
        local_subgraphs = []
        # get subgraphs 
        while i < maxN:
            node_idx = random.choice(train_anomaly_indices)
            subgraph = get_khop_subgraph(self.pyg_graph, node_idx, maxNodes)
            local_subgraphs.append(subgraph)
            i += 1
        
        # save the local subgraphs for examination
        os.makedirs('./local_subgraphs', exist_ok=True)
        torch.save(local_subgraphs, f'./local_subgraphs/{self.name}_{self.trial_id}.pt')
        print(f'saved {maxN} local {maxNodes}-subgraphs for {self.name}')    
        
        return local_subgraphs
        
    

# cluster aware sampling 

def get_khop_subgraph_v1(pyg_data, node_idx, maxN):
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
            walks = random_walk_until_maxN(hop2_edge_index[0], hop2_edge_index[1], torch.tensor([walk_start]), maxN, walk_length=5)
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


def get_khop_subgraph(pyg_data, node_idx, maxN, onlyE=False):
    # Extract 2-hop subgraph
    hop2_subset, hop2_edge_index, hop2_mapping, hop2_edge_mask = k_hop_subgraph(
        node_idx, 2, pyg_data.edge_index, relabel_nodes=True, flow='source_to_target')

    # print('hop2_subset: ', len(hop2_subset))
    hop2_edge_index = to_undirected(hop2_edge_index)
    relabeled_to_original = {i: hop2_subset[i].item() for i in range(len(hop2_subset))}
    # print(relabeled_to_original)
    # print('first relabeled_to_original: ', len(relabeled_to_original))
    if len(hop2_subset) > maxN:
        walk_start = hop2_mapping[0].item() # center node
        walks = random_walk_until_maxN(hop2_edge_index[0], hop2_edge_index[1], torch.tensor([walk_start]), maxN, walk_length=5)
        subsample_subset = torch.unique(walks.flatten())
        
        while len(subsample_subset) < maxN:
            walks = random_walk_until_maxN(hop2_edge_index[0], hop2_edge_index[1], torch.tensor([walk_start]), maxN, walk_length=6)
            subsample_subset = torch.unique(torch.cat((subsample_subset, torch.unique(walks.flatten()))))

        if len(subsample_subset) > maxN:
            subsample_subset = subsample_subset[:maxN]
        # update relabeled_to_original by removing nodes not in subsample_subset
        # print('subsample_subset: ', len(subsample_subset))
        relabeled_to_original = {k: v for k, v in relabeled_to_original.items() if k in subsample_subset}
        
        # print('second relabeled_to_original: ', len(relabeled_to_original))
        # print(relabeled_to_original)
        # reindex keys in relabeled_to_original from 0 to len(relabeled_to_original)
        
        relabeled_to_original = {i: relabeled_to_original[k] for i, k in enumerate(sorted(relabeled_to_original.keys()))}
        
        # print('third relabeled_to_original: ', relabeled_to_original)        

        subsample_edge_index, subsample_edge_mask = subgraph(subsample_subset, hop2_edge_index, relabel_nodes=True)
    else:
        subsample_subset = hop2_subset
        subsample_edge_index = hop2_edge_index
    
    # dict is not iterable by pytorch dataloader, convert relabeled_to_original to tensor
    relabeled_to_original = torch.tensor(list(relabeled_to_original.values()))

    x = pyg_data.y[subsample_subset]
    # convert x to one-hot encoding
    x = torch.nn.functional.one_hot(x, num_classes=2).float()
    edge_index = subsample_edge_index
    edge_attr = torch.tensor([[0, 1] for _ in range(subsample_edge_index.shape[1])])
    extra_x = pyg_data.x[subsample_subset]
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    y = y = torch.empty(1, 0)

    if onlyE:
        print('onlyE')
        x = torch.ones((len(subsample_subset), 1))
        
    subgraph_data = Data(x=x, edge_index=edge_index, edge_attr = edge_attr, extra_x = extra_x,
                          num_nodes=len(subsample_subset), node_mapping = relabeled_to_original, y=y)

    subgraph_data.center_node_idx = node_idx

    return subgraph_data


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


### GADBENCH CODE

import random
from models.detector import *
from dgl.data.utils import load_graphs
import os
import json


class Dataset:
    def __init__(self, name='tfinance', prefix='datasets/'):
        graph = load_graphs(prefix + name)[0][0]
        self.name = name
        self.graph = graph

    def split(self, semi_supervised=True, trial_id=0):
        assert isinstance(trial_id, int) and trial_id < 10, "trial_id must be an integer smaller than 10."
        if semi_supervised:
            trial_id += 10
        self.graph.ndata['train_mask'] = self.graph.ndata['train_masks'][:,trial_id]
        self.graph.ndata['val_mask'] = self.graph.ndata['val_masks'][:,trial_id]
        self.graph.ndata['test_mask'] = self.graph.ndata['test_masks'][:,trial_id]
        print("Train nodes:", self.graph.ndata['train_mask'].sum(), "Valid nodes:", self.graph.ndata['val_mask'].sum(), "Test nodes:", self.graph.ndata['test_mask'].sum())


dataset_dict = {
    0: 'reddit',
    1: 'weibo',
    2: 'amazon',
    3: 'yelp',
    4: 'tfinance',
    5: 'elliptic',
    6: 'tolokers',
    7: 'questions',
    8: 'dgraphfin',
    9: 'tsocial',
    10: 'hetero/amazon',
    11: 'hetero/yelp',   
}


model_detector_dict = {
    # Classic Methods
    'MLP': BaseGNNDetector,
    'KNN': KNNDetector,
    'SVM': SVMDetector,
    'RF': RFDetector,
    'XGBoost': XGBoostDetector,
    'XGBOD': XGBODDetector,
    'NA': XGBNADetector,

    # Standard GNNs
    'GCN': BaseGNNDetector,
    'SGC': BaseGNNDetector,
    'GIN': BaseGNNDetector,
    'GraphSAGE': BaseGNNDetector,
    'GAT': BaseGNNDetector,
    'GT': BaseGNNDetector,
    'PNA': BaseGNNDetector,
    'BGNN': BGNNDetector,

    # Specialized GNNs
    'GAS': GASDetector,
    'BernNet': BaseGNNDetector,
    'AMNet': BaseGNNDetector,
    'BWGNN': BaseGNNDetector,
    'GHRN': GHRNDetector,
    'GATSep': BaseGNNDetector,
    'PCGNN': PCGNNDetector,
    'DCI': DCIDetector,

    # Heterogeneous GNNs
    'RGCN': HeteroGNNDetector,
    'HGT': HeteroGNNDetector,
    'CAREGNN': CAREGNNDetector,
    'H2FD': H2FDetector, 

    # Tree Ensembles with Neighbor Aggregation
    'RFGraph': RFGraphDetector,
    'XGBGraph': XGBGraphDetector,
}


def save_results(results, file_id):
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if file_id is None:
        file_id = 0
        while os.path.exists('results/{}.xlsx'.format(file_id)):
            file_id += 1

    results.transpose().to_excel('results/{}.xlsx'.format(file_id))
    print('save to file ID: {}'.format(file_id))
    return file_id


def sample_param(model, dataset, t=0):
    model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0}
    if t == 0:
        return model_config
    for k, v in param_space[model].items():
        model_config[k] = random.choice(v)
    # Avoid OOM in Random Search
    if model in ['GAT', 'GATSep', 'GT'] and dataset in ['tfinance', 'dgraphfin', 'tsocial']:
        model_config['h_feats'] = 16
        model_config['num_heads'] = 2
    if dataset == 'tsocial':
        model_config['h_feats'] = 16
    if dataset in ['dgraphfin', 'tsocial']:
        if 'k' in model_config:
            model_config['k'] = min(5, model_config['k'])
        if 'num_cluster' in model_config:
            model_config['num_cluster'] = 2
        # if 'num_layers' in model_config:
        #     model_config['num_layers'] = min(2, model_config['num_layers'])
    return model_config


param_space = {}

param_space['MLP'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['GCN'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['SGC'] = {
    'h_feats': [16, 32, 64],
    'k': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'mlp_layers': [1, 2],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['GIN'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'agg': ['sum', 'max', 'mean'],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['GraphSAGE'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'agg': ['mean', 'gcn', 'pool'],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['ChebNet'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'mlp_layers': [1, 2],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['BernNet'] = {
    'h_feats': [16, 32, 64],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'mlp_layers': [1, 2],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'orders': [2, 3, 4, 5],
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['AMNet'] = {
    'h_feats': [16, 32, 64],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'orders': [2, 3],
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['BWGNN'] = {
    'h_feats': [16, 32, 64],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'mlp_layers': [1, 2],
    'activation': ['ReLU', 'LeakyReLU', 'Tanh'],
}

param_space['GAS'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'k': range(3, 51),
    'dist': ['euclidean', 'cosine'],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['GHRN'] = {
    'h_feats': [16, 32, 64],
    'del_ratio': 10 ** np.linspace(-2, -1, 1000),
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'mlp_layers': [1, 2],
}

param_space['KNNGCN'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'k': list(range(3, 51)),
    'dist': ['euclidean', 'cosine'],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['XGBoost'] = {
    'n_estimators': list(range(10, 201)),
    'eta': 0.5 * 10 ** np.linspace(-1, 0, 1000),
    'lambda': [0, 1, 10],
    'subsample': [0.5, 0.75, 1]
}

param_space['XGBGraph'] = {
    'n_estimators': list(range(10, 201)),
    'eta': 0.5 * 10 ** np.linspace(-1, 0, 1000),
    'lambda': [0, 1, 10],
    # 'alpha': [0, 0.5, 1],
    'subsample': [0.5, 0.75, 1],
    'num_layers': [1, 2, 3, 4],
    'agg': ['sum', 'max', 'mean'],
    'booster': ['gbtree', 'dart']
}

param_space['RF'] = {
    'n_estimators': list(range(10, 201)),
    'criterion': ['gini', 'entropy'],
    'max_samples': list(np.linspace(0.1, 1, 1000)),
}

param_space['RFGraph'] = {
    'n_estimators': list(range(10, 201)),
    'criterion': ['gini', 'entropy'],
    'max_samples': [0.5, 0.75, 1],
    'max_features': ['sqrt', 'log2', None],
    'num_layers': [1, 2, 3, 4],
    'agg': ['sum', 'max', 'mean'],
}

param_space['SVM'] = {
    'weights': ['uniform', 'distance'],
    'C': list(10 ** np.linspace(-1, 1, 1000))
}

param_space['KNN'] = {
    'k': list(range(3, 51)),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

param_space['XGBOD'] = {
    'n_estimators': list(range(10, 201)),
    'learning_rate': 0.5 * 10 ** np.linspace(-1, 0, 1000),  # [0.05, 0.1, 0.2, 0.3, 0.5],
    'lambda': [0, 1, 10],
    'subsample': [0.5, 0.75, 1],
    'booster': ['gbtree', 'dart']
}

param_space['GAT'] = {
    'h_feats': [16, 32],
    'num_heads': [1, 2, 4, 8],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['GATSep'] = {
    'h_feats': [16, 32],
    'num_heads': [1, 2, 4, 8],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['GT'] = {
    'h_feats': [16, 32],
    'num_heads': [1, 2, 4, 8],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['PCGNN'] = {
    'h_feats': [16, 32, 64],
    'del_ratio': np.linspace(0.01, 0.8, 1000),
    'add_ratio': np.linspace(0.01, 0.8, 1000),
    'dist': ['euclidean', 'cosine'],
    # 'k': list(range(3, 10)),
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['DCI'] = {
    'h_feats': [16, 32, 64],
    'pretrain_epochs': [20, 50, 100],
    'num_cluster': list(range(2,31)),
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['BGNN'] = {
    'depth': [4,5,6,7],
    'iter_per_epoch': [2,5,10,20],
    'gbdt_lr': 10 ** np.linspace(-2, -0.5, 1000),
    'normalize_features': [True, False],
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['NA'] = {
    'n_estimators': list(range(10, 201)),
    'eta': 0.5 * 10 ** np.linspace(-1, 0, 1000),
    'lambda': [0, 1, 10],
    'subsample': [0.5, 0.75, 1],
    'k': list(range(0, 51)),
}

param_space['PNA'] = {
    'h_feats': [16, 32, 64],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh'],
}