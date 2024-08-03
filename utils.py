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
from collections import Counter
import netlsd
from tqdm import tqdm
from collections import deque
# cluster-aware sampling

### edge mapping 
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

    if orig_edge_index.numel() == 0:
        print(orig_edge_index)
        raise ValueError("orig_edge_index is empty.")
    if edge_index.numel() == 0:
        print(edge_index)
        raise ValueError("edge_index is empty.")

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

    edge_ids = edge_to_unique_id(edge_index)
    sorted_edge_ids = torch.sort(edge_ids).values
    duplicates = sorted_edge_ids[1:] == sorted_edge_ids[:-1]
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
        # self.clusters = self.cluster_anomalous_nodes()

    def split(self, semi_supervised=True, trial_id=0):
        
        self.trial_id = trial_id
        if semi_supervised:
            trial_id += 10
        self.graph.ndata['train_mask'] = self.graph.ndata['train_masks'][:,trial_id]
        self.graph.ndata['val_mask'] = self.graph.ndata['val_masks'][:,trial_id]
        self.graph.ndata['test_mask'] = self.graph.ndata['test_masks'][:,trial_id]
        print(self.graph.ndata['train_mask'].sum(), self.graph.ndata['val_mask'].sum(), self.graph.ndata['test_mask'].sum())

    def get_pyg_graph(self, save=True):
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
        
        if save:
            torch.save(pyg_graph, f'./pyg_dataset/{self.name}.pt')
        print(pyg_graph)
        return pyg_graph
        
    def get_local_subgraphs(self, maxNodes, maxN, onlyE=True):
        assert hasattr(self, 'trial_id'), "call the split function first"
        anomaly_indices = torch.nonzero(self.pyg_graph.y, as_tuple=False).squeeze().tolist()
        train_indices = torch.nonzero(self.pyg_graph.train_masks[:, self.trial_id], as_tuple=False).squeeze().tolist()  
        train_anomaly_indices = list(set(anomaly_indices).intersection(set(train_indices)))
        
        i = 0
        local_subgraphs = []
        # get subgraphs: implementing cluster-aware sampling 
        while i < maxN:
            node_idx = random.choice(train_anomaly_indices) # cluster-aware sampling 
            subgraph = bfs_subgraph_sampling(pyg_graph=self.pyg_graph, start_node=node_idx, max_nodes=maxNodes, onlyE = onlyE)
            local_subgraphs.append(subgraph)
            i += 1
        
        # save the local subgraphs for examination
        # os.makedirs('./local_subgraphs', exist_ok=True)
        torch.save(local_subgraphs, f'./local_subgraphs/{self.name}.pt')
        # print(f'saved {maxN} local {maxNodes}-subgraphs for {self.name}')    
        
        return local_subgraphs

    def cluster_anomalous_nodes(self, k=10):
        from sklearn.cluster import KMeans
        anomaly_indices = torch.nonzero(self.pyg_graph.y, as_tuple=False).squeeze().tolist()
        train_indices = torch.nonzero(self.pyg_graph.train_masks[:, self.trial_id], as_tuple=False).squeeze().tolist()
        train_anomaly_indices = list(set(anomaly_indices).intersection(set(train_indices)))

        # get 2-hop subgraph around each anomalous node
        hop2_subgraphs = []
        for node_idx in tqdm(train_anomaly_indices):
            hop2_subset, hop2_edge_index, mapping, _ = k_hop_subgraph(node_idx, 2, self.pyg_graph.edge_index, relabel_nodes=True)
            subgraph = Data(x=self.pyg_graph.x[hop2_subset], edge_index=hop2_edge_index, y=self.pyg_graph.y[hop2_subset], num_nodes=len(hop2_subset))
            hop2_subgraphs.append(subgraph)

        print('getting subgraph embeddings')
        subgraph_embeddings = [get_graph_embedding(subgraph) for subgraph in tqdm(hop2_subgraphs)]
        subgraph_embeddings = np.array(subgraph_embeddings)

        kmeans = KMeans(n_clusters=k)
        print('fitting kmeans')
        labels = kmeans.fit_predict(subgraph_embeddings)

        # create a dict to map each anomalous node to its cluster, e.g. {cluster_id: [node1, node2, ...]}
        cluster_dict = {}
        for i, label in enumerate(labels):
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(train_anomaly_indices[i])

        return cluster_dict

def get_graph_embedding(d):
    nx_g = to_networkx(d, to_undirected=True)
    emb = netlsd.heat(nx_g, timescales=np.logspace(-2, 2, 50))  # what is the shape of emb?
    return emb
    
def random_walk_subgraph(pyg_graph, start_node, walk_length, max_nodes, onlyE=False):
    edge_index = to_undirected(pyg_graph.edge_index)
    center_node = start_node
    # Extract a 2-hop subgraph around the start_node
    print('exytracting 2-hop subgraph')
    hop2_subset, hop2_edge_index, mapping, _ = k_hop_subgraph(start_node, num_hops=2, edge_index=edge_index, relabel_nodes=True)
    node_mapping = {i: hop2_subset[i].item() for i in range(len(hop2_subset))}
    print(len(hop2_subset))
    if len(hop2_subset) > max_nodes:
        print('proceeding to random walk')
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
             num_nodes=len(subset), node_mapping=node_mapping, y = y, center_node_idx=center_node)
    return d
    
    
def random_walk(pyg_graph, start_node, walk_length=2):
    walk = [start_node]
    edge_index = pyg_graph.edge_index
    for _ in range(walk_length):
        neighbors = edge_index[1][edge_index[0] == walk[-1]]
        if len(neighbors) == 0:  # If no neighbors, stop the walk
            break
        next_node = np.random.choice(neighbors.cpu().numpy())
        walk.append(next_node)
    return walk




def bfs_subgraph_sampling(pyg_graph, start_node, max_nodes, khops = 2, onlyE=False, seed=0):
    edge_index = to_undirected(pyg_graph.edge_index)
    center_node = start_node

    # Extract a 2-hop subgraph around the start_node
    hop2_subset, hop2_edge_index, mapping, _ = k_hop_subgraph(start_node, num_hops=khops, edge_index=edge_index, relabel_nodes=True)
    node_mapping = {i: hop2_subset[i].item() for i in range(len(hop2_subset))}

    if len(hop2_subset) > max_nodes:
        # BFS to downsample the subgraph while maintaining connectivity
        if seed is not None:
            np.random.seed(seed)
        
        visited = set()
        queue = deque([start_node])
        subset = []

        while queue and len(subset) < max_nodes:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                subset.append(node)
                neighbors = edge_index[1][edge_index[0] == node].cpu().numpy()
                
                np.random.shuffle(neighbors)
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)

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

    if onlyE:
        x = torch.ones((len(subset), 1))
        
    edge_index, edge_attr = remove_self_loops(subg_edge_index, edge_attr)
        
    d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, extra_x=extra_x,
             num_nodes=len(subset), node_mapping=node_mapping, y=y, center_node_idx=center_node)
    return d




### GADBENCH CODE

import random
from models.detector import *
from dgl.data.utils import load_graphs
import os
import json
import wandb
import omegaconf

def setup_wandb(cfg):
    gad_train_config = cfg.gad.train_config
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': f'reddit_local_augmentation', 
              'project': f'GADBench_EXP_v2', 
              'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')

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

    results.transpose().to_excel('results/{}.xlsx'.format(file_id), float_format="%.6f")
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