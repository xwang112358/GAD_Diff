from utils import GADDataset
from torch_geometric.utils import from_dgl, to_networkx, k_hop_subgraph
import argparse
import torch
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.data import Data
from tqdm import tqdm

# subgraph statistics for the dataset
# tolokers 2hop --> mean: 1095, std: 1343; 1hop --> mean: 45, std: 99
# reddit 2hop --> mean: 1936, std: 1569; 1hop --> mean: 15, std: 54
# questions 2hop --> mean: 110, std: 353; 1hop --> mean: 4, std: 15



def get_khop_subgraph_and_count_nodes(pyg_data, node_idx, k):
    subset, edge_index, _, _ = k_hop_subgraph(node_idx, k, pyg_data.edge_index, relabel_nodes=False, directed=False)
    return len(subset), edge_index

def main(config):
    data = GADDataset(config.name)
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

    # print all the attributes of pyg_data

    print(pyg_data)
    subgraph_sizes = []
    for idx in tqdm(range(pyg_data.num_nodes)):
        num_nodes, edge_index = get_khop_subgraph_and_count_nodes(pyg_data, idx, config.khops)
        subgraph_sizes.append(num_nodes)

    # get mean and std of subgraph sizes
    mean = torch.tensor(subgraph_sizes).float().mean()
    std = torch.tensor(subgraph_sizes).float().std()
    print(f"Dataset: {config.name} mean: {mean}, Std: {std}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create k-hop subgraphs from a PyG dataset.')
    parser.add_argument('--name', type=str, default='questions  ', help='Name of the dataset')
    parser.add_argument('--khops', type=int, default=2, help='Number of hops for subgraphs')
    config = parser.parse_args()
    main(config=config)