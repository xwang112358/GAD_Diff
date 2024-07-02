from utils import GADDataset
from torch_geometric.utils import from_dgl, to_networkx, k_hop_subgraph
import argparse
import torch
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.data import Data
from tqdm import tqdm

# subgraph statistics for the dataset
# tolokers all 2hop --> mean: 1095, std: 1343; 1hop --> mean: 45, std: 99; # train anomaly 1-hop (72, 163) 2-hop (1347, 1647)
# reddit(undirected) 2hop --> mean: 1936, std: 1569; 1hop --> mean: 15, std: 54; # train anomaly 1-hop (12, 12) 2-hop (2459, 1670)
# questions 2hop --> mean: 110, std: 353; 1hop --> mean: 4, std: 15    # train anomaly 1-hop (8, 23) 2-hop (240, 611)

# use source to target get the computation graph

def get_khop_subgraph_and_count_nodes(pyg_data, node_idx, k):
    subset, edge_index, _, _ = k_hop_subgraph(node_idx, k, pyg_data.edge_index, relabel_nodes=False, directed=False)
    return len(subset), edge_index

def main(config):
    data = GADDataset(config.name)
    print(data.graph.number_of_nodes())
    print(data.graph.number_of_edges())

    pyg_data = from_dgl(data.graph)
    
    pyg_data.edge_index = pyg_data.edge_index.long()

    print(pyg_data.edge_index.size(1))

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
    torch.save(pyg_data, f'./pyg_dataset/{config.name}.pt')
    # pyg_data.edge_index = to_undirected(pyg_data.edge_index)
    subgraph_sizes = []
    for idx in tqdm(range(pyg_data.num_nodes)):
        num_nodes, edge_index = get_khop_subgraph_and_count_nodes(pyg_data, idx, config.khops)
        subgraph_sizes.append(num_nodes)

    # get mean and std of subgraph sizes
    mean = torch.tensor(subgraph_sizes).float().mean()
    std = torch.tensor(subgraph_sizes).float().std()
    print(f"Dataset {config.name}: for all {config.khops} subgraph mean # nodes: {mean}, std: {std}")

    # get idx where pyg.y == 1
    anomaly_idx = (pyg_data.y == 1).nonzero().squeeze()
    train_indices = torch.nonzero(pyg_data.train_masks, as_tuple=False).squeeze().tolist()
    train_anomaly_idx = list(set(anomaly_idx.tolist()).intersection(set(train_indices)))

    print(f"Dataset {config.name}: # anomalies in train set: {len(train_anomaly_idx)}")
    train_anomaly_subgraph_sizes = []
    for idx in train_anomaly_idx:
        num_nodes, edge_index = get_khop_subgraph_and_count_nodes(pyg_data, idx, config.khops)
        train_anomaly_subgraph_sizes.append(num_nodes)
    train_anomaly_mean = torch.tensor(train_anomaly_subgraph_sizes).float().mean()
    train_anomaly_std = torch.tensor(train_anomaly_subgraph_sizes).float().std()

    # draw histogram of subgraph sizes for anomalies in train set
    import matplotlib.pyplot as plt

    plt.hist(train_anomaly_subgraph_sizes, bins=20)
    plt.xlabel('Subgraph Size')
    plt.ylabel('Frequency')
    plt.title(f'{config.name} {config.khops}-hop Subgraph Sizes for Anomalies in Train Set')

    plt.savefig(f'./figures/{config.name}.png')
    plt.close()

    print(f"Dataset {config.name}: for all {config.khops}-hop subgraph mean # nodes for anomalies in train set: {train_anomaly_mean}, std: {train_anomaly_std}")
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create k-hop subgraphs from a PyG dataset.')
    parser.add_argument('--name', type=str, default='questions  ', help='Name of the dataset')
    parser.add_argument('--khops', type=int, default=2, help='Number of hops for subgraphs')
    config = parser.parse_args()
    main(config=config)