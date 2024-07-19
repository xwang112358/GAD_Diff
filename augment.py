import torch
from torch_geometric.utils import from_dgl, to_networkx, k_hop_subgraph
import random
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected
from torch_cluster import random_walk
from utils import *
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import sys
import os
from omegaconf import DictConfig
# local_subgraph = torch.load('local_subgraphs/reddit_0.pt')
# dataloader = DataLoader(local_subgraph, batch_size=5, shuffle=False)
# for i, batch in enumerate(dataloader):
#     print(i, batch)
#     break
import hydra
from diffusion_model.dataset import init_dataset, compute_input_output_dims 
from diffusion_model.extra_features import ExtraFeatures, DummyExtraFeatures
from diffusion_model.diffusion_discrete import DiscreteDenoisingDiffusion
from diffusion_model.analysis.spectre_utils import CrossDomainSamplingMetrics
from diffusion_model import utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'diffusion_model')))


@hydra.main(version_base='1.3', config_path='./configs', config_name='config')
def main(cfg: DictConfig):

    print(hydra.utils.get_original_cwd())
    hydra_path = hydra.utils.get_original_cwd() # /home/allenwang/gad_diff/GAD_Diff

    subgraph = torch.load('local_subgraphs/reddit_0.pt')
    print(subgraph[0])
    # print('node mapping', subgraph[0].node_mapping)  # double check the labeled = True 
    original_data = torch.load('pyg_dataset/undirected_reddit.pt')
    orig_edge_index = original_data.edge_index 
    print(original_data)

    aug_loader = DataLoader(subgraph, batch_size=1, shuffle=False) # keep batch size as 1 now 
    num_classes, max_n_nodes, nodes_dist, edge_types, node_types, n_nodes = init_dataset('reddit', aug_loader)


    extra_features = ExtraFeatures(cfg.model.extra_features, max_n_nodes)
    domain_features = DummyExtraFeatures()   
    input_dims, output_dims = compute_input_output_dims(aug_loader, extra_features, domain_features)
    print(input_dims, output_dims)

    sampling_metrics = CrossDomainSamplingMetrics(aug_loader)
    model = DiscreteDenoisingDiffusion(cfg, input_dims, output_dims, nodes_dist, node_types, edge_types, extra_features, domain_features, aug_loader, sampling_metrics, augment=True) 


    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/E_kl', # epoch_NLL
                                              save_top_k=3,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)


    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)


    use_gpu = 1>0 and torch.cuda.is_available() # multiple gpus
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      log_every_n_steps=50,
                      logger = [],
                      accumulate_grad_batches=cfg.train.accumulate_grad_batches)

    # get the augmented data (edge_index)
    augment_samples = trainer.predict(model, aug_loader, ckpt_path = 'checkpoints/reddit/reddit_onehot.ckpt')
    print(type(augment_samples))



    # mapping the augmented data to the original data
    for i, batch in enumerate(aug_loader):
        print(i, batch)
        print(batch.edge_index.shape)

        # print(batch.node_mapping.shape) 
        # flat_tensor = batch.edge_index.flatten() 
        # unique_values = torch.unique(flat_tensor)
        # print(unique_values.numel())

        edge_index = batch.edge_index
        node_mapping = batch.node_mapping
        remapped_edge_index = node_mapping[edge_index]

        # remove the original subgraph topology
        assert_edges_exist(orig_edge_index, remapped_edge_index)
        removed_edge_index = remove_edges(orig_edge_index, remapped_edge_index)
        print(count_duplicate_edges(remapped_edge_index))

        print(orig_edge_index.shape)
        print(removed_edge_index.shape)
        print(remapped_edge_index.shape)

        augmented_data = augment_samples[i][0]
        print(augmented_data.edge_index.shape, edge_index.shape)

        break




if __name__ == '__main__':
    main()
