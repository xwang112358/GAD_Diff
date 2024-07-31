#!/bin/bash

# Declare parameters
aug_intervals=(20 30 40)
num_subgraphs=(15 30 45)
diffusion_steps=(10 25 50)
start_aug_epoch=50

# Loop over all combinations of parameter values
for aug_interval in "${aug_intervals[@]}"; do
    for num_subgraph in "${num_subgraphs[@]}"; do
        for diffusion_step in "${diffusion_steps[@]}"; do
            echo "Running with start_aug_epoch=$start_aug_epoch, aug_interval=$aug_interval, NumSubgraphs=$num_subgraph, diffusion_steps=$diffusion_step"
            python3 benchmark_new.py model.transition="marginal" augment.start_aug_epoch=$start_aug_epoch augment.aug_interval=$aug_interval augment.NumSubgraphs=$num_subgraph augment.diffusion_steps=$diffusion_step
        done
    done
done
