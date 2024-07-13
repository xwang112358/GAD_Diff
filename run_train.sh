#!/bin/bash

echo "Start training..."

logfile="run_train_output.txt"

exec > >(tee -a "$logfile") 2>&1

python benchmark.py --trial 21 --datasets 0,6,7 --models GCN-GIN

echo "Training finished."