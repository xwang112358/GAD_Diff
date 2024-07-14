#!/bin/bash

echo "Start training..."

logfile="run_train_output.txt"

exec > >(tee -a "$logfile") 2>&1

python benchmark.py --trial 3 --datasets 0 --models GCN
echo "Training finished."