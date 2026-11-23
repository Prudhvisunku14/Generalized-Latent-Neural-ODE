#!/usr/bin/env bash
# Capacity-scaling study (five-shape dataset). Identical to the existing five-shape run
# (configs/model_d.yaml, --experiment five_shapes, seed 42, 300 epochs, patience 60, 4 threads)
# except latent_dim and encoder_hidden (configs/capacity_scaling.yaml). The baseline is NOT retrained.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/capacity_scaling checkpoints/capacity_scaling
T=scripts/02_train_model.py; C=configs/model_d.yaml; O=checkpoints/capacity_scaling
run() { name=$1; shift; nohup python $T --config $C --experiment five_shapes "$@" --checkpoint $O/$name.pt > logs/capacity_scaling/$name.log 2>&1 & }
run five_shapes_medium_seed42 --latent-dim 16 --encoder-hidden 128
run five_shapes_large_seed42 --latent-dim 32 --encoder-hidden 128
wait
