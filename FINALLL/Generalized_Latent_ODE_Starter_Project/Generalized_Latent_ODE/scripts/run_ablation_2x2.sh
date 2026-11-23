#!/usr/bin/env bash
# Controlled 2x2 ablation (latent dimension x GRU width), five-shape dataset. Identical to the earlier capacity runs
# (configs/model_d.yaml, --experiment five_shapes, seed 42, 300 epochs, patience 60, 4 threads) plus --data-seed 42.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/ablation_2x2 checkpoints/ablation_2x2
T=scripts/02_train_model.py; C=configs/model_d.yaml; O=checkpoints/ablation_2x2
run() { name=$1; shift; nohup python $T --config $C --experiment five_shapes --seed 42 --data-seed 42 "$@" --checkpoint $O/$name.pt > logs/ablation_2x2/$name.log 2>&1 & }
run five_shapes_A_lat8_gru64_seed42   --latent-dim 8  --encoder-hidden 64
run five_shapes_B_lat16_gru64_seed42  --latent-dim 16 --encoder-hidden 64
run five_shapes_C_lat8_gru128_seed42  --latent-dim 8  --encoder-hidden 128
run five_shapes_D_lat16_gru128_seed42 --latent-dim 16 --encoder-hidden 128
wait
