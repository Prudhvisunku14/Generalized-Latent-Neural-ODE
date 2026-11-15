#!/usr/bin/env bash
# Exact commands for the Model D validation study (2026-11-15). Runs in parallel on CPU.
# Seed 42 of the two-shape model is the existing checkpoint
# checkpoints/improved_two_shapes/D_candidate_lr3e-3.pt (identical configuration).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/validation
T=scripts/02_train_model.py; C=configs/model_d.yaml; O=checkpoints/validation
run() { name=$1; shift; nohup python $T --config $C "$@" --checkpoint $O/$name.pt > logs/validation/$name.log 2>&1 & }
# Phase 1: independent seeds (same data, same protocol)
for s in 123 2024 2025 2026; do run two_shapes_seed$s --seed $s; done
# Phase 3: specialised models. Half the data per epoch -> 2x epochs to match optimiser steps.
run circle_only_seed42 --families circle --epochs 600 --patience 120
run spiral_only_seed42 --families spiral --epochs 600 --patience 120
# Phase 4: one shared model on circle + spiral + ellipse
run three_shapes_seed42 --experiment three_shapes
wait
# Phase 3 addendum (launched once the two-shape seed runs above had finished, to free CPU):
# two extra seeds per specialist so specialist seed variance can be compared with the unified model's.
# for s in 123 2024; do run circle_only_seed$s --families circle --epochs 600 --patience 120 --seed $s
#                       run spiral_only_seed$s --families spiral --epochs 600 --patience 120 --seed $s; done
# Ablation (motivated by the VALIDATION-split observation-count sweep; not Model D):
# nohup python scripts/02_train_model.py --config configs/model_d.yaml --n-obs-min 10 --n-obs-max 50 \
#   --checkpoint checkpoints/validation/ablation_variable_nobs_seed42.pt > logs/validation/ablation_variable_nobs_seed42.log 2>&1 &
