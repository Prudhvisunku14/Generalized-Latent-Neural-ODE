#!/usr/bin/env bash
# Conditioning study: M1 (unified, Model D), M2 (param-matched unconditioned), M3 (dynamics-conditioned),
# plus ablation M2aug (unconditioned, latent 12 = M3's ODE state size). --data-seed = seed, so every model
# at a given seed sees identical batches and training observation subsets.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/conditioning checkpoints/conditioning
run() { m=$1; s=$2; nohup python scripts/02_train_model.py --config configs/conditioning/$m.yaml --seed $s --data-seed $s \
          --checkpoint checkpoints/conditioning/${m}_seed$s.pt > logs/conditioning/${m}_seed$s.log 2>&1 & }
for s in 42 123 2024 2025 2026; do for m in m1_unified m2_param_matched m3_conditioned; do run $m $s; done; done
for s in 42 123 2024; do run m2aug_latent12 $s; done
wait
