# Experiment 5: model capacity scaling

## 1. Objective
How do reconstruction and forecasting change with model size on the five-family data?

## 2. Dataset
`data/processed/five_shapes/`.

## 3. Model configuration
Baseline (latent 8, GRU 64; 24,602 parameters), medium (16/128; 67,250), large (32/128; 74,466). `configs/capacity_scaling.yaml`.

## 4. Training procedure
`bash scripts/run_capacity_scaling.sh` (seed 42, single training seed; the data stream was not controlled in this study).

## 5. Evaluation methodology
`python scripts/32_evaluate_capacity_scaling.py` (test split, shared observations).

## 6. Saved results
`results/capacity_scaling/` including `CAPACITY_SCALING_REPORT.md` (regenerated from the JSON after an earlier draft contained unsupported paired statistics; see `PROJECT_COMPLETION_AUDIT.md`).

## 7. How to reproduce the evaluation
`python scripts/32_evaluate_capacity_scaling.py`.

## 8. Limitations
Single seed; data stream not controlled; do not read this as "larger models always forecast worse".

Back to the [experiment index](../README.md). Full numbers and sources: [`docs/RESULTS.md`](../../docs/RESULTS.md).
