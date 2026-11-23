# Experiment 8: GeoLife transfer (synthetic to real GPS)

## 1. Objective
Can a model trained on synthetic geometry be used on real GPS trajectories, with or without fine-tuning?

## 2. Dataset
GeoLife 1.3 (Microsoft Research; **not distributed here**, see `data/geolife/raw/SOURCE.md`). 200-record segments, held-out users.

## 3. Model configuration
Frozen `three_shapes_unified`; GeoLife-trained from scratch; fine-tuned from the three-family weights. Linear, cubic and PCHIP interpolation as baselines. `configs/geolife.yaml`.

## 4. Training procedure
`scripts/22_geolife_training_data.py`, `scripts/23_geolife_trained_models.py` (one training run each).

## 5. Evaluation methodology
`scripts/21_evaluate_geolife_transfer.py`, `scripts/23_geolife_trained_models.py evaluate`; user-cluster bootstrap.

## 6. Saved results
Median error with 30 observations (m): linear interpolation 7.10, frozen synthetic 126.4, from scratch 60.6, fine-tuned 47.1. The synthetic model does not transfer, and both trained models remain far worse than interpolation. Aggregate results and tables only: `results/geolife/`, `reports/geolife_real_world_interpolation_report.md`.

## 7. How to reproduce the evaluation
Requires you to download GeoLife yourself and place it as described in the README. The trained GeoLife models are **not published** (licence question); they are kept locally.

## 8. Limitations
One dataset, one run per model, no raw or per-segment data published. The GeoLife report sections 9 and 11 (paired analysis) have not been written into the report.

Back to the [experiment index](../README.md). Full numbers and sources: [`docs/RESULTS.md`](../../docs/RESULTS.md).
