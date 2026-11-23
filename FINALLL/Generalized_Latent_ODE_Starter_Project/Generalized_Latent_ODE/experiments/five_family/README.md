# Experiment 2: five-family unified model (circle, spiral, ellipse, figure-eight, sinusoid)

## 1. Objective
Extend the shared model to five families and compare it with the three-family model.

## 2. Dataset
`data/processed/five_shapes/` (1,000 per family; 3,500 / 750 / 750 train/val/test). Circle, spiral, ellipse are identical to the three-family data. Generator: `scripts/30_generate_five_shapes.py`, `src/glo/five_shapes.py`.

## 3. Model configuration
Model D (latent 8, GRU 64; 24,602 parameters), same recipe as Experiment 1; 33,000 optimiser steps (19,800 for the three-family run).

## 4. Training procedure
Seed 42 (single training seed); selected epoch 297 of 300.

## 5. Evaluation methodology
`python scripts/31_evaluate_five_shapes.py` (test split, shared observations for all models, baselines: hold-last, constant velocity).

## 6. Saved results
Overall test: reconstruction 0.0908, forecast 0.3265 (hold-last forecast 1.0238; linear-interpolation reconstruction 0.0672, so the learned reconstruction is not better than interpolation). Report: `reports/five_shapes_report.md`; data: `results/five_shapes/`.

## 7. How to reproduce the evaluation
`python scripts/31_evaluate_five_shapes.py`; `python scripts/predict_trajectory.py --model five_shapes_unified --input input_trajectories/example_figure_eight.csv`.

## 8. Limitations
Single seed. On the three original families the five-family model forecasts worse than the three-family model (same data stream differences confound the comparison).

Back to the [experiment index](../README.md). Full numbers and sources: [`docs/RESULTS.md`](../../docs/RESULTS.md).
