# Experiment 1: three-family unified model (circle, spiral, ellipse)

## 1. Objective
Train one shared Latent Neural ODE on three trajectory families without a family label and measure reconstruction (t <= 7) and forecasting (7 < t <= 10).

## 2. Dataset
`data/processed/three_shapes/` (1,000 trajectories per family; 200 time points on [0, 10]; 30 noisy irregular observations at t <= 7). Split 70/15/15 per family (450 test trajectories).

## 3. Model configuration
Model D: latent 8, GRU 64, tanh MLPs 64x2, RK4 step 0.1; 24,602 parameters. `configs/model_d.yaml`.

## 4. Training procedure
Adam 3e-3 (cosine to 1e-5), batch 32, 300 epochs, -ELBO (fixed sigma 0.05, beta 1), checkpoint = lowest validation -ELBO. Seed 42 (single training seed).

## 5. Evaluation methodology
Test split, 30 observations, observation seed 123, posterior-mean latent state; per-trajectory RMSE averaged.

## 6. Saved results
Overall test: reconstruction RMSE 0.0817, forecast RMSE 0.2110 (original units); per-family values in `saved_models/three_shapes_unified/metadata.json` and `results/validation/three_shapes/`.

## 7. How to reproduce the evaluation
`python scripts/03_evaluate_model.py --help`; packaged model: `python scripts/predict_trajectory.py --model three_shapes_unified --input input_trajectories/example_circle.csv`.

## 8. Limitations
Single seed. Forecasting error grows with horizon. Unified models forecast worse than specialists (Experiment 3). Only works with about 30 observations.

Back to the [experiment index](../README.md). Full numbers and sources: [`docs/RESULTS.md`](../../docs/RESULTS.md).
