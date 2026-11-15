# Improved two-shape Latent ODE: results (2026-11-15)

Written for: the project owner, who needs to know whether the reconstruction failure is fixed and what is still open.

## Result in one paragraph

The original checkpoint (epoch 3 of a run that never finished) reconstructed the observations
poorly and forecast by predicting a near-constant trajectory near the origin. The corrected
model is a single shared Latent ODE for circles and spirals with a larger architecture
(latent 8, hidden 64, two hidden layers in dynamics and decoder, RK4 solver). On the held-out
test split of 300 trajectories it reduces reconstruction RMSE from 0.756 to 0.056 and forecast
RMSE from 0.843 to 0.192. It is also clearly better than constant-position and
constant-velocity baselines on forecasting. Remaining limitations are listed at the end.

## Selection and protocol

- Candidate runs, all trained on the full 1,400-trajectory training split with 30 noisy observations
  at t ≤ 7 and a clean full-trajectory evaluation target:

| Run | Architecture | Other changes | Best epoch | Val context RMSE | Val forecast RMSE |
|---|---|---|---|---|---|
| A | original (latent 4, hidden 25/20/20, 1 layer) | lr 1e-3, cosine, 300 ep | 257 | 0.335 | 0.815 |
| B | latent 8, hidden 64, 2 layers | lr 1e-3, cosine | 294 | 0.125 | 0.475 |
| C | as B | + KL annealing over 30 epochs | 294 | 0.113 | 0.364 |
| **D** | as B | lr 3e-3, cosine | **296** | **0.059** | **0.262** |

- **D was chosen by validation ELBO (−23.04)**, the lowest of the four. The test split was not
  used for selection. The negative ELBO is expected: the Gaussian log-likelihood includes
  constant log-σ terms, so it can be negative.
- Final evaluation: `scripts/06_full_evaluation.py` on the 300-trajectory test split, using the
  same 30 observation subsets for every model and baseline (seed 123).
- Single seed (42) for training. No repeats, so the difference between B, C and D is not established.

## Test-split results (original coordinate units; mean over trajectories)

| Model | Family | Recon RMSE | Forecast RMSE | Forecast MAE | Forecast ADE | Forecast FDE | Chamfer |
|---|---|---|---|---|---|---|---|
| **Improved (D)** | overall | **0.056** | **0.192** | **0.163** | **0.255** | **0.393** | **0.027** |
| Improved (D) | circle | 0.051 | 0.142 | 0.121 | 0.189 | 0.295 | 0.010 |
| Improved (D) | spiral | 0.061 | 0.242 | 0.204 | 0.320 | 0.492 | 0.043 |
| Original checkpoint | overall | 0.756 | 0.843 | 0.748 | 1.177 | 1.262 | 1.806 |
| Original checkpoint | circle | 0.755 | 0.775 | 0.689 | 1.082 | 1.106 | 1.706 |
| Original checkpoint | spiral | 0.756 | 0.911 | 0.808 | 1.272 | 1.418 | 1.905 |
| Constant position | overall | 0.066 | 1.209 | 1.014 | 1.593 | 2.137 | 0.075 |
| Constant velocity | overall | 0.066 | 2.366 | 1.811 | 2.837 | 5.909 | 2.429 |

- Reconstruction baselines use linear interpolation of the noisy observations, so the 0.066 for them
  is the error of an interpolated noisy path, not a clean-data fit. The model reconstructs
  slightly better than this (0.056) on the unobserved context points.
- Sampled posterior (10 draws per trajectory): recon 0.059, forecast 0.195. The posterior mean and sampled predictions
  agree, so posterior variance is not hurting predictions.
- Solver check: RK4 (step 0.1) vs dopri5 at rtol 1e-6 differs by at most 0.0046 and on average by 0.0009 in
  original units, so the RK4 training solver is not distorting the learned trajectories.
- Median per-trajectory forecast RMSE is 0.12 for circles and 0.21 for spirals. The worst is 0.46 for circles
  and 0.82 for spirals, so some forecasts are poor.

## Geometry checks (against the known generating parameters)

| Check | Truth | Improved model |
|---|---|---|
| Rotation direction correct (circle / spiral) | 100% / 100% | 100% / 100% |
| Mean abs. angular-velocity error (rad/unit t) | 0 | 0.019 (circle) / 0.030 (spiral) |
| Circle radius relative error (context / future) | 0 | 5.7% / 10.6% |
| Circle radius coefficient of variation | 0 | 0.063 |
| Spiral expansion vs contraction correct | 100% | 100% |
| Spiral growth-rate correlation with truth | 1.0 | 0.94 |

- The model captures rotation direction, angular speed, the spiral's expansion or contraction, and the
  circle's near-constant radius.
- Forecast error grows with time. Radius error doubles from context to future for both families.

## Variance and "average-trajectory" check

- Temporal variance of predicted trajectories relative to ground truth: circle 1.04, spiral 0.93
  (original checkpoint: 0.015 and 0.010). The improved model is no longer collapsing toward the mean.
- RMSE of predicting the family's mean test trajectory: circle 0.80, spiral 0.92. The improved model
  is far better than that, so it is not predicting an average shape.

## Training-curve observations

- Run A (original architecture, 300 epochs) plateaus at context RMSE ≈ 0.33. The small network
  cannot fit the data, and training longer does not change that.
- The larger model keeps improving with a higher learning rate. KL stays at about 33–35 nats per
  trajectory and does not collapse toward zero, so the encoder is still using the latent code.
- KL annealing (C) gave a small forecast gain over B (0.364 vs 0.475 validation forecast RMSE). This
  is a single seed and should be treated as suggestive.

## Figures and files

Directory: `results/improved_two_shapes/`
- `circle_reconstruction.png`, `spiral_reconstruction.png`: six test trajectories each, in test-split order
- `circle_coordinates.png`, `spiral_coordinates.png`: x(t) and y(t), with the forecast boundary at t = 7
- `before_after_comparison.png`: original vs improved, same trajectories, same observations, same axis limits.
  The suptitle overlaps the subplot titles. This is cosmetic and not yet fixed.
- `median_and_worst_cases.png`: median and worst forecast per family (honest counterpart to the representative figures)
- `loss_curves.png`: objective, NLL, KL and validation RMSE against epoch
- `evaluation.json`, `baseline_comparison.json`, `per_trajectory_metrics.csv`

Checkpoints: `checkpoints/improved_two_shapes/` (A, B, C, D, and `best_model.pt` = D).
Histories: `*_history.csv` next to each checkpoint. Logs: `logs/`.
The original checkpoint is unchanged; a backup is in `backups/original_2026-11-15/`.

## Limitations

1. **Forecasting is the weakest part.** Forecast RMSE is 0.19 against 0.056 reconstruction, and worst-case
   spiral forecasts reach 0.82. Some circle forecasts drift inward or outward.
2. **One training seed.** No confidence intervals across seeds. The B/C/D differences are not established.
3. **Model size was chosen by experiment, not by a search.** Latent 8 and hidden 64 are one candidate
   that works. Other settings were not explored.
4. **Solver:** the model was trained with fixed-step RK4 because adaptive dopri5 was too slow on the
   shared GPU for this model. Cross-check shows the difference is negligible (see above).
5. **Unit tests** pass (10/10) including analytic ODE and gradient-through-solver tests. The circle/spiral
   test split was used once, for the final evaluation.
6. **Not done:** the dynamics-conditioned architecture (Phase 8), the ellipse extension (Phase 9), and the
   original paper's exact spiral generator (Phase 7). These are outside this task and were deliberately not started.
