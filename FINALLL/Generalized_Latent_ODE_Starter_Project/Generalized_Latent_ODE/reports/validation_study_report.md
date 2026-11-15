# Validation study of the improved Latent ODE (Model D): technical report

Date: 2026-11-15. Written for: the research team deciding whether the Model D architecture is a sound
baseline for multi-family trajectory modelling.

All numbers come from runs in this repository. Inputs: `results/validation/*/` (JSON, figures),
`checkpoints/validation/` (checkpoints plus per-epoch histories) and `logs/validation/`. Earlier results in
`results/improved_two_shapes/` are unchanged apart from figure layout; their metrics files are byte-identical.

## 1. Summary

| Question | Answer (evidence below) |
|---|---|
| Is Model D reproducible? | Reconstruction is reproducible: test RMSE **0.060 ± 0.007** over 5 seeds. Forecasting varies more: **0.219 ± 0.050**. The reported seed-42 run (0.192) is one of the better seeds. |
| Is the checkpoint-selection criterion sound? | Yes. It is −ELBO, verified by an independent unit test, and lower is better. Across seeds, validation −ELBO ranks runs almost exactly as test forecast error does. |
| Why are some forecasts poor? | Error grows steadily after t = 7, driven by systematic over-estimation of angular speed, under-estimated spiral growth magnitude and radius drift. Large or expanding spirals and large or slow circles are hardest. |
| Is the model robust to sparsity and noise? | **Noise: yes. Sparsity: no.** The encoder works only near the 30 observations it was trained with; even 50 observations gives reconstruction RMSE 0.58. |
| Does one shared model cost accuracy? | **Yes, for forecasting.** Circle forecast RMSE is 0.161 ± 0.037 unified vs 0.087 ± 0.006 specialist; spiral 0.284 ± 0.067 vs 0.198 ± 0.009. Reconstruction is close. |
| Does one model learn three families? | Yes, qualitatively: all rotation directions are correct and ellipse shape is recovered. Spiral accuracy drops relative to the two-shape model. |
| Does it generalise to unseen ellipses? | Partly. Rotation and scale transfer and forecasts beat the baselines, but ellipses are rounded toward circles (axis-ratio correlation 0.18). |

## 2. Common protocol

- Data: `data/processed/two_shapes` (1,400 / 300 / 300) and `three_shapes` (2,100 / 450 / 450). The circle
  and spiral trajectories in the two-shape and three-shape test splits are identical (verified by ID and value).
- Inputs: 30 noisy observations (σ = 0.05) at t ≤ 7, a random irregular subset; the first point is always t = 0.
  Targets are the clean trajectory on 200 points over [0, 10]. The forecast window is t > 7.
- Model D configuration: `configs/model_d.yaml` (latent 8, GRU 64, dynamics and decoder MLPs 64×2, RK4 step 0.1,
  Adam 3e-3 with cosine schedule, 300 epochs, batch 32, KL weight 1). The checkpoint is the best validation −ELBO.
- Evaluation: posterior-mean z0 and the same observation subsets for every model compared (seed 123).
  Metrics are in original coordinate units and averaged per trajectory.
- Across-seed intervals: two-sided Student-t 95% CI with n−1 degrees of freedom over per-run means.
- Paired comparisons: the per-trajectory difference on identical trajectories and observations, with a bootstrap 95% CI
  (5,000 resamples) and a Wilcoxon signed-rank p-value.

## 3. Phase 1: Reproducibility (5 seeds: 42, 123, 2024, 2025, 2026)

Source: `results/validation/seed_study/` (`seed_table.md`, `seed_study.json`, `seed_variability.png`).

| Family | Metric | Mean ± SD | 95% CI | Per-seed (42, 123, 2024, 2025, 2026) |
|---|---|---|---|---|
| overall | recon RMSE | 0.0599 ± 0.0069 | [0.0514, 0.0684] | 0.056, 0.062, 0.055, 0.056, 0.071 |
| overall | forecast RMSE | 0.2194 ± 0.0502 | [0.1571, 0.2817] | 0.192, 0.283, 0.169, 0.190, 0.262 |
| overall | forecast ADE | 0.2898 ± 0.0642 | [0.2101, 0.3695] | |
| overall | forecast FDE | 0.4446 ± 0.1006 | [0.3198, 0.5695] | |
| overall | Chamfer | 0.0344 ± 0.0127 | [0.0187, 0.0501] | |
| circle | recon / forecast RMSE | 0.0523 ± 0.0053 / 0.1571 ± 0.0374 | [0.046, 0.059] / [0.111, 0.204] | |
| spiral | recon / forecast RMSE | 0.0676 ± 0.0086 / 0.2816 ± 0.0672 | [0.057, 0.078] / [0.198, 0.365] | |

- Every seed predicts the rotation direction correctly 100% of the time. The spiral expansion/contraction sign is correct for
  87–100% of trajectories depending on the seed.
- Best epochs were 288–297 of 300 in every run, so training had **not** fully converged.
- **Validation ELBO.** `batch_objective` returns −ELBO = −E[log p(x|z)] + KL(q‖N(0,I)), averaged over
  trajectories. `tests/test_objective.py` recomputes it with `torch.distributions` and agrees to 1e-5.
  Selection takes the minimum, which is correct. Negative values are legitimate, because a Gaussian with
  σ = 0.056 (normalised) has density above 1 near its mean. Validation plugs in z0 = μ, a deterministic
  surrogate, not a Monte-Carlo ELBO. Across the five seeds, validation −ELBO (−25.9, −23.0, −22.3, −2.6, +4.9) orders
  the runs almost exactly as test forecast RMSE does (0.169, 0.192, 0.190, 0.283, 0.262).

## 4. Phase 2: Forecasting analysis

Source: `results/validation/forecast_analysis/`. Per-trajectory analyses use seed 42; sweeps use all 5 seeds.

**Error vs horizon** (mean pointwise Euclidean error, seed 42):

| t | 3.5 | 7.0 | 8.0 | 9.0 | 10.0 |
|---|---|---|---|---|---|
| circle | 0.058 | 0.102 | 0.155 | 0.213 | 0.295 |
| spiral | 0.059 | 0.151 | 0.255 | 0.378 | 0.492 |

**Geometric errors** (forecast window, seed 42):
- Angular velocity: |ω̂| exceeds |ω| by +0.038 (circles) and +0.049 (spirals) on average, so forecasts run
  ahead in phase. Over the full trajectory the error is only 0.019–0.030, so the speed-up is concentrated after t = 7.
- Circle radius: mean |drift| at t = 10 is 15%, with a net outward bias (+8%; two thirds drift outward).
- Spiral growth rate: the sign is correct for 100% of trajectories, but the magnitude is under-estimated by 0.026 on average
  (≈30% of a typical |g|), which is worse for contracting spirals (0.035) than expanding ones (0.022).

**What makes a trajectory hard** (Spearman correlation with forecast RMSE, seed 42):
- Spirals: final radius r0·e^{10g} (ρ = 0.75, p ≈ 6e-28), growth rate (ρ = 0.66; expanding is harder), slow rotation
  (|ω|, ρ = −0.32).
- Circles: radius (ρ = 0.37) and slow rotation (ρ = −0.36). The slowest circles complete less than one revolution before t = 7.
- Observation pattern (largest gap, last observation time, density near t = 7) is not significant (p > 0.05).
- The same trajectories are hard for every seed: pairwise Spearman correlation of per-trajectory errors is 0.64, and the worst-10%
  sets overlap by 59% against 10% expected by chance.

**Noise** (fresh noise, 5 seeds): forecast RMSE for circles is 0.158 at σ = 0, 0.162 at 0.05 and 0.214 at 0.2.
Spirals: 0.274, 0.277, 0.312. Removing noise entirely barely helps, so the error is **model-limited, not noise-limited**.

**Sparsity**: see §8. Error is V-shaped with its minimum at the trained count of 30 observations.

Figures: `error_vs_horizon.png`, `geometry_errors.png`, `error_vs_parameters.png`,
`best_median_worst.png`, `sparsity_and_noise.png`.

## 5. Phase 3: Shared vs separate models

Source: `results/validation/shared_vs_separate/`. The architecture and configuration are identical. Specialists train on
700 trajectories for 600 epochs, matching the unified model's 1,400 for 300 epochs in optimiser steps and samples seen.
Each family's 150 test trajectories are evaluated with identical observations for both models.

| Family | Metric | Unified (5 seeds) | Specialist (3 seeds) | Welch p | Paired seed 42: diff [95% CI] |
|---|---|---|---|---|---|
| circle | recon RMSE | 0.0532 ± 0.0059 | 0.0457 ± 0.0008 | 0.047 | +0.0055 [0.0017, 0.0092] |
| circle | forecast RMSE | 0.1606 ± 0.0369 | 0.0868 ± 0.0061 | 0.010 | +0.064 [0.050, 0.079] |
| circle | forecast FDE | 0.3161 ± 0.0693 | 0.1545 ± 0.0111 | 0.006 | +0.158 [0.127, 0.191] |
| spiral | recon RMSE | 0.0680 ± 0.0102 | 0.0632 ± 0.0008 | 0.36 | −0.0018 [−0.0057, 0.0019] |
| spiral | forecast RMSE | 0.2841 ± 0.0671 | 0.1983 ± 0.0086 | 0.045 | +0.052 [0.032, 0.073] |
| spiral | forecast FDE | 0.5774 ± 0.1402 | 0.3943 ± 0.0261 | 0.041 | +0.113 [0.069, 0.159] |

- **The shared model costs forecasting accuracy.** Forecast RMSE is about 85% higher for circles and 43% higher for spirals at
  the seed level. Reconstruction cost is small (circles) or not detectable (spirals).
- **Specialists are far more stable across seeds.** Forecast SD is 0.006–0.009 for specialists and 0.037–0.067 for the
  unified model, which suggests the unified objective is harder to optimise, not only less accurate.
- Caveats: the specialists' best epochs were 581–596 of 600, so they were also still improving. Specialists have three seeds.

## 6. Phase 4: Three-shape shared model (one seed, 42)

Source: `results/validation/three_shapes/`. Dataset checks: ellipses satisfy their equation to 6e-7 and their phase advances as ω·t
to 2e-6; there are no duplicate parameters across splits; circle and spiral entries are identical to the two-shape dataset.

| Family | Recon RMSE | Forecast RMSE | Forecast FDE | Constant-position forecast RMSE |
|---|---|---|---|---|
| circle | 0.058 | 0.137 | 0.253 | 1.135 |
| spiral | 0.094 | 0.298 | 0.540 | 1.285 |
| ellipse | 0.093 | 0.198 | 0.321 | 1.118 |

- Rotation direction is correct for 100% of trajectories in all three families. Spiral growth sign accuracy is 92.7%. The ratio of
  predicted to true temporal variance is 0.92–1.03, so there is no collapse.
- Ellipse geometry: predicted minor/major axis ratio 0.67 vs 0.59 true (somewhat too round), correlation with truth
  0.83, median major-axis orientation error 4.9°.
- **Against the two-shape models on identical circle and spiral test trajectories:** circles are comparable (forecast
  0.136 vs the two-shape 5-seed range 0.118–0.217; reconstruction 0.059, just above the two-shape CI of 0.046–0.059).
  **Spiral reconstruction is worse**: 0.096, outside the two-shape range of 0.061–0.082. Spiral forecasting (0.298) is within the
  two-shape range (0.220–0.360). This is a single three-shape seed, so the comparison is suggestive.

## 7. Phase 5: Out-of-distribution ellipses (evaluation only)

The two-shape models were never trained on ellipses. Their inputs are normalised with the **two-shape**
training statistics, enforced by `glo.evaluation.collect` and tested. No model was changed or selected using this experiment.

| On the 150 test ellipses | Recon RMSE | Forecast RMSE | Forecast FDE | Axis-ratio corr. | Orientation err. (median) |
|---|---|---|---|---|---|
| Two-shape models, OOD (5 seeds) | 0.210 [0.205, 0.215] | 0.321 [0.287, 0.355] | 0.506 | 0.18 (seed 42) | 34° |
| Three-shape model, in-distribution | 0.089 | 0.185 | 0.302 | 0.86 | 4.6° |
| Constant position (observations only) | 0.063* | 1.121 | 1.912 | – | – |

\*Interpolation of the noisy observations.

- **What transfers:** rotation direction (100%), rough scale and the orbit's centre. OOD forecasts are much better than the
  observation-only baselines.
- **What does not transfer:** shape. The two-shape model rounds ellipses toward circles (predicted axis ratio 0.92 vs 0.59) and
  its reconstruction is *worse than interpolating the observations* (0.210 vs 0.063).
- So the evidence for out-of-family generalisation is weak. The model reuses its learned rotation dynamics and does not represent a new shape.
- Note on Chamfer: it ignores timing, and when the context already covers the whole orbit even a constant-position
  forecast scores well (ellipse Chamfer 0.007). Chamfer should not be used alone.

## 8. Observation-count robustness and an ablation

Source: `results/validation/forecast_analysis/` (sweep) and `results/validation/nobs_ablation/`.
The problem was found on the **validation** split. On the test split, recon / forecast RMSE for Model D:

| n_obs | 10 | 20 | 28 | **30** | 32 | 40 | 50 | 100 |
|---|---|---|---|---|---|---|---|---|
| Model D (trained with 30) | 0.95 / 1.47 | 0.48 / 1.02 | 0.11 / 0.31 | **0.058 / 0.197** | 0.10 / 0.22 | 0.36 / 0.70 | 0.58 / 1.09 | 0.77 / 1.03 |
| Ablation: trained with random 10–50 | 0.16 / 0.49 | 0.069 / 0.40 | 0.059 / 0.39 | 0.058 / 0.39 | 0.057 / 0.40 | 0.056 / 0.40 | 0.056 / 0.41 | 0.075 / 0.47 |

- Model D's GRU encoder has learned the sequence length. More observations make it worse.
- Training with a variable number of observations fixes reconstruction robustness for 10–100 observations, but **doubles forecast error at n = 30**
  (0.39 vs 0.20), with a single seed. Robustness and forecast accuracy trade off in this configuration. The ablation is
  not part of Model D and was not used for any other result.

## 9. Limitations

1. Forecasting accuracy degrades steadily over the 3-unit horizon. Systematic phase lead and radius drift
   dominate, and noise is not the cause.
2. The model only works with the observation count it was trained with (§8).
3. Seed sensitivity: unified forecast SD is about 0.05, so single-seed comparisons of unified models are unreliable.
4. Every run's best epoch is within the last 3% of training. Longer training could change the absolute numbers.
5. Single seed for the three-shape model and the variable-n_obs ablation; three seeds for specialists.
6. Model D's hyperparameters were selected on two-shape validation data and reused unchanged everywhere, which may favour the two-shape setting.

## 10. Reproduction

```
bash scripts/run_validation_experiments.sh          # training (commands for every run are listed in the file)
python scripts/07_seed_study.py   --checkpoints seed42=checkpoints/improved_two_shapes/D_candidate_lr3e-3.pt seed123=checkpoints/validation/two_shapes_seed123.pt ...
python scripts/08_forecast_analysis.py --checkpoints <5 two-shape checkpoints>
python scripts/09_shared_vs_separate.py --unified <5> --circle <3> --spiral <3>
python scripts/10_three_shapes_and_ood.py --three checkpoints/validation/three_shapes_seed42.pt --two <5>
python scripts/11_nobs_ablation.py
python -m pytest -q tests                             # 16 tests
```
