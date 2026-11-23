# Capacity scaling on the five-family dataset: first experiment (one seed)

> **Correction notice (2026-11-21).** An earlier version of this file contained (a) paired statistics in its section 6
> (geometric-mean ratios, "win rates", Wilcoxon p-values) that were **not produced by the evaluation script**, (b) a statement
> that 44 checkpoint files were verified (that figure belongs to the GeoLife check; this evaluation verified 4 checkpoints), and
> (c) mechanistic claims ("overfitting the context window", "posterior collapse", "hypothesis refuted") that no measurement supports.
> All three have been removed. Every number below comes from `capacity_scaling_results.json` or the tables written by
> `scripts/32_evaluate_capacity_scaling.py`. The controlled follow-up is in `results/ablation_2x2/`.

## 1. Question
Does making the encoder bigger (latent dimension and GRU width) change reconstruction and forecasting accuracy of the Latent Neural ODE on the five-family dataset?

## 2. Design (what was and was not controlled)
Same data (`data/processed/five_shapes`, 3,500/750/750), same `configs/model_d.yaml`, 300 epochs, batch 32, lr 3e-3 cosine, KL weight 1, sigma 0.05, seed 42, 110 optimiser steps per epoch (33,000 steps). Only `latent_dim` and `encoder_hidden` changed. The ODE field and decoder keep 64 hidden units x 2 layers, so parameters grow mostly in the encoder.
**Not controlled:** `data_seed` was unset, so batch order and training observation subsets came from the global random stream, which model initialisation consumes differently for each architecture. Latent dimension and GRU width were changed together. One training seed.

| Model | latent | GRU | parameters | best epoch | best val. -ELBO | training min |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8 | 64 | 24,602 | 297 | 48.50 | 77.9 |
| Medium | 16 | 128 | 67,250 | 294 | -31.47 | 80.0 |
| Large | 32 | 128 | 74,466 | 296 | -19.08 | 84.1 |

## 3. Test results (750 test trajectories, identical observations for all models)
Mean of per-trajectory RMSE (data units). Pooled RMSE is stored separately in the JSON and is a different aggregate.

| Family | Recon: baseline | medium | large | Forecast: baseline | medium | large |
|---|---:|---:|---:|---:|---:|---:|
| circle | 0.0701 | 0.0348 | 0.0386 | 0.2594 | 0.3738 | 0.3699 |
| spiral | 0.1131 | 0.0417 | 0.0454 | 0.4270 | 0.5027 | 0.5405 |
| ellipse | 0.0973 | 0.0386 | 0.0448 | 0.2609 | 0.3644 | 0.3845 |
| figure-eight | 0.1086 | 0.0548 | 0.0572 | 0.2992 | 0.4388 | 0.4851 |
| sinusoid | 0.0647 | 0.0294 | 0.0303 | 0.3861 | 0.7797 | 1.2074 |
| overall | 0.0908 | 0.0399 | 0.0432 | 0.3265 | 0.4919 | 0.5975 |

Reference: hold-last forecast RMSE on sinusoid is 0.6248, so the large model is worse than hold-last on that family (1.2074) and the medium model is too (0.7797).

## 4. Paired statistics (from the JSON; Holm correction over 30 capacity-pair tests)
- **Reconstruction, medium minus baseline:** negative on all five families (mean differences -0.035 to -0.072), medium better on 95-100% of trajectories; all Holm p < 1e-22.
- **Forecast, medium minus baseline:** positive on all five families (circle +0.114, spiral +0.076, ellipse +0.104, figure-eight +0.140, sinusoid +0.394). Medium is better on only 20%, 47%, 21%, 20%, 2% of trajectories. Spiral is **not significant after Holm correction** (p = 0.20); the other four are (p <= 3.6e-13).
- **Forecast, large minus baseline:** positive on all five (+0.111, +0.114, +0.124, +0.186, +0.821), better on 34%, 37%, 28%, 21%, 0% of trajectories.
- **Large minus medium:** reconstruction differences are tiny (+0.001 to +0.006; significant for circle, spiral, ellipse). Forecast differences are not significant for circle, spiral, ellipse (Holm p >= 0.67) or figure-eight (p = 0.056); sinusoid is clearly worse for large (+0.428, 0% improved).
- The large increase on sinusoid dominates the "overall" forecast figures.

## 5. What is established and what is not
- **Observed (one seed, this protocol):** reconstruction of withheld context points improves a lot with the larger encoders; forecasting RMSE is higher for the larger encoders on every family, with sinusoid the extreme case.
- **Not established:** the cause. The data-stream difference above, run-to-run variation, and the latent-vs-GRU confound have not been separated. No claim is made that more capacity cannot help, and nothing here was tuned or selected on the test set.
- Section-level mechanisms (e.g. "overfitting") are hypotheses only; see `results/ablation_2x2/` for the training-history audit and the controlled 2x2 design.
