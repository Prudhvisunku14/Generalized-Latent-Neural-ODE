# Five trajectory families: dataset, zero-shot test of the three-shape model, and a five-shape model

Date: 2026-11-21. Written for: the project owner and supervisor. All numbers come from results already saved in this repository (listed in section 10); nothing was trained or re-run for this report.

## 1. Summary

We added two trajectory families, **figure-eight** and **sinusoid**, to circle, spiral and ellipse (1,000 trajectories each, same time grid, noise and observation protocol), then asked two questions:

1. **Does the frozen three-shape model handle the two families it never saw (zero-shot)?** No. On figure-eight and sinusoid its forecasts (1.1174 and 0.9734) are worse than simply holding the last observation (0.9610 and 0.6248), and its reconstruction (0.7321 and 0.3441) is far worse than linear interpolation (0.0917 and 0.0452).
2. **Can one model with the same architecture and settings be trained on all five families?** Yes. It forecasts all five far better than the naive forecast baselines (0.2594 to 0.4270 against hold-last 0.6248 to 1.2763). But, compared with the three-shape model, it is **worse on the three original families** (forecast +85% on circle, +47% on spiral, +38% on ellipse), and its reconstruction of withheld points is no better than plain linear interpolation on any family (essentially tied on circle, +0.2%, and worse by 18% to 73% on the other four).

What this does **not** show: the five-shape model was trained on all five families, so its results on them are **not zero-shot generalisation**. All results are from one training run (seed 42). No claim of novelty, of improved forecasting, or of generalisation to arbitrary trajectories is made. The project is an adapted re-implementation and experimental extension of Latent Neural ODEs, not a replication of Chen et al. (2018).

## 2. The five families

Time-ordered 2-D trajectories on 200 equally spaced times in [0, 10]; Gaussian observation noise SD 0.05 (clean trajectories are kept as targets). The first three families are the project's existing generators (unchanged); the two new ones are:

| Family | Equations (theta = omega t + phase) | Parameter ranges (uniform) |
|---|---|---|
| figure-eight | x = cx + A sin(theta), y = cy + B sin(2 theta) | A in [0.85, 1.5], B in [0.45, 1.0], abs(omega) in [0.75, 1.45], phase in [-pi, pi], centre in [-0.35, 0.35], direction = sign of omega (+/-, p = 1/2) |
| sinusoid | x = x0 + v t, y = cy + A sin(omega t + phase), x0 = cx - 5v | A in [0.4, 1.0], omega in [0.75, 1.45], abs(v) in [0.2, 0.3], phase in [-pi, pi], centre in [-0.35, 0.35], direction = sign of v |

Circle, spiral and ellipse keep their original generators and parameter ranges (`src/glo/data.py`).

## 3. Dataset construction and validation

- **Counts:** 1000 per family, 5,000 in total (5,000 unique ids). Stratified split by family: train 3,500 / validation 750 / test 750 (700 / 150 / 150 per family in every split; no trajectory id appears in more than one split).
- **Original families preserved:** the circle, spiral and ellipse trajectories, their noisy observations, observation masks and split membership are array-for-array identical to `three_shapes` in all three splits (checked in `dataset_checks.json`: every field `true`).
- **Observation protocol:** 30 observed points per trajectory, all at t <= 7 (first at t = 0), randomly spaced; no observation after t = 7 reaches the encoder. Forecast window: 7 < t <= 10. No shape label is given to any model.
- **Shape checks:** figure-eight: all 1000 cover a full eight with alternating centre crossings (largest vertical offset at the centre crossing 3.7e-05 of B; median y-to-x zero-crossing ratio 2.0). Sinusoid: all 1000 have monotone x (maximum deviation from linear 0.0) and the y half-range matches the amplitude (ratio 0.9994 to 1.0000). Failures recorded: 0.
- **Scale:** median RMS radius about the own centroid is 0.89 to 1.02 across the five families, so no family is trivially different in size. Normalisation constants are computed from the five-shape training split only (mean [0.0005, 0.0043], scale 0.7944).
- The splits and leakage checks were independently re-verified in the project audit (`PROJECT_COMPLETION_AUDIT.md`, `audit/verify_1_datasets.py`).

## 4. Models and evaluation protocol

| Model | Trained on | Parameters | Epochs / best epoch | Best validation -ELBO | Optimiser steps | Training time |
|---|---|---:|---|---:|---:|---:|
| Three-shape (frozen, packaged) | circle, spiral, ellipse (2,100 trajectories) | 24,602 | 300 / 298 | 20.12 (three-shape validation data) | 19,800 | 60 min |
| Five-shape | all five (3,500 trajectories) | 24,602 | 300 / 297 | 48.50 (five-shape validation data) | 33,000 | 78 min |

Both use the same architecture and training configuration (`configs/model_d.yaml`: latent 8, GRU 64, tanh MLPs 64x2, rk4 step 0.1, Adam 3e-3 with cosine schedule, batch 32, 300 epochs, KL weight 1, fixed sigma 0.05, seed 42). The two validation -ELBO values are on different validation sets and must not be compared.

**Evaluation:** five-shape test split (750 trajectories, 150 per family); every model and baseline receives the **identical** 30 observations (same times and noise, asserted); posterior-mean initial state; each model is normalised with its own training-set constants; metrics are per-trajectory RMSE in data units, then averaged. *Reconstruction RMSE* uses the withheld context points (t <= 7); *forecast RMSE* uses all points with 7 < t <= 10. Baselines: linear interpolation (reconstruction), hold-last observation and constant velocity (forecast). Paired comparisons use the per-trajectory difference on identical trajectories with a bootstrap 95% interval and a Wilcoxon test; Holm correction over the 10 tests (five families x two metrics). Test-set intervals describe test-trajectory sampling only, not training-seed variability.

## 5. Zero-shot test: the frozen three-shape model on figure-eight and sinusoid

The three-shape model (checksum verified; its stored three-shape results were reproduced exactly, maximum deviation 0.0) was applied unchanged. Mean (median) of per-trajectory RMSE:

| Family | Status | Recon. RMSE | Linear interp. | Forecast RMSE | Hold-last | Const. velocity |
|---|---|---|---:|---|---:|---:|
| circle | seen in training | 0.0622 (0.0553) | 0.0699 | 0.1401 (0.1217) | 1.1370 | 2.3274 |
| spiral | seen in training | 0.0933 (0.0894) | 0.0653 | 0.2908 (0.2549) | 1.2763 | 2.3877 |
| ellipse | seen in training | 0.0888 (0.0806) | 0.0637 | 0.1885 (0.1727) | 1.1200 | 2.0143 |
| figure_eight | **unseen (zero-shot)** | 0.7321 (0.6745) | 0.0917 | 1.1174 (1.0973) | 0.9610 | 2.6176 |
| sinusoid | **unseen (zero-shot)** | 0.3441 (0.3422) | 0.0452 | 0.9734 (0.9517) | 0.6248 | 1.3898 |
| overall | all five | 0.2641 (0.1120) | 0.0672 | 0.5421 (0.3190) | 1.0238 | 2.1474 |

On the two unseen families the model is worse than the hold-last baseline for forecasting and worse than linear interpolation for reconstruction (by a large margin for reconstruction). It is better than constant-velocity extrapolation, but that baseline is itself very poor on these trajectories (forecast RMSE 2.6 and 1.4). The model does not generalise to the unseen families. (`overall` mixes seen and unseen families and should not be read as a generalisation score.) This is one frozen model from one training seed.

## 6. The five-shape model

| Family | Recon. RMSE mean (median) | Linear interp. | Forecast RMSE mean (median) | Hold-last | Const. velocity |
|---|---|---:|---|---:|---:|
| circle | 0.0701 (0.0643) | 0.0699 | 0.2594 (0.2342) | 1.1370 | 2.3274 |
| spiral | 0.1131 (0.1057) | 0.0653 | 0.4270 (0.3744) | 1.2763 | 2.3877 |
| ellipse | 0.0973 (0.0870) | 0.0637 | 0.2609 (0.2390) | 1.1200 | 2.0143 |
| figure_eight | 0.1086 (0.0970) | 0.0917 | 0.2992 (0.2589) | 0.9610 | 2.6176 |
| sinusoid | 0.0647 (0.0600) | 0.0452 | 0.3861 (0.3676) | 0.6248 | 1.3898 |
| overall | 0.0908 (0.0809) | 0.0672 | 0.3265 (0.2723) | 1.0238 | 2.1474 |

The model forecasts every family far better than the naive baselines. Its **reconstruction** of withheld context points is not better than plain linear interpolation on any family: essentially tied on circle (0.0701 vs 0.0699, +0.2%) and worse on spiral, ellipse, figure_eight and sinusoid. (The later, larger models in the capacity and ablation studies do beat linear interpolation; see section 9.)

## 7. Three-shape versus five-shape model (paired, identical observations)

Difference = five-shape minus three-shape (positive = the five-shape model has higher error). Holm-adjusted p-values.

| Family | Status for the three-shape model | Forecast: three | Forecast: five | Difference [95% CI] | Five better on | Holm p | Recon: three | Recon: five | Holm p (recon.) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| circle | trained on | 0.1401 | 0.2594 | +0.1193 [+0.0982, +0.1404] | 15% | 3e-20 | 0.0622 | 0.0701 | 0.00026 |
| spiral | trained on | 0.2908 | 0.4270 | +0.1362 [+0.0998, +0.1726] | 27% | 1.7e-11 | 0.0933 | 0.1131 | 4.3e-10 |
| ellipse | trained on | 0.1885 | 0.2609 | +0.0723 [+0.0536, +0.0911] | 21% | 1.7e-11 | 0.0888 | 0.0973 | 0.0048 |
| figure_eight | **unseen** | 1.1174 | 0.2992 | -0.8182 [-0.8940, -0.7424] | 99% | 2.3e-25 | 0.7321 | 0.1086 | 2.3e-25 |
| sinusoid | **unseen** | 0.9734 | 0.3861 | -0.5874 [-0.6212, -0.5536] | 99% | 2.3e-25 | 0.3441 | 0.0647 | 2.3e-25 |

- On the three original families the five-shape model is **worse** (higher forecast and reconstruction error; better on only 15% to 27% of trajectories for forecasting). Sharing one model across five families cost accuracy on the original three, consistent with the earlier unified-versus-specialist result for two families.
- On the two new families the five-shape model is far better than the three-shape model, which is expected because it was trained on them.
- **Caveat for this comparison:** the five-shape model also had a different training set size and more optimiser steps (section 8, item 4), so the cost on the original families cannot be attributed to the number of families alone.

## 8. Known issues and caveats

1. **Figure `error_vs_time_three_vs_five.png`:** the third curve is labelled "const. velocity" along the whole time axis, but before t = 7 it is the linear-interpolation baseline (constant velocity applies only after the last observation), and it passes through the observed timestamps, which is why its error is so low there. The numbers in the tables are not affected. The figure was left unchanged to preserve the original results.
2. **Figure `per_family_rmse_three_vs_five.png`:** the x-axis labels of figure_eight and sinusoid overlap. Cosmetic.
3. **Observation draws differ between test splits.** Observations are drawn in dataset order, so the same three-shape model evaluated on the three-shape test split (stored results) and on the five-shape test split sees different observation subsets. Forecast RMSE for circle is 0.1372 on the stored three-shape evaluation and 0.1401 here (spiral 0.2977 vs 0.2908; ellipse 0.1981 vs 0.1885). The two tables are each internally consistent but are not interchangeable; this report uses only the five-shape test-split evaluation for the three-versus-five comparison.
4. **Training budget differs.** Both models ran 300 epochs, but with batch size 32 the three-shape model took 66 optimiser steps per epoch (19,800 in total) and the five-shape model 110 (33,000). The normalisation scale also differs (five-shape 0.7944).
5. **One training seed** for each model. The only multi-seed evidence in the project is the earlier five-seed two-shape study, which shows forecast RMSE of one architecture varying by about 23% (coefficient of variation) between seeds. Differences between single runs should be read with that in mind.
6. Metrics are averages of per-trajectory RMSE; pooled RMSE is a different aggregate and is stored separately in the capacity and ablation results.

## 9. Related later experiments (separate reports)

Using the five-shape model as the baseline, two single-seed follow-ups varied encoder size: `results/capacity_scaling/CAPACITY_SCALING_REPORT.md` (latent 8/16/32, GRU 64/128; training streams not controlled) and `results/ablation_2x2/ABLATION_2X2_REPORT.md` (a controlled 2x2 of latent 8/16 by GRU 64/128). In both, larger encoders reconstructed withheld points better; the full-horizon forecast was higher for some larger variants, but the differences are of the same order as seed-to-seed variation and have not been confirmed across seeds. See those reports for the exact, qualified statements.

## 10. What can and cannot be claimed, and sources

**Can be claimed (from these results):**
- The five-family dataset is valid, leakage-free, and contains the three-shape data unchanged.
- A model trained on circle, spiral and ellipse is worse than hold-last at forecasting and worse than linear interpolation at reconstructing figure-eight and sinusoid trajectories (it is better only than constant-velocity extrapolation, a very poor baseline here); one frozen model, one seed.
- A model with the same architecture can be trained on all five families without labels and forecasts them far better than naive baselines, at a cost in accuracy on the original three families (one seed each, budgets not matched).

**Cannot be claimed:** generalisation to unseen families by the five-shape model (it saw all five); that adding families causes the accuracy loss on the original ones (confounded with training size and steps); any improved forecasting method; novelty; results that hold across seeds.

**Plain-English answers.** *Can the three-family model draw figure-eights and waves it has never seen?* No: it does worse than just repeating the last observation. *Can one model learn all five?* Yes, and it forecasts them well compared with simple baselines, but it is a bit worse than the three-family model on the original three. *Is that proof that more families hurt?* No, because the training size and number of steps also changed and only one run was made.

**Sources (all existing):** `results/five_shapes/compare_three_vs_five.json`, `zeroshot_three_shape_model.json`, `dataset_checks.json`, `tables/`, `figures/`, `predictions_test.npz`; `data/processed/five_shapes/metadata.json`; `results/capacity_scaling/capacity_scaling_results.json` (training facts); `results/validation/three_shapes/three_shapes_and_ood.json` (stored three-shape evaluation); `src/glo/five_shapes.py`, `scripts/30_generate_five_shapes.py`, `scripts/31_evaluate_five_shapes.py`; `tests/test_five_shapes.py`. Report generated by `audit/build_five_shapes_report.py`.
