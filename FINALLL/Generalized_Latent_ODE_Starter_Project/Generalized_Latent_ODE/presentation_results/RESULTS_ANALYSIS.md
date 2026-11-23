# Results analysis: Latent Neural ODEs, shared models, unseen families and model size

Date: 2026-11-21. Written for: the project owner, to prepare a discussion with the research supervisor. **Every number comes from a result file already saved in this project** (listed in `RESULTS_SOURCE_INDEX.md`); nothing was trained or re-run. The only new calculations are simple summaries of saved results (percent differences, rankings, a Bonferroni adjustment, horizon windows, and a paired comparison of the two GeoLife models computed from the existing per-segment output), and each is labelled where it appears.

## 0. How to read this document

**Plain-English glossary**
- **Trajectory family:** a type of 2-D path: circle, spiral, ellipse, figure-eight, sinusoid (a wavy line that drifts sideways).
- **Observations:** for each path the model sees only 30 noisy points from the first 7 time units. It never sees the rest.
- **Reconstruction error (RMSE):** how far, on average, the model's curve is from the true path at the times *inside* the observed period where no point was given. Smaller is better.
- **Forecast error (RMSE):** how far the model's prediction is from the true path in the *future* (times 7 to 10). Smaller is better.
- **Baselines (simple comparisons):** *linear interpolation* joins the observed dots with straight lines (reconstruction); *hold-last* predicts the future by staying at the last observed point; *constant velocity* continues in a straight line at the last speed. A learned model is only useful if it beats these.
- **Shared model:** one model trained on several families. **Specialist model:** a separate model trained on one family only.
- **Zero-shot (unseen family):** testing a model on a family that was **not** in its training data. A model tested on a family it was trained on is *not* zero-shot.
- **Seed / training run:** each training run starts from random numbers; two runs of the same recipe give slightly different models. "Single-seed" = one run only (can be luck); "multi-seed" = several runs (shows how much results vary).
- **p-value / confidence interval:** a small p-value (for example below 0.05) says a difference is unlikely to be pure chance *in the data tested*. "Holm" and "Bonferroni" are corrections used when many comparisons are made at once.

**Evidence labels used below:** **[MULTI-SEED]** several training runs were compared; **[SINGLE-SEED]** one training run per model, so the result may change with another seed; **[DESCRIPTIVE]** numbers are reported but not tested.

## 1. Experiments completed

| # | Experiment | Evidence | Main report / result files |
|---|---|---|---|
| E1 | Why the first model failed, and the working "Model D" (two families) | single-seed | `reports/reconstruction_failure_analysis.md`, `reports/improved_two_shapes_results.md` |
| E2 | Is Model D reproducible? (5 seeds, two families) | MULTI-SEED | `results/validation/seed_study/` |
| E3 | Shared model vs specialist models (circle, spiral) | MULTI-SEED (5 vs 3) | `results/validation/shared_vs_separate/` |
| E4 | Three families; two-family models tested on unseen ellipses | MULTI-SEED (ellipse test), single-seed (3-family model) | `results/validation/three_shapes/` |
| E5 | Does conditioning the dynamics on an inferred code help? | MULTI-SEED (5 seeds) | `results/conditioning/`, `reports/conditioning_study_report.md` |
| E6 | Sensitivity to the number of observations | single-seed | `results/validation/nobs_ablation/` |
| E7 | Frozen synthetic model on real GPS (GeoLife) | single-seed | `results/geolife/main_evaluation/`, `reports/geolife_real_world_interpolation_report.md` |
| E8 | GeoLife-trained models (scratch vs fine-tuned from synthetic) | single-seed | `results/geolife/main_evaluation/geolife_trained_*` |
| E9 | Five families: dataset, zero-shot test of the 3-family model, 5-family model | single-seed | `results/five_shapes/`, `reports/five_shapes_report.md` |
| E10 | Capacity scaling (latent 8/16/32, GRU 64/128), training streams not controlled | single-seed | `results/capacity_scaling/` |
| E11 | Controlled 2x2 ablation (latent x GRU width) | single-seed | `results/ablation_2x2/` |
| E12 | Independent audit and report corrections | verification | `PROJECT_COMPLETION_AUDIT.md` |

Files named `SMOKE_ONLY_*` in `results/` and `checkpoints/` are quick software tests, not experiments, and are not used.

### Details of each experiment

**E1. Failure diagnosis and Model D** [SINGLE-SEED].
- *Purpose:* find out why the first checkpoint produced collapsed curves, and obtain a working model. *Models:* the original-size configuration (latent 4, small networks) versus Model D (latent 8, hidden 64, two layers; 24,602 parameters). *Data:* two families (circle, spiral), 1,400 / 300 / 300 train / validation / test. *Method:* the first checkpoint turned out to be from a run stopped after about 3 epochs; candidate configurations were trained fully and chosen by validation loss. *Metrics:* reconstruction and forecast RMSE on the 300 test trajectories.
- *Result (test):* reconstruction 0.756 to 0.056; forecast 0.843 to 0.192. Baselines: linear interpolation 0.066 (reconstruction); hold-last 1.209 and constant velocity 2.366 (forecast). *Supported:* the failure was mostly an unfinished training run plus a too-small latent space, not a coding bug; Model D forecasts far better than the simple baselines. It does not show the larger configuration is optimal.

**E2. Reproducibility of Model D** [MULTI-SEED].
- *Purpose:* check that one lucky run was not behind the results. *Models:* Model D trained with 5 seeds (42, 123, 2024, 2025, 2026). *Data:* two-family test split (300). *Metrics:* mean per-trajectory RMSE, mean and SD across seeds, t-based 95% interval.
- *Result:* reconstruction 0.0599 +/- 0.0069; forecast 0.2194 +/- 0.0502 (95% interval [0.157, 0.282]). The five forecast values range from 0.169 to 0.283. *Supported:* reconstruction is stable; **forecasting varies a lot between seeds** (coefficient of variation about 23%). This is the key yardstick for judging all single-seed differences.

**E3. Shared vs specialist** [MULTI-SEED]. Details in section 3.

**E4. Three families, and unseen ellipses** [MULTI-SEED for the ellipse test].
- *Purpose:* does adding a third family (ellipse) work, and do models trained only on circles and spirals handle ellipses? *Models:* a three-family model (1 seed) and five two-family models (5 seeds). *Data:* the three-family test split (450); the 150 ellipses are unseen by the two-family models. *Metrics:* reconstruction / forecast RMSE and ellipse geometry (axis ratio, rotation direction).
- *Result:* see section 4. *Supported:* rotation direction transfers; ellipse shape does not (seed-42 model: predicted axis ratio 0.92 against true 0.59).

**E5. Dynamics conditioning** [MULTI-SEED]. Details in section 6.1.

**E6. Number of observations** [SINGLE-SEED].
- *Purpose:* does a model trained with exactly 30 observations work with other counts? *Model:* Model D (validation split). *Result:* reconstruction RMSE 0.061 at 30 observations but 0.494 at 20, 0.363 at 40 and 0.595 at 50. *Supported:* the models only work at the observation count they were trained on. All results in this document use 30 observations.

**E7-E8. GeoLife (real GPS)** [SINGLE-SEED]. Details in section 6.2.

**E9. Five families** [SINGLE-SEED]. Details in sections 2 and 4.

**E10. Capacity scaling** [SINGLE-SEED]. Details in section 6.3.

**E11. Controlled 2x2 ablation** [SINGLE-SEED]. Details in section 5.

## 2. The five trajectory families

Data: 1,000 trajectories per family (700 / 150 / 150 train / validation / test), 200 time points on [0, 10], noise SD 0.05, 30 observations at t <= 7. Test: 750 trajectories (150 per family). Models: the three-family model trained on circle, spiral, ellipse only (frozen, so figure-eight and sinusoid are unseen to it); the five-family models trained on all five: the earlier 8/64 run, and the four controlled cells A to D (A = 8/64 is the same architecture as the earlier run). **Every model here was trained once** [SINGLE-SEED]. Mean of per-trajectory RMSE.

**Reconstruction error** (lower is better; simple baseline = linear interpolation)

| Family | Linear interp. | 3-family model (frozen) | 5-family earlier 8/64 | A 8/64 | B 16/64 | C 8/128 | D 16/128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| circle | 0.0699 | 0.0622 | 0.0701 | 0.0607 | 0.0416 | 0.0514 | 0.0369 |
| spiral | 0.0653 | 0.0933 | 0.1131 | 0.0883 | 0.0466 | 0.0711 | 0.0434 |
| ellipse | 0.0637 | 0.0888 | 0.0973 | 0.0946 | 0.0436 | 0.0668 | 0.0408 |
| figure-eight (3-family model: unseen) | 0.0917 | 0.7321 | 0.1086 | 0.1364 | 0.0654 | 0.0754 | 0.0568 |
| sinusoid (3-family model: unseen) | 0.0452 | 0.3441 | 0.0647 | 0.0607 | 0.0336 | 0.0450 | 0.0282 |
| all five | 0.0672 | 0.2641 | 0.0908 | 0.0882 | 0.0462 | 0.0619 | 0.0412 |

**Forecast error** (lower is better; simple baselines = hold-last and constant velocity)

| Family | Hold-last | Const. velocity | 3-family model (frozen) | 5-family earlier 8/64 | A 8/64 | B 16/64 | C 8/128 | D 16/128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| circle | 1.1370 | 2.3274 | 0.1401 | 0.2594 | 0.2676 | 0.3782 | 0.2882 | 0.5182 |
| spiral | 1.2763 | 2.3877 | 0.2908 | 0.4270 | 0.4915 | 0.5353 | 0.4987 | 0.6605 |
| ellipse | 1.1200 | 2.0143 | 0.1885 | 0.2609 | 0.3916 | 0.3749 | 0.3312 | 0.5240 |
| figure-eight (3-family model: unseen) | 0.9610 | 2.6176 | 1.1174 | 0.2992 | 0.4740 | 0.4511 | 0.4310 | 0.4845 |
| sinusoid (3-family model: unseen) | 0.6248 | 1.3898 | 0.9734 | 0.3861 | 0.3744 | 0.8602 | 0.5751 | 0.7808 |
| all five | 1.0238 | 2.1474 | 0.5421 | 0.3265 | 0.3998 | 0.5199 | 0.4248 | 0.5936 |

(The 3-family model's "all five" row mixes seen and unseen families. Figures: `figures/fig1_reconstruction_by_family.png`, `figures/fig2_forecast_by_family.png`.)

### Answers

**Which family is easiest to reconstruct?** **Sinusoid** in 6 of the 7 five-family models listed (all except cell A, where circle 0.0607 and sinusoid 0.0607 are tied). For the earlier 8/64 run the values are sinusoid 0.0647 and circle 0.0701. Note that linear interpolation also finds the sinusoid easiest (0.0452), so part of this is simply that a slowly drifting wave is easy to join up. Relative to linear interpolation, B and D do about equally well on every family (ratios 0.5 to 0.75), so sinusoid is not *relatively* easier.
**Which is hardest to reconstruct?** **Figure-eight** in 6 of 7 models (all except the earlier 8/64 run, where spiral is hardest). For cell A, figure-eight is 0.1364, about 2.2 times the circle's error. Linear interpolation also finds it hardest (0.0917): the path turns sharply and crosses itself.
**Which is hardest to forecast?** It depends on the model size. For the two smallest models (earlier 8/64 run: spiral 0.4270; cell A: spiral 0.4915) the **spiral** is hardest. For the 5 larger models (B, C, D, earlier 16/128, earlier 32/128) the **sinusoid** is hardest: B 0.8602, D 0.7808, and for B and D this is *worse than the hold-last baseline* (0.6248). Circle and ellipse are generally the easiest to forecast.
**Does the model perform equally well on all five families?** **No.** (i) Forecast RMSE for cell A ranges from 0.2676 (circle) to 0.4915 (spiral). (ii) The smallest models mostly do not reconstruct better than linear interpolation (earlier 8/64 run: tied on circle, worse on the other four, ratios 1.00 to 1.73; cell A: better only on circle, ratio 0.87, and worse on the others, 1.34 to 1.49), while B and D beat it on every family (ratios 0.5 to 0.75) and C beats it on circle and figure-eight and, by a hair, on sinusoid (0.0450 vs 0.0452), but not on spiral or ellipse. (iii) All models forecast far better than hold-last (overall 1.0238 against 0.3998 to 0.5936) and constant velocity (2.1474), with the one exception noted for sinusoid.

**What can we conclude about one shared model for different patterns?** *Verified observation:* a single model with no shape label, trained once on all five families, produces sensible reconstructions and forecasts for all of them and forecasts far better than the simple baselines (one training run per model). *Cost observed:* compared with the three-family model, the five-family model is worse on the original three families (section 4). *Not established:* that it learns the *same underlying dynamics* as specialist models, or that it generalises to new families. These are single-seed results.

## 3. Shared model vs specialist models

**What was compared.** *A. Shared model:* one Model D trained on circles and spirals together (1,400 training trajectories, 300 epochs, 5 training runs: seeds 42, 123, 2024, 2025, 2026). *B. Specialist models:* a separate Model D trained on circles only and another on spirals only (700 training trajectories, 600 epochs, so the same number of optimiser steps; 3 training runs each: seeds 42, 123, 2024). Same architecture and settings; evaluated on the same 150 test circles and 150 test spirals with identical observations. **[MULTI-SEED: 5 shared vs 3 specialist runs]**. Source: `results/validation/shared_vs_separate/shared_vs_separate.json`.

| Family | Metric | Shared (5 runs) mean +/- SD | Specialist (3 runs) mean +/- SD | Shared vs specialist | Welch test p (raw / x4 Bonferroni*) | Every shared run worse than every specialist run? | Same-seed paired test on trajectories (seed 42): difference, p |
|---|---|---|---|---:|---|---|---|
| circle | reconstruction | 0.0532 +/- 0.0059 | 0.0457 +/- 0.0008 | +16.3% | 0.047 / 0.19 | yes | +12.3%, p=0.0004 |
| circle | forecast | 0.1606 +/- 0.0369 | 0.0868 +/- 0.0061 | +84.9% | 0.0099 / 0.04 | yes | +75.6%, p=4.8e-15 |
| spiral | reconstruction | 0.0680 +/- 0.0102 | 0.0632 +/- 0.0008 | +7.5% | 0.36 / 1 | no | -2.8%, p=0.64 |
| spiral | forecast | 0.2841 +/- 0.0671 | 0.1983 +/- 0.0086 | +43.3% | 0.045 / 0.18 | yes | +26.9%, p=7.9e-07 |

*Bonferroni adjustment (multiply by the 4 tests) computed in this analysis; the study itself reported the raw Welch p-values. Figure: `figures/fig3_shared_vs_specialist.png`.

**Which model performs better?** The **specialists**, especially for forecasting. Forecast error is 85% higher for the shared model on circles and 43% higher on spirals. Reconstruction is close: 16% higher on circles (small but consistent) and 8% on spirals (not distinguishable).

**Does the shared model learn all families?** Yes: it reconstructs and forecasts both families well compared with the simple baselines (forecast about 0.16 and 0.28 against hold-last about 1.1 to 1.3). It just does so less accurately than a model devoted to one family.

**Does combining families hurt forecasting?** For circle + spiral, **yes, by a lot**. Evidence strength:
- *Statistically supported:* circle forecast. Every one of the 5 shared runs forecasts worse than every one of the 3 specialist runs (see figure 3); Welch p = 0.0099 raw, 0.04 after Bonferroni; the same-seed trajectory-level comparison gives p = 4.8e-15 with the shared model better on only 21% of trajectories.
- *Supported but weaker:* spiral forecast. Again all 5 shared runs are worse than all 3 specialists, but with only 3 specialist runs the Welch p-value is 0.045 raw (0.18 after Bonferroni), so after correcting for four tests it is not significant at the 5% level.
- *Descriptive only:* reconstruction. Circle p = 0.047 raw (0.19 corrected); spiral p = 0.36.
- The trajectory-level p-values come from a single pair of runs (seed 42) and only describe how the two trained models differ on these test trajectories; they do not show the result would repeat for new training runs. The seed-level comparison (5 vs 3 runs) is the more relevant one.

**Advantages and limitations of a shared model**
- *Advantages (observed):* one model and one set of weights for several families; it needs no shape label; it handles all of them far better than the simple baselines.
- *Limitations (observed):* about 40% to 85% worse forecasts than specialists on circle and spiral; the shared model's forecasts also vary much more between training runs (SD 0.037 and 0.067 against 0.006 and 0.009 for specialists; the specialists had only 3 runs).
- *Not tested:* specialists for ellipse, figure-eight and sinusoid do not exist, so the shared-vs-specialist comparison is only for circle and spiral. An attempt to close the gap by conditioning the dynamics on an inferred code did not work (section 6.1).
- *Related, from the same study:* adding a third family (ellipse) to the shared model did not clearly change forecasting for circles and spirals: circle forecast 0.136 for the three-family model, 0.142 for the seed-42 two-family model (paired test on trajectories, p = 0.74) and 0.157 +/- 0.037 for the two-family model over 5 seeds. Circle reconstruction was 17% worse than the seed-42 two-family model.

## 4. Cross-family generalisation (unseen families)

**Rule used:** a result counts as zero-shot only if the family was absent from that model's training data.

### 4a. Two-family models on unseen **ellipses** [MULTI-SEED: 5 training runs]

Trained on: circle + spiral. Excluded: ellipse. Tested on the 150 ellipses of the three-family test split (30 observations). Source: `results/validation/three_shapes/three_shapes_and_ood.json`.

| Model | Seen ellipses in training? | Reconstruction RMSE | Forecast RMSE |
|---|---|---:|---:|
| Two-family models (5 runs), mean +/- SD | **no (zero-shot)** | 0.2096 +/- 0.0040 | 0.3212 +/- 0.0274 |
| Three-family model (1 run) | yes (not zero-shot) | 0.0887 | 0.1850 |
| Linear interpolation (baseline) | n/a | 0.0625 | n/a |
| Hold-last (baseline) | n/a | n/a | 1.1213 |
| Constant velocity (baseline) | n/a | n/a | 2.2082 |

*What happened:* forecasting transfers partly (zero-shot 0.32, far better than hold-last 1.12, but worse than the in-training 0.18); reconstruction does not (0.21, more than three times worse than simply joining the dots, 0.06). Rotation direction is right for every trajectory (geometry numbers in this paragraph are for the seed-42 two-family model only; accuracy 100%), but the model draws ellipses as near-circles: predicted axis ratio 0.92 against true 0.59, correlation 0.18 (the three-family model that saw ellipses: 0.86). An ellipse is a stretched circle, so this is the *easiest* possible unseen family; it counts as partial transfer of the forecast, not successful generalisation.

### 4b. The three-family model on unseen **figure-eight** and **sinusoid** [SINGLE-SEED: one frozen model]

Trained on: circle + spiral + ellipse. Excluded: figure-eight, sinusoid. Tested on the five-family test split (150 of each unseen family). Source: `results/five_shapes/zeroshot_three_shape_model.json`.

| Family | Status | Recon. RMSE | Linear interp. | Forecast RMSE | Hold-last | Const. velocity |
|---|---|---:|---:|---:|---:|---:|
| circle | seen in training | 0.0622 | 0.0699 | 0.1401 | 1.1370 | 2.3274 |
| spiral | seen in training | 0.0933 | 0.0653 | 0.2908 | 1.2763 | 2.3877 |
| ellipse | seen in training | 0.0888 | 0.0637 | 0.1885 | 1.1200 | 2.0143 |
| figure-eight | **unseen (zero-shot)** | 0.7321 | 0.0917 | 1.1174 | 0.9610 | 2.6176 |
| sinusoid | **unseen (zero-shot)** | 0.3441 | 0.0452 | 0.9734 | 0.6248 | 1.3898 |

*What happened:* on the unseen families the model is **worse than hold-last** at forecasting (figure-eight 1.1174 vs 0.9610; sinusoid 0.9734 vs 0.6248) and **far worse than linear interpolation** at reconstruction (figure-eight 0.7321 vs 0.0917; sinusoid 0.3441 vs 0.0452). It is better than constant velocity, but that baseline is itself very poor on these paths. The model mostly redraws what it has seen (loops and spirals). **This is not successful zero-shot generalisation.**

### 4c. Five-family model: not zero-shot

The five-family model was trained on all five families, so its results on them (section 2) are **not** generalisation. What it shows is that the same architecture *can* be trained on all five. Compared with the three-family model on the original three families it is worse (paired, identical observations; Holm-adjusted p < 0.01 for every comparison):

| Family | Forecast: 3-family | Forecast: 5-family | Change | Recon: 3-family | Recon: 5-family |
|---|---:|---:|---:|---:|---:|
| circle | 0.1401 | 0.2594 | +85% | 0.0622 | 0.0701 |
| spiral | 0.2908 | 0.4270 | +47% | 0.0933 | 0.1131 |
| ellipse | 0.1885 | 0.2609 | +38% | 0.0888 | 0.0973 |

The five-family model also had more training data and more optimiser steps (33,000 against 19,800), so this cost cannot be blamed on the number of families alone. [SINGLE-SEED]

### 4d. Did the experiments demonstrate zero-shot generalisation?
**No.** Partial transfer for ellipses (a stretched circle: direction and forecast partly right, shape wrong) [MULTI-SEED]; **failure** for figure-eight and sinusoid [SINGLE-SEED]; and, in the real-world test (section 6.2), failure on real GPS. There is no leave-one-family-out experiment (training on four families, testing on the fifth), so nothing here shows generalisation across *many* families.

## 5. Controlled 2x2 ablation [SINGLE-SEED]

Four models, same data, same training recipe, same training data stream (`data_seed` 42), differing only in latent dimension and GRU (encoder) width; one training run each. Test: the same 750 trajectories with identical observations.

| Model | Latent dimension | GRU width | Parameters |
|---|---:|---:|---:|
| A | 8 | 64 | 24,602 |
| B | 16 | 64 | 27,186 |
| C | 8 | 128 | 63,642 |
| D | 16 | 128 | 67,250 |

### Overall results (mean per-trajectory RMSE; change relative to A)

| Model | Reconstruction | vs A | Forecast (7 < t <= 10) | vs A | Trajectories with lower forecast error than A |
|---|---:|---:|---:|---:|---:|
| A | 0.0882 | n/a | 0.3998 | n/a | n/a |
| B | 0.0462 | -47.6% | 0.5199 | +30.1% | 35% |
| C | 0.0619 | -29.8% | 0.4248 | +6.3% | 48% |
| D | 0.0412 | -53.2% | 0.5936 | +48.5% | 27% |

Linear interpolation: 0.0672 (reconstruction). Hold-last: 1.0238; constant velocity: 2.1474 (forecast). Figure: `figures/fig5_ablation_overall.png`.

### Per family (change vs A; Holm-adjusted p-value of the paired difference, "ns" = not significant)

| Family | Metric | A | B vs A | C vs A | D vs A |
|---|---|---:|---|---|---|
| circle | forecast | 0.2676 | 0.3782 (+41.4%; p=3.4e-10) | 0.2882 (+7.7%; ns) | 0.5182 (+93.7%; p=2e-17) |
| circle | reconstruction | 0.0607 | 0.0416 (-31.4%; p=1.8e-19) | 0.0514 (-15.3%; p=8.3e-06) | 0.0369 (-39.1%; p=4.4e-22) |
| spiral | forecast | 0.4915 | 0.5353 (+8.9%; ns) | 0.4987 (+1.5%; ns) | 0.6605 (+34.4%; p=0.001) |
| spiral | reconstruction | 0.0883 | 0.0466 (-47.3%; p=3.1e-23) | 0.0711 (-19.5%; p=4.8e-10) | 0.0434 (-50.9%; p=1.8e-24) |
| ellipse | forecast | 0.3916 | 0.3749 (-4.2%; ns) | 0.3312 (-15.4%; p=0.013) | 0.5240 (+33.8%; p=6.7e-09) |
| ellipse | reconstruction | 0.0946 | 0.0436 (-53.9%; p=1.6e-24) | 0.0668 (-29.4%; p=3.2e-15) | 0.0408 (-56.9%; p=1.5e-24) |
| figure-eight | forecast | 0.4740 | 0.4511 (-4.8%; ns) | 0.4310 (-9.1%; ns) | 0.4845 (+2.2%; ns) |
| figure-eight | reconstruction | 0.1364 | 0.0654 (-52.1%; p=1.6e-23) | 0.0754 (-44.8%; p=5.1e-22) | 0.0568 (-58.4%; p=1.3e-23) |
| sinusoid | forecast | 0.3744 | 0.8602 (+129.8%; p=1.5e-24) | 0.5751 (+53.6%; p=4.9e-18) | 0.7808 (+108.6%; p=1.7e-24) |
| sinusoid | reconstruction | 0.0607 | 0.0336 (-44.7%; p=9.1e-24) | 0.0450 (-25.9%; p=4.5e-13) | 0.0282 (-53.6%; p=1.6e-24) |
| all five | forecast | 0.3998 | 0.5199 (+30.1%; p=1.9e-23) | 0.4248 (+6.3%; ns) | 0.5936 (+48.5%; p=1.2e-50) |
| all five | reconstruction | 0.0882 | 0.0462 (-47.6%; p=8.4e-116) | 0.0619 (-29.8%; p=1.5e-67) | 0.0412 (-53.2%; p=1.1e-119) |

### Separate comparisons

**A vs B: raising only the latent dimension (8 to 16).** Reconstruction improves in every family (-47.6% overall). Overall forecast error rises +30.1% (Holm p = 1.9e-23); B beats A on 35% of the paired trajectories. Per family the forecast difference is clearly significant only for circle (+41%) and sinusoid (+130%); spiral, ellipse and figure-eight show no significant difference.
**A vs C: raising only the GRU width (64 to 128).** Reconstruction improves in every family (-29.8% overall, a smaller gain than B). Forecast: **no clearly distinguishable overall difference** (+6.3%; mean difference +0.0250, bootstrap 95% interval [+0.0031, +0.0461], median difference +0.0109, better on 48% of trajectories, Holm p = 0.18). By family: significantly worse on sinusoid (+54%), significantly *better* on ellipse (-15%), nothing distinguishable on the others.
**A vs D: raising both.** Reconstruction improves most (-53.2%). Forecast error rises +48.5% (Holm p = 1.2e-50), D beating A on only 27% of trajectories. D has the higher mean on all five families; significantly so on circle, spiral, ellipse and sinusoid, not on figure-eight.

### The interaction, explained simply
An *interaction* means the effect of one change depends on the other. Overall forecast error: raising the latent size changes it by +0.120 when the GRU is 64 (B minus A) but by +0.169 when the GRU is 128 (D minus C). The extra damage (+0.049, 95% interval [+0.025, +0.073], Holm p = 0.0018) means the two changes **reinforce each other** overall. For reconstruction the interaction is +0.021: the two changes overlap (each helps less once the other is made). The pattern is not the same in every family: for the sinusoid the forecast interaction is -0.28, the opposite sign. The "GRU main effect" of +0.049 is an average over both latent sizes and is carried mainly by D versus B (+0.074), not by C versus A (+0.025), so the two factors should not be described as independently worsening forecasts.

### Forecast error over time (verified)

Mean per-trajectory RMSE inside time windows (computed in this analysis from the saved predictions; 750 trajectories):

| Time window | A | B | C | D |
|---|---:|---:|---:|---:|
| 7 < t <= 8 | 0.2630 | 0.2174 | 0.1755 | 0.2332 |
| 8 < t <= 9 | 0.3940 | 0.4926 | 0.3649 | 0.5684 |
| 9 < t <= 10 | 0.4818 | 0.7037 | 0.5951 | 0.8101 |

**Verified:** in the first part of the forecast (7 < t <= 8) the larger models B, C and D are *better* than A (lower error). Their error then grows faster and crosses above A at t = 8.09 (B), 8.64 (C) and 7.94 (D) (first time the across-trajectory RMSE exceeds A's, from the saved error-versus-time curves). By 9 < t <= 10 all three are worse than A. At t = 10 the across-trajectory RMSE is A 0.623, B 0.912, C 0.828, D 1.012. So the full-window "worse forecast" comes from the long-horizon part. A shorter forecast window would give the opposite sign for B and D. (Figure: `figures/fig6_forecast_error_vs_time.png`.)

### What the ablation does and does not show
**Observed (single seed):** larger encoders reconstruct much better; over the full 7 to 10 window latent size is associated with higher forecast error, GRU width alone is not clearly distinguishable from baseline, the effect appears late in the horizon and is driven mostly by the sinusoid and circle families (for B and D on sinusoid the model is worse than hold-last).
**Not shown:** that capacity universally worsens forecasting; a causal mechanism; that the result holds for other training seeds. For scale: training the *same* architecture twice with different data streams changed mean forecast error by +22.4% (A vs the earlier 8/64 run) and +20.7% (D vs the earlier 16/128 run), and the five-seed two-family study has a forecast coefficient of variation of about 23%. The forecast differences above (+30%, +6%, +48%) are of that same order. (Figure: `figures/fig7_effects_vs_seed_variation.png`.) A tiny p-value here only says the two *fixed trained models* differ on these test trajectories.
**Possible explanations (untested):** the training objective supervises only the 30 observed points, so extrapolation is not directly trained; larger latent spaces might make errors grow faster; optimisation differences (all cells were still improving at epoch 300). Classical overfitting was not observed (validation loss did not rise; training and validation negative log-likelihood differ by at most 8.4).

## 6. Other completed experiments

### 6.1 Dynamics-conditioning study [MULTI-SEED: 5 seeds; M2aug 3 seeds]

*Purpose:* the shared model forecasts worse than specialists; does letting the model infer an extra "dynamics code" that controls the ODE close that gap? *Models:* M1 = shared Model D; M2 = shared model with the same extra size but no code (a fair control); M3 = shared model with the dynamics code; M2aug = an ablation. *Data:* circle + spiral, 300 test trajectories. Forecast RMSE, mean +/- SD over seeds (source `results/conditioning/conditioning_study.json`):

| Model | Circle | Spiral |
|---|---:|---:|
| Specialists (3 runs) | 0.087 | 0.198 |
| M1 shared | 0.145 +/- 0.007 | 0.264 +/- 0.036 |
| M2 control, no code | 0.172 +/- 0.096 | 0.270 +/- 0.113 |
| M3 with dynamics code | 0.176 +/- 0.066 | 0.299 +/- 0.068 |
| M2aug ablation (3 runs) | 0.301 +/- 0.108 | 0.435 +/- 0.095 |

*Result:* M3 does **not** forecast better than M1 or M2 (circle +21% and spiral +13% worse than M1 when averaged over seeds; seed-level paired p = 0.35 and 0.38, so not distinguishable across seeds). M3 reconstructs very slightly better (circle -4.3%). The learned code does control radial growth, so it is *used*, but it did not help forecasting. M3 was not tuned separately, which may favour M1. *Supported:* at this scale the hypothesis is not supported. *Note:* two sets of five "shared" runs (this study's M1 and the E3 study) give the same recipe different SDs for circle forecast (0.007 here, 0.037 in E3), a reminder that five-seed SDs are themselves uncertain.

### 6.2 GeoLife real-world transfer [SINGLE-SEED]

*Purpose:* does what was learned on synthetic shapes help on real GPS? *Data:* 1,000 held-out GPS segments from 97 test users (the models never saw these users; development users only were used for training the GeoLife-trained models). *Task:* interpolate withheld GPS positions with 30 observations per segment. *Metric:* median per-segment error in metres. (GeoLife data and weights are not distributed; only aggregates are reported.)

| Method (30 observations) | Median error, metres [95% CI, user bootstrap] | Ratio to linear interpolation (geometric mean) |
|---|---|---:|
| Linear interpolation | 7.10 [5.8, 8.5] | 1.000 |
| Cubic spline | 5.56 [4.6, 6.9] | 0.799 |
| PCHIP | 5.62 [4.5, 6.7] | 0.778 |
| Frozen synthetic (three-family) model | 126.40 [110.9, 147.9] | 17.278 |
| GeoLife-trained, from scratch | 60.62 [51.6, 71.7] | 8.222 |
| Synthetic, then fine-tuned on GeoLife | 47.11 [42.2, 54.6] | 6.415 |

*Results and conclusions supported.* (1) The **frozen synthetic model does not transfer**: about 17 times worse than linear interpolation. (2) Training on real GPS data helps (about 2 to 3 times better than the frozen model) but both GeoLife-trained models remain 6 to 8 times worse than linear interpolation and beat it on only 0.1% of segments. (3) Forecasts are also worse than the trivial baselines (protocol A medians: scratch 322 m, fine-tuned 371 m, hold-last 229 m, constant velocity 151 m). The likely reason (hypothesis) is that the model can only produce loop-like curves. Figure: `figures/fig8_geolife_transfer.png`.

**Does synthetic pre-training help? Paired comparison, fine-tuned vs from-scratch (computed in this analysis from the saved per-segment results of the same 1,000 segments; ratio below 1 favours fine-tuning; 95% interval from a bootstrap over the 97 users):**

| Observations per segment | Median error, scratch (m) | Median error, fine-tuned (m) | Geometric-mean ratio fine-tuned / scratch [95% CI] | Fine-tuned better on |
|---|---:|---:|---|---:|
| 10 | 119.6 | 191.7 | 1.65 [1.61, 1.68] | 7% of segments |
| 20 | 75.1 | 94.8 | 1.22 [1.19, 1.25] | 29% of segments |
| 30 | 60.6 | 47.1 | 0.78 [0.76, 0.80] | 81% of segments |
| 40 | 59.8 | 76.7 | 1.29 [1.25, 1.32] | 23% of segments |
| 50 | 60.0 | 115.5 | 1.93 [1.87, 2.00] | 8% of segments |

*Observed:* fine-tuning from synthetic weights is clearly better only at **30 observations**, the number both models were trained with (ratio 0.78); at 10, 20, 40 and 50 observations it is clearly *worse* (ratios 1.65, 1.22, 1.29, 1.93), and the from-scratch model degrades much less away from 30. So **no general benefit of synthetic pre-training is shown**; the one advantage is at the training count. *Possible explanation (untested):* the fine-tuned model is more tied to the observation count it was trained with. Single training run per model, so this could change with other seeds.

### 6.3 Capacity scaling, first version [SINGLE-SEED]

*Purpose:* does a bigger encoder help? *Models (five-family data):* earlier 8/64 baseline (24,602 parameters), medium 16/128 (67,250), large 32/128 (74,466), one run each, **training data streams not controlled** (this is why the 2x2 ablation was run). *Overall mean RMSE (reconstruction / forecast):* baseline 0.0908 / 0.3265; medium 0.0399 / 0.4919; large 0.0432 / 0.5975. *Supported:* reconstruction improves with size; forecasting is worse for the larger models on the full window, with the sinusoid worst (large: 1.207, worse than hold-last 0.6248). The controlled 2x2 reproduces both directions: for the combined 16/128 change the full-window forecast difference is +48.5% against the controlled baseline A, similar to the +50.6% of the earlier uncontrolled medium run, and the 2x2 shows that it is split unevenly between latent size and GRU width. The earlier 8/64 baseline (0.3265) was better than the controlled rerun A (0.3998), which shows how much a different data stream alone can change a single run.

## 7. The five most important findings

**1. One shared model can learn several families, but it forecasts worse than specialists.**
- *Simple explanation:* Training one model on circles and spirals together gives about 85% (circle) and 43% (spiral) higher forecast error than separate models; reconstruction is close.
- *Numbers:* Circle forecast 0.161 vs 0.087; spiral 0.284 vs 0.198 (E3).
- *Seed evidence:* MULTI-SEED (5 shared vs 3 specialist runs).
- *What we can safely conclude:* Combining families costs forecast accuracy on circle and spiral for this model and recipe; every shared run was worse than every specialist run.
- *What remains uncertain:* Only two families have specialists; spiral difference is not significant after multiple-test correction; whether the gap closes with more capacity or data is untested; conditioning did not close it.

**2. The models do not generalise to unseen families.**
- *Simple explanation:* A model trained on circle, spiral and ellipse cannot draw figure-eights or sinusoids it has not seen, and models trained without ellipses draw ellipses as near-circles.
- *Numbers:* Figure-eight forecast 1.1174 vs hold-last 0.9610; sinusoid 0.9734 vs 0.6248; reconstruction 0.7321 / 0.3441 vs linear interpolation 0.0917 / 0.0452. Unseen ellipses (5 runs): forecast 0.32 (hold-last 1.12) but reconstruction 0.21 vs 0.06.
- *Seed evidence:* MULTI-SEED for ellipses (5 runs); SINGLE-SEED for figure-eight and sinusoid.
- *What we can safely conclude:* There is no zero-shot generalisation to dissimilar families; the ellipse case shows only partial transfer (direction and rough forecast).
- *What remains uncertain:* No leave-one-family-out experiment; only one frozen model for the two dissimilar families; other training recipes might generalise differently.

**3. Larger encoders reconstruct much better, consistently.**
- *Simple explanation:* Making the encoder bigger (latent dimension, GRU width, or both) lowers reconstruction error in every family, and turns a model that is worse than straight-line interpolation into one that beats it.
- *Numbers:* Overall reconstruction A 0.0882, B 0.0462 (-47.6%), C 0.0619 (-29.8%), D 0.0412 (-53.2%); linear interpolation 0.0672 (E10, E11).
- *Seed evidence:* SINGLE-SEED (but the same direction in two separate experiments and in all five families).
- *What we can safely conclude:* Within this setup, more encoder capacity improves reconstruction of withheld points.
- *What remains uncertain:* Not confirmed across seeds; reconstruction is what the training loss directly optimises, so this may be expected; one dataset family.

**4. Forecasting gets worse with model size at long horizons, but this is not established.**
- *Simple explanation:* Larger models forecast better just after the observed period and worse later; over the whole forecast window the latent-size changes are worse than baseline A while widening the GRU alone makes no clear difference.
- *Numbers:* Overall forecast A 0.3998, B 0.5199 (+30.1%), C 0.4248 (+6.3%, not significant), D 0.5936 (+48.5%). Window 7-8: A 0.263, D 0.233; window 9-10: A 0.482, D 0.810 (E11).
- *Seed evidence:* SINGLE-SEED.
- *What we can safely conclude:* In this one set of runs the longer-horizon forecast error grows faster for models with a larger latent space.
- *What remains uncertain:* The differences are the same size as run-to-run variation (+22% and +21% for identical architectures; five-seed CV about 23%); no cause is established; needs multi-seed runs.

**5. A model trained on synthetic shapes does not transfer to real GPS, and real-data training still does not beat simple interpolation.**
- *Simple explanation:* On real GPS tracks, straight-line (or spline) interpolation of the observed points is far more accurate than any of our learned models.
- *Numbers:* Median error at 30 observations: linear 7.1 m, PCHIP 5.6 m; frozen synthetic model 126.4 m; GeoLife-trained 60.6 m; fine-tuned 47.1 m (E7, E8). Fine-tuning helps only at 30 observations.
- *Seed evidence:* SINGLE-SEED.
- *What we can safely conclude:* Synthetic-only training does not give a usable real-world model; the learned models are 6 to 17 times worse than linear interpolation at interpolating GPS.
- *What remains uncertain:* One dataset, one training run per model, models trained with 30 observations only, and the Latent ODE was not tuned specifically for GPS beyond one configuration.

**Verified experimental observations (each with the seed caveat given in its finding):** findings 1, 2 and 5, and the direction of the reconstruction improvement in finding 3.
**Possible explanations that need further investigation:** why shared models forecast worse; why larger latent spaces forecast worse late in the horizon (is it the observed-points-only training objective? optimisation? seed luck?); why fine-tuned models are tied to the training observation count; whether any of these change with more seeds.

## 8. Strength of evidence at a glance

| Finding | Seeds | Strength |
|---|---|---|
| Shared model forecasts worse than specialists (circle, spiral) | 5 vs 3 | Supported (circle clearly; spiral weaker) |
| Reproducibility of Model D reconstruction | 5 | Supported (stable); forecast varies by about 23% |
| Conditioning does not help | 5 | Supported at this scale |
| Two-family models on unseen ellipses | 5 | Supported: partial transfer only |
| 3-family model on unseen figure-eight, sinusoid | 1 | Descriptive; one frozen model |
| Five-family model results; three vs five comparison | 1 | Descriptive; budgets differ |
| Larger encoders improve reconstruction | 1 (two experiments) | Consistent but single-seed |
| Larger models forecast worse late in the horizon | 1 | Exploratory; same size as seed noise |
| GeoLife transfer fails; fine-tuned vs scratch pattern | 1 | Single-seed; large effect sizes |

**Not done:** multi-seed repeats of the five-family experiments, leave-one-family-out generalisation, stronger baselines (per-family curve fitting, sequence models), a test of the observed-points-only objective hypothesis, any method that improves forecasting.
