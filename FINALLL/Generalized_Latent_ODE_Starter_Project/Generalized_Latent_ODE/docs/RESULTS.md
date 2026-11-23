# Results

Verified numerical results of every experiment, with single-seed and multi-seed findings kept apart. Every number is traceable to a file under `results/`; the table of sources follows the analysis. Tables of per-experiment metrics: `results/*/tables/`. Per-segment GeoLife tables are not published (see [DATASETS.md](DATASETS.md)).

## Results analysis: Latent Neural ODEs, shared models, unseen families and model size

Date: 2026-11-21. Written for: the project owner, to prepare a discussion with the research supervisor. **Every number comes from a result file already saved in this project** (listed in `RESULTS_SOURCE_INDEX.md`); nothing was trained or re-run. The only new calculations are simple summaries of saved results (percent differences, rankings, a Bonferroni adjustment, horizon windows, and a paired comparison of the two GeoLife models computed from the existing per-segment output), and each is labelled where it appears.

### 0. How to read this document

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

### 1. Experiments completed

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

#### Details of each experiment

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

### 2. The five trajectory families

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

(The 3-family model's "all five" row mixes seen and unseen families. Figures: `../presentation_results/figures/fig1_reconstruction_by_family.png`, `../presentation_results/figures/fig2_forecast_by_family.png`.)

#### Answers

**Which family is easiest to reconstruct?** **Sinusoid** in 6 of the 7 five-family models listed (all except cell A, where circle 0.0607 and sinusoid 0.0607 are tied). For the earlier 8/64 run the values are sinusoid 0.0647 and circle 0.0701. Note that linear interpolation also finds the sinusoid easiest (0.0452), so part of this is simply that a slowly drifting wave is easy to join up. Relative to linear interpolation, B and D do about equally well on every family (ratios 0.5 to 0.75), so sinusoid is not *relatively* easier.
**Which is hardest to reconstruct?** **Figure-eight** in 6 of 7 models (all except the earlier 8/64 run, where spiral is hardest). For cell A, figure-eight is 0.1364, about 2.2 times the circle's error. Linear interpolation also finds it hardest (0.0917): the path turns sharply and crosses itself.
**Which is hardest to forecast?** It depends on the model size. For the two smallest models (earlier 8/64 run: spiral 0.4270; cell A: spiral 0.4915) the **spiral** is hardest. For the 5 larger models (B, C, D, earlier 16/128, earlier 32/128) the **sinusoid** is hardest: B 0.8602, D 0.7808, and for B and D this is *worse than the hold-last baseline* (0.6248). Circle and ellipse are generally the easiest to forecast.
**Does the model perform equally well on all five families?** **No.** (i) Forecast RMSE for cell A ranges from 0.2676 (circle) to 0.4915 (spiral). (ii) The smallest models mostly do not reconstruct better than linear interpolation (earlier 8/64 run: tied on circle, worse on the other four, ratios 1.00 to 1.73; cell A: better only on circle, ratio 0.87, and worse on the others, 1.34 to 1.49), while B and D beat it on every family (ratios 0.5 to 0.75) and C beats it on circle and figure-eight and, by a hair, on sinusoid (0.0450 vs 0.0452), but not on spiral or ellipse. (iii) All models forecast far better than hold-last (overall 1.0238 against 0.3998 to 0.5936) and constant velocity (2.1474), with the one exception noted for sinusoid.

**What can we conclude about one shared model for different patterns?** *Verified observation:* a single model with no shape label, trained once on all five families, produces sensible reconstructions and forecasts for all of them and forecasts far better than the simple baselines (one training run per model). *Cost observed:* compared with the three-family model, the five-family model is worse on the original three families (section 4). *Not established:* that it learns the *same underlying dynamics* as specialist models, or that it generalises to new families. These are single-seed results.

### 3. Shared model vs specialist models

**What was compared.** *A. Shared model:* one Model D trained on circles and spirals together (1,400 training trajectories, 300 epochs, 5 training runs: seeds 42, 123, 2024, 2025, 2026). *B. Specialist models:* a separate Model D trained on circles only and another on spirals only (700 training trajectories, 600 epochs, so the same number of optimiser steps; 3 training runs each: seeds 42, 123, 2024). Same architecture and settings; evaluated on the same 150 test circles and 150 test spirals with identical observations. **[MULTI-SEED: 5 shared vs 3 specialist runs]**. Source: `results/validation/shared_vs_separate/shared_vs_separate.json`.

| Family | Metric | Shared (5 runs) mean +/- SD | Specialist (3 runs) mean +/- SD | Shared vs specialist | Welch test p (raw / x4 Bonferroni*) | Every shared run worse than every specialist run? | Same-seed paired test on trajectories (seed 42): difference, p |
|---|---|---|---|---:|---|---|---|
| circle | reconstruction | 0.0532 +/- 0.0059 | 0.0457 +/- 0.0008 | +16.3% | 0.047 / 0.19 | yes | +12.3%, p=0.0004 |
| circle | forecast | 0.1606 +/- 0.0369 | 0.0868 +/- 0.0061 | +84.9% | 0.0099 / 0.04 | yes | +75.6%, p=4.8e-15 |
| spiral | reconstruction | 0.0680 +/- 0.0102 | 0.0632 +/- 0.0008 | +7.5% | 0.36 / 1 | no | -2.8%, p=0.64 |
| spiral | forecast | 0.2841 +/- 0.0671 | 0.1983 +/- 0.0086 | +43.3% | 0.045 / 0.18 | yes | +26.9%, p=7.9e-07 |

*Bonferroni adjustment (multiply by the 4 tests) computed in this analysis; the study itself reported the raw Welch p-values. Figure: `../presentation_results/figures/fig3_shared_vs_specialist.png`.

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

### 4. Cross-family generalisation (unseen families)

**Rule used:** a result counts as zero-shot only if the family was absent from that model's training data.

#### 4a. Two-family models on unseen **ellipses** [MULTI-SEED: 5 training runs]

Trained on: circle + spiral. Excluded: ellipse. Tested on the 150 ellipses of the three-family test split (30 observations). Source: `results/validation/three_shapes/three_shapes_and_ood.json`.

| Model | Seen ellipses in training? | Reconstruction RMSE | Forecast RMSE |
|---|---|---:|---:|
| Two-family models (5 runs), mean +/- SD | **no (zero-shot)** | 0.2096 +/- 0.0040 | 0.3212 +/- 0.0274 |
| Three-family model (1 run) | yes (not zero-shot) | 0.0887 | 0.1850 |
| Linear interpolation (baseline) | n/a | 0.0625 | n/a |
| Hold-last (baseline) | n/a | n/a | 1.1213 |
| Constant velocity (baseline) | n/a | n/a | 2.2082 |

*What happened:* forecasting transfers partly (zero-shot 0.32, far better than hold-last 1.12, but worse than the in-training 0.18); reconstruction does not (0.21, more than three times worse than simply joining the dots, 0.06). Rotation direction is right for every trajectory (geometry numbers in this paragraph are for the seed-42 two-family model only; accuracy 100%), but the model draws ellipses as near-circles: predicted axis ratio 0.92 against true 0.59, correlation 0.18 (the three-family model that saw ellipses: 0.86). An ellipse is a stretched circle, so this is the *easiest* possible unseen family; it counts as partial transfer of the forecast, not successful generalisation.

#### 4b. The three-family model on unseen **figure-eight** and **sinusoid** [SINGLE-SEED: one frozen model]

Trained on: circle + spiral + ellipse. Excluded: figure-eight, sinusoid. Tested on the five-family test split (150 of each unseen family). Source: `results/five_shapes/zeroshot_three_shape_model.json`.

| Family | Status | Recon. RMSE | Linear interp. | Forecast RMSE | Hold-last | Const. velocity |
|---|---|---:|---:|---:|---:|---:|
| circle | seen in training | 0.0622 | 0.0699 | 0.1401 | 1.1370 | 2.3274 |
| spiral | seen in training | 0.0933 | 0.0653 | 0.2908 | 1.2763 | 2.3877 |
| ellipse | seen in training | 0.0888 | 0.0637 | 0.1885 | 1.1200 | 2.0143 |
| figure-eight | **unseen (zero-shot)** | 0.7321 | 0.0917 | 1.1174 | 0.9610 | 2.6176 |
| sinusoid | **unseen (zero-shot)** | 0.3441 | 0.0452 | 0.9734 | 0.6248 | 1.3898 |

*What happened:* on the unseen families the model is **worse than hold-last** at forecasting (figure-eight 1.1174 vs 0.9610; sinusoid 0.9734 vs 0.6248) and **far worse than linear interpolation** at reconstruction (figure-eight 0.7321 vs 0.0917; sinusoid 0.3441 vs 0.0452). It is better than constant velocity, but that baseline is itself very poor on these paths. The model mostly redraws what it has seen (loops and spirals). **This is not successful zero-shot generalisation.**

#### 4c. Five-family model: not zero-shot

The five-family model was trained on all five families, so its results on them (section 2) are **not** generalisation. What it shows is that the same architecture *can* be trained on all five. Compared with the three-family model on the original three families it is worse (paired, identical observations; Holm-adjusted p < 0.01 for every comparison):

| Family | Forecast: 3-family | Forecast: 5-family | Change | Recon: 3-family | Recon: 5-family |
|---|---:|---:|---:|---:|---:|
| circle | 0.1401 | 0.2594 | +85% | 0.0622 | 0.0701 |
| spiral | 0.2908 | 0.4270 | +47% | 0.0933 | 0.1131 |
| ellipse | 0.1885 | 0.2609 | +38% | 0.0888 | 0.0973 |

The five-family model also had more training data and more optimiser steps (33,000 against 19,800), so this cost cannot be blamed on the number of families alone. [SINGLE-SEED]

#### 4d. Did the experiments demonstrate zero-shot generalisation?
**No.** Partial transfer for ellipses (a stretched circle: direction and forecast partly right, shape wrong) [MULTI-SEED]; **failure** for figure-eight and sinusoid [SINGLE-SEED]; and, in the real-world test (section 6.2), failure on real GPS. There is no leave-one-family-out experiment (training on four families, testing on the fifth), so nothing here shows generalisation across *many* families.

### 5. Controlled 2x2 ablation [SINGLE-SEED]

Four models, same data, same training recipe, same training data stream (`data_seed` 42), differing only in latent dimension and GRU (encoder) width; one training run each. Test: the same 750 trajectories with identical observations.

| Model | Latent dimension | GRU width | Parameters |
|---|---:|---:|---:|
| A | 8 | 64 | 24,602 |
| B | 16 | 64 | 27,186 |
| C | 8 | 128 | 63,642 |
| D | 16 | 128 | 67,250 |

#### Overall results (mean per-trajectory RMSE; change relative to A)

| Model | Reconstruction | vs A | Forecast (7 < t <= 10) | vs A | Trajectories with lower forecast error than A |
|---|---:|---:|---:|---:|---:|
| A | 0.0882 | n/a | 0.3998 | n/a | n/a |
| B | 0.0462 | -47.6% | 0.5199 | +30.1% | 35% |
| C | 0.0619 | -29.8% | 0.4248 | +6.3% | 48% |
| D | 0.0412 | -53.2% | 0.5936 | +48.5% | 27% |

Linear interpolation: 0.0672 (reconstruction). Hold-last: 1.0238; constant velocity: 2.1474 (forecast). Figure: `../presentation_results/figures/fig5_ablation_overall.png`.

#### Per family (change vs A; Holm-adjusted p-value of the paired difference, "ns" = not significant)

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

#### Separate comparisons

**A vs B: raising only the latent dimension (8 to 16).** Reconstruction improves in every family (-47.6% overall). Overall forecast error rises +30.1% (Holm p = 1.9e-23); B beats A on 35% of the paired trajectories. Per family the forecast difference is clearly significant only for circle (+41%) and sinusoid (+130%); spiral, ellipse and figure-eight show no significant difference.
**A vs C: raising only the GRU width (64 to 128).** Reconstruction improves in every family (-29.8% overall, a smaller gain than B). Forecast: **no clearly distinguishable overall difference** (+6.3%; mean difference +0.0250, bootstrap 95% interval [+0.0031, +0.0461], median difference +0.0109, better on 48% of trajectories, Holm p = 0.18). By family: significantly worse on sinusoid (+54%), significantly *better* on ellipse (-15%), nothing distinguishable on the others.
**A vs D: raising both.** Reconstruction improves most (-53.2%). Forecast error rises +48.5% (Holm p = 1.2e-50), D beating A on only 27% of trajectories. D has the higher mean on all five families; significantly so on circle, spiral, ellipse and sinusoid, not on figure-eight.

#### The interaction, explained simply
An *interaction* means the effect of one change depends on the other. Overall forecast error: raising the latent size changes it by +0.120 when the GRU is 64 (B minus A) but by +0.169 when the GRU is 128 (D minus C). The extra damage (+0.049, 95% interval [+0.025, +0.073], Holm p = 0.0018) means the two changes **reinforce each other** overall. For reconstruction the interaction is +0.021: the two changes overlap (each helps less once the other is made). The pattern is not the same in every family: for the sinusoid the forecast interaction is -0.28, the opposite sign. The "GRU main effect" of +0.049 is an average over both latent sizes and is carried mainly by D versus B (+0.074), not by C versus A (+0.025), so the two factors should not be described as independently worsening forecasts.

#### Forecast error over time (verified)

Mean per-trajectory RMSE inside time windows (computed in this analysis from the saved predictions; 750 trajectories):

| Time window | A | B | C | D |
|---|---:|---:|---:|---:|
| 7 < t <= 8 | 0.2630 | 0.2174 | 0.1755 | 0.2332 |
| 8 < t <= 9 | 0.3940 | 0.4926 | 0.3649 | 0.5684 |
| 9 < t <= 10 | 0.4818 | 0.7037 | 0.5951 | 0.8101 |

**Verified:** in the first part of the forecast (7 < t <= 8) the larger models B, C and D are *better* than A (lower error). Their error then grows faster and crosses above A at t = 8.09 (B), 8.64 (C) and 7.94 (D) (first time the across-trajectory RMSE exceeds A's, from the saved error-versus-time curves). By 9 < t <= 10 all three are worse than A. At t = 10 the across-trajectory RMSE is A 0.623, B 0.912, C 0.828, D 1.012. So the full-window "worse forecast" comes from the long-horizon part. A shorter forecast window would give the opposite sign for B and D. (Figure: `../presentation_results/figures/fig6_forecast_error_vs_time.png`.)

#### What the ablation does and does not show
**Observed (single seed):** larger encoders reconstruct much better; over the full 7 to 10 window latent size is associated with higher forecast error, GRU width alone is not clearly distinguishable from baseline, the effect appears late in the horizon and is driven mostly by the sinusoid and circle families (for B and D on sinusoid the model is worse than hold-last).
**Not shown:** that capacity universally worsens forecasting; a causal mechanism; that the result holds for other training seeds. For scale: training the *same* architecture twice with different data streams changed mean forecast error by +22.4% (A vs the earlier 8/64 run) and +20.7% (D vs the earlier 16/128 run), and the five-seed two-family study has a forecast coefficient of variation of about 23%. The forecast differences above (+30%, +6%, +48%) are of that same order. (Figure: `../presentation_results/figures/fig7_effects_vs_seed_variation.png`.) A tiny p-value here only says the two *fixed trained models* differ on these test trajectories.
**Possible explanations (untested):** the training objective supervises only the 30 observed points, so extrapolation is not directly trained; larger latent spaces might make errors grow faster; optimisation differences (all cells were still improving at epoch 300). Classical overfitting was not observed (validation loss did not rise; training and validation negative log-likelihood differ by at most 8.4).

### 6. Other completed experiments

#### 6.1 Dynamics-conditioning study [MULTI-SEED: 5 seeds; M2aug 3 seeds]

*Purpose:* the shared model forecasts worse than specialists; does letting the model infer an extra "dynamics code" that controls the ODE close that gap? *Models:* M1 = shared Model D; M2 = shared model with the same extra size but no code (a fair control); M3 = shared model with the dynamics code; M2aug = an ablation. *Data:* circle + spiral, 300 test trajectories. Forecast RMSE, mean +/- SD over seeds (source `results/conditioning/conditioning_study.json`):

| Model | Circle | Spiral |
|---|---:|---:|
| Specialists (3 runs) | 0.087 | 0.198 |
| M1 shared | 0.145 +/- 0.007 | 0.264 +/- 0.036 |
| M2 control, no code | 0.172 +/- 0.096 | 0.270 +/- 0.113 |
| M3 with dynamics code | 0.176 +/- 0.066 | 0.299 +/- 0.068 |
| M2aug ablation (3 runs) | 0.301 +/- 0.108 | 0.435 +/- 0.095 |

*Result:* M3 does **not** forecast better than M1 or M2 (circle +21% and spiral +13% worse than M1 when averaged over seeds; seed-level paired p = 0.35 and 0.38, so not distinguishable across seeds). M3 reconstructs very slightly better (circle -4.3%). The learned code does control radial growth, so it is *used*, but it did not help forecasting. M3 was not tuned separately, which may favour M1. *Supported:* at this scale the hypothesis is not supported. *Note:* two sets of five "shared" runs (this study's M1 and the E3 study) give the same recipe different SDs for circle forecast (0.007 here, 0.037 in E3), a reminder that five-seed SDs are themselves uncertain.

#### 6.2 GeoLife real-world transfer [SINGLE-SEED]

*Purpose:* does what was learned on synthetic shapes help on real GPS? *Data:* 1,000 held-out GPS segments from 97 test users (the models never saw these users; development users only were used for training the GeoLife-trained models). *Task:* interpolate withheld GPS positions with 30 observations per segment. *Metric:* median per-segment error in metres. (GeoLife data and weights are not distributed; only aggregates are reported.)

| Method (30 observations) | Median error, metres [95% CI, user bootstrap] | Ratio to linear interpolation (geometric mean) |
|---|---|---:|
| Linear interpolation | 7.10 [5.8, 8.5] | 1.000 |
| Cubic spline | 5.56 [4.6, 6.9] | 0.799 |
| PCHIP | 5.62 [4.5, 6.7] | 0.778 |
| Frozen synthetic (three-family) model | 126.40 [110.9, 147.9] | 17.278 |
| GeoLife-trained, from scratch | 60.62 [51.6, 71.7] | 8.222 |
| Synthetic, then fine-tuned on GeoLife | 47.11 [42.2, 54.6] | 6.415 |

*Results and conclusions supported.* (1) The **frozen synthetic model does not transfer**: about 17 times worse than linear interpolation. (2) Training on real GPS data helps (about 2 to 3 times better than the frozen model) but both GeoLife-trained models remain 6 to 8 times worse than linear interpolation and beat it on only 0.1% of segments. (3) Forecasts are also worse than the trivial baselines (protocol A medians: scratch 322 m, fine-tuned 371 m, hold-last 229 m, constant velocity 151 m). The likely reason (hypothesis) is that the model can only produce loop-like curves. Figure: `../presentation_results/figures/fig8_geolife_transfer.png`.

**Does synthetic pre-training help? Paired comparison, fine-tuned vs from-scratch (computed in this analysis from the saved per-segment results of the same 1,000 segments; ratio below 1 favours fine-tuning; 95% interval from a bootstrap over the 97 users):**

| Observations per segment | Median error, scratch (m) | Median error, fine-tuned (m) | Geometric-mean ratio fine-tuned / scratch [95% CI] | Fine-tuned better on |
|---|---:|---:|---|---:|
| 10 | 119.6 | 191.7 | 1.65 [1.61, 1.68] | 7% of segments |
| 20 | 75.1 | 94.8 | 1.22 [1.19, 1.25] | 29% of segments |
| 30 | 60.6 | 47.1 | 0.78 [0.76, 0.80] | 81% of segments |
| 40 | 59.8 | 76.7 | 1.29 [1.25, 1.32] | 23% of segments |
| 50 | 60.0 | 115.5 | 1.93 [1.87, 2.00] | 8% of segments |

*Observed:* fine-tuning from synthetic weights is clearly better only at **30 observations**, the number both models were trained with (ratio 0.78); at 10, 20, 40 and 50 observations it is clearly *worse* (ratios 1.65, 1.22, 1.29, 1.93), and the from-scratch model degrades much less away from 30. So **no general benefit of synthetic pre-training is shown**; the one advantage is at the training count. *Possible explanation (untested):* the fine-tuned model is more tied to the observation count it was trained with. Single training run per model, so this could change with other seeds.

#### 6.3 Capacity scaling, first version [SINGLE-SEED]

*Purpose:* does a bigger encoder help? *Models (five-family data):* earlier 8/64 baseline (24,602 parameters), medium 16/128 (67,250), large 32/128 (74,466), one run each, **training data streams not controlled** (this is why the 2x2 ablation was run). *Overall mean RMSE (reconstruction / forecast):* baseline 0.0908 / 0.3265; medium 0.0399 / 0.4919; large 0.0432 / 0.5975. *Supported:* reconstruction improves with size; forecasting is worse for the larger models on the full window, with the sinusoid worst (large: 1.207, worse than hold-last 0.6248). The controlled 2x2 reproduces both directions: for the combined 16/128 change the full-window forecast difference is +48.5% against the controlled baseline A, similar to the +50.6% of the earlier uncontrolled medium run, and the 2x2 shows that it is split unevenly between latent size and GRU width. The earlier 8/64 baseline (0.3265) was better than the controlled rerun A (0.3998), which shows how much a different data stream alone can change a single run.

### 7. The five most important findings

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

### 8. Strength of evidence at a glance

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

---

## Source index: where every number and figure comes from

Generated automatically by `audit/build_results_documents.py`: every value read through an accessor is logged with its file and key path, so this index lists exactly the values used in the documents. Paths are relative to the project root. "key path" = nested keys in the JSON file joined by dots.

### A. Figures

| Figure (in `presentation_results/figures/`) | Drawn from (saved data only) | Existing original figure with similar content |
|---|---|---|
| `fig1_reconstruction_by_family.png` | `results/ablation_2x2/ablation_2x2_results.json` (results.A..D, baselines.linear_interpolation_recon) | `results/ablation_2x2/figures/fig1_cells_by_family.png` |
| `fig2_forecast_by_family.png` | `results/ablation_2x2/ablation_2x2_results.json` (results.A..D, baselines.hold_last_forecast) | `results/ablation_2x2/figures/fig1_cells_by_family.png` |
| `fig3_shared_vs_specialist.png` | `results/validation/shared_vs_separate/shared_vs_separate.json` (circle/spiral.seed_level.*.unified|specialist, values) | `results/validation/shared_vs_separate/shared_vs_separate.png` |
| `fig4_cross_family_generalisation.png` | `results/five_shapes/zeroshot_three_shape_model.json`, `results/five_shapes/compare_three_vs_five.json` | `results/five_shapes/figures/per_family_rmse_three_vs_five.png` |
| `fig5_ablation_overall.png` | `results/ablation_2x2/ablation_2x2_results.json` (results.*.overall) | `results/ablation_2x2/figures/fig2_interaction.png` |
| `fig6_forecast_error_vs_time.png` | `results/ablation_2x2/ablation_2x2_results.json` (error_vs_time) | `results/ablation_2x2/figures/fig3_error_vs_time.png` |
| `fig7_effects_vs_seed_variation.png` | `ablation_2x2_results.json` (results, reproducibility_pairs); `results/validation/seed_study/seed_study.json` (across_seeds.overall.forecast_rmse.values) | none |
| `fig8_geolife_transfer.png` | `results/geolife/main_evaluation/geolife_trained_summary.json` (interpolation.30.*.median, median_ci95_user_bootstrap) | `results/geolife/figures/fig3_method_comparison.png`, `fig9_synthetic_vs_geolife_trained.png` |

### B. Values computed in this analysis from saved results (not read directly from a result file)

- Percent differences and ratios between two saved means (for example "+84.9%" shared vs specialist = shared mean / specialist mean - 1, from `shared_vs_separate.json` seed-level means).
- Bonferroni-adjusted p = min(1, 4 x Welch p), over the four seed-level tests (circle/spiral x reconstruction/forecast) in `shared_vs_separate.json`.
- "Every shared run worse than every specialist run": compares min of the shared per-run values with max of the specialist per-run values in the same file.
- Family rankings (easiest/hardest) across the 7 five-family models: argmin / argmax over `results.<model>.<family>.*.mean` in `ablation_2x2_results.json`.
- Window means (7-8, 8-9, 9-10): mean over the 750 trajectories of per-trajectory RMSE inside the window, recomputed from `results/ablation_2x2/predictions_test.npz` (checked against the JSON in `audit/verify_3_ablation_statistics.py`).
- Crossing times: first query time after t = 7 at which the across-trajectory RMSE of B, C or D exceeds that of A in `ablation_2x2_results.json` (error_vs_time).
- GeoLife fine-tuned vs scratch paired comparison: per segment ratio of `interp_geolife_finetuned_from_three_shapes` to `interp_geolife_scratch` in `results/geolife/main_evaluation/geolife_trained_per_segment.csv` (1,000 segments per observation count); geometric mean of the ratios; 95% interval from a bootstrap over the 97 users (5,000 resamples, seed 0); share of segments where fine-tuned is lower. Per-segment data are not copied anywhere.
- Per-family ranking relative to baselines: ratios of saved means to the saved linear-interpolation / hold-last baselines.

### C. Cross-checks

`audit/verify_5_corrected_documents.py` re-derives the five-family and ablation tables from the raw saved predictions; `audit/verify_6_results_documents.py` checks the tables of these new documents against raw predictions and against the numbers printed in the earlier reports (see the end of this file for the last run).

### D. Numerical claims and their sources

| Claim | Value | Source file :: key path |
|---|---:|---|
| latent_ode_original.overall.recon_rmse.mean | 0.75564 | `results/improved_two_shapes/baseline_comparison.json :: latent_ode_original.overall.recon_rmse.mean` |
| latent_ode_improved.overall.recon_rmse.mean | 0.0556867 | `results/improved_two_shapes/baseline_comparison.json :: latent_ode_improved.overall.recon_rmse.mean` |
| latent_ode_original.overall.forecast_rmse.mean | 0.842842 | `results/improved_two_shapes/baseline_comparison.json :: latent_ode_original.overall.forecast_rmse.mean` |
| latent_ode_improved.overall.forecast_rmse.mean | 0.191941 | `results/improved_two_shapes/baseline_comparison.json :: latent_ode_improved.overall.forecast_rmse.mean` |
| constant_position.overall.recon_rmse.mean | 0.0661935 | `results/improved_two_shapes/baseline_comparison.json :: constant_position.overall.recon_rmse.mean` |
| constant_position.overall.forecast_rmse.mean | 1.20895 | `results/improved_two_shapes/baseline_comparison.json :: constant_position.overall.forecast_rmse.mean` |
| constant_velocity.overall.forecast_rmse.mean | 2.36562 | `results/improved_two_shapes/baseline_comparison.json :: constant_velocity.overall.forecast_rmse.mean` |
| across_seeds.overall.recon_rmse.mean | 0.0599329 | `results/validation/seed_study/seed_study.json :: across_seeds.overall.recon_rmse.mean` |
| across_seeds.overall.recon_rmse.std | 0.00685212 | `results/validation/seed_study/seed_study.json :: across_seeds.overall.recon_rmse.std` |
| across_seeds.overall.forecast_rmse.mean | 0.219355 | `results/validation/seed_study/seed_study.json :: across_seeds.overall.forecast_rmse.mean` |
| across_seeds.overall.forecast_rmse.std | 0.0501823 | `results/validation/seed_study/seed_study.json :: across_seeds.overall.forecast_rmse.std` |
| across_seeds.overall.forecast_rmse.ci95_low | 0.157055 | `results/validation/seed_study/seed_study.json :: across_seeds.overall.forecast_rmse.ci95_low` |
| across_seeds.overall.forecast_rmse.ci95_high | 0.281654 | `results/validation/seed_study/seed_study.json :: across_seeds.overall.forecast_rmse.ci95_high` |
| C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_pred_mean | 0.923604 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_pred_mean` |
| C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_true_mean | 0.591632 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_true_mean` |
| baselines.linear_interpolation_recon.circle.mean | 0.0699414 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.linear_interpolation_recon.circle.mean` |
| three_shape_model.circle.recon_rmse.mean | 0.0621759 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.circle.recon_rmse.mean` |
| results.orig_baseline.circle.recon_rmse.mean | 0.0700578 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.circle.recon_rmse.mean` |
| results.A.circle.recon_rmse.mean | 0.0606554 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.circle.recon_rmse.mean` |
| results.B.circle.recon_rmse.mean | 0.0415857 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.circle.recon_rmse.mean` |
| results.C.circle.recon_rmse.mean | 0.0513546 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.circle.recon_rmse.mean` |
| results.D.circle.recon_rmse.mean | 0.0369487 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.circle.recon_rmse.mean` |
| baselines.linear_interpolation_recon.spiral.mean | 0.0653297 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.linear_interpolation_recon.spiral.mean` |
| three_shape_model.spiral.recon_rmse.mean | 0.0932613 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.spiral.recon_rmse.mean` |
| results.orig_baseline.spiral.recon_rmse.mean | 0.113143 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.spiral.recon_rmse.mean` |
| results.A.spiral.recon_rmse.mean | 0.0883345 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.spiral.recon_rmse.mean` |
| results.B.spiral.recon_rmse.mean | 0.0465585 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.spiral.recon_rmse.mean` |
| results.C.spiral.recon_rmse.mean | 0.0710906 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.spiral.recon_rmse.mean` |
| results.D.spiral.recon_rmse.mean | 0.0434099 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.spiral.recon_rmse.mean` |
| baselines.linear_interpolation_recon.ellipse.mean | 0.0637371 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.linear_interpolation_recon.ellipse.mean` |
| three_shape_model.ellipse.recon_rmse.mean | 0.0887923 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.ellipse.recon_rmse.mean` |
| results.orig_baseline.ellipse.recon_rmse.mean | 0.0973143 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.ellipse.recon_rmse.mean` |
| results.A.ellipse.recon_rmse.mean | 0.0945919 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.ellipse.recon_rmse.mean` |
| results.B.ellipse.recon_rmse.mean | 0.0436283 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.ellipse.recon_rmse.mean` |
| results.C.ellipse.recon_rmse.mean | 0.0667816 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.ellipse.recon_rmse.mean` |
| results.D.ellipse.recon_rmse.mean | 0.0407961 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.ellipse.recon_rmse.mean` |
| baselines.linear_interpolation_recon.figure_eight.mean | 0.091708 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.linear_interpolation_recon.figure_eight.mean` |
| three_shape_model.figure_eight.recon_rmse.mean | 0.732054 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.figure_eight.recon_rmse.mean` |
| results.orig_baseline.figure_eight.recon_rmse.mean | 0.10858 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.figure_eight.recon_rmse.mean` |
| results.A.figure_eight.recon_rmse.mean | 0.136442 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.figure_eight.recon_rmse.mean` |
| results.B.figure_eight.recon_rmse.mean | 0.0654012 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.figure_eight.recon_rmse.mean` |
| results.C.figure_eight.recon_rmse.mean | 0.0753711 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.figure_eight.recon_rmse.mean` |
| results.D.figure_eight.recon_rmse.mean | 0.0567664 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.figure_eight.recon_rmse.mean` |
| baselines.linear_interpolation_recon.sinusoid.mean | 0.0452147 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.linear_interpolation_recon.sinusoid.mean` |
| three_shape_model.sinusoid.recon_rmse.mean | 0.344138 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.sinusoid.recon_rmse.mean` |
| results.orig_baseline.sinusoid.recon_rmse.mean | 0.064719 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.sinusoid.recon_rmse.mean` |
| results.A.sinusoid.recon_rmse.mean | 0.0607323 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.sinusoid.recon_rmse.mean` |
| results.B.sinusoid.recon_rmse.mean | 0.0335856 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.sinusoid.recon_rmse.mean` |
| results.C.sinusoid.recon_rmse.mean | 0.0450182 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.sinusoid.recon_rmse.mean` |
| results.D.sinusoid.recon_rmse.mean | 0.0282045 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.sinusoid.recon_rmse.mean` |
| baselines.linear_interpolation_recon.overall.mean | 0.0671862 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.linear_interpolation_recon.overall.mean` |
| three_shape_model.overall.recon_rmse.mean | 0.264084 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.overall.recon_rmse.mean` |
| results.orig_baseline.overall.recon_rmse.mean | 0.0907628 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.overall.recon_rmse.mean` |
| results.A.overall.recon_rmse.mean | 0.0881513 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.overall.recon_rmse.mean` |
| results.B.overall.recon_rmse.mean | 0.0461518 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.overall.recon_rmse.mean` |
| results.C.overall.recon_rmse.mean | 0.0619232 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.overall.recon_rmse.mean` |
| results.D.overall.recon_rmse.mean | 0.0412251 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.overall.recon_rmse.mean` |
| baselines.hold_last_forecast.circle.mean | 1.13701 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.hold_last_forecast.circle.mean` |
| baselines.const_velocity_forecast.circle.mean | 2.32739 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.const_velocity_forecast.circle.mean` |
| three_shape_model.circle.forecast_rmse.mean | 0.140127 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.circle.forecast_rmse.mean` |
| results.orig_baseline.circle.forecast_rmse.mean | 0.25942 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.circle.forecast_rmse.mean` |
| results.A.circle.forecast_rmse.mean | 0.267574 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.circle.forecast_rmse.mean` |
| results.B.circle.forecast_rmse.mean | 0.37823 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.circle.forecast_rmse.mean` |
| results.C.circle.forecast_rmse.mean | 0.288246 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.circle.forecast_rmse.mean` |
| results.D.circle.forecast_rmse.mean | 0.518226 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.circle.forecast_rmse.mean` |
| baselines.hold_last_forecast.spiral.mean | 1.27631 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.hold_last_forecast.spiral.mean` |
| baselines.const_velocity_forecast.spiral.mean | 2.38767 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.const_velocity_forecast.spiral.mean` |
| three_shape_model.spiral.forecast_rmse.mean | 0.290796 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.spiral.forecast_rmse.mean` |
| results.orig_baseline.spiral.forecast_rmse.mean | 0.427031 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.spiral.forecast_rmse.mean` |
| results.A.spiral.forecast_rmse.mean | 0.49147 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.spiral.forecast_rmse.mean` |
| results.B.spiral.forecast_rmse.mean | 0.535259 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.spiral.forecast_rmse.mean` |
| results.C.spiral.forecast_rmse.mean | 0.49866 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.spiral.forecast_rmse.mean` |
| results.D.spiral.forecast_rmse.mean | 0.660501 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.spiral.forecast_rmse.mean` |
| baselines.hold_last_forecast.ellipse.mean | 1.12001 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.hold_last_forecast.ellipse.mean` |
| baselines.const_velocity_forecast.ellipse.mean | 2.0143 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.const_velocity_forecast.ellipse.mean` |
| three_shape_model.ellipse.forecast_rmse.mean | 0.188535 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.ellipse.forecast_rmse.mean` |
| results.orig_baseline.ellipse.forecast_rmse.mean | 0.26087 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.ellipse.forecast_rmse.mean` |
| results.A.ellipse.forecast_rmse.mean | 0.391559 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.ellipse.forecast_rmse.mean` |
| results.B.ellipse.forecast_rmse.mean | 0.374921 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.ellipse.forecast_rmse.mean` |
| results.C.ellipse.forecast_rmse.mean | 0.331235 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.ellipse.forecast_rmse.mean` |
| results.D.ellipse.forecast_rmse.mean | 0.524001 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.ellipse.forecast_rmse.mean` |
| baselines.hold_last_forecast.figure_eight.mean | 0.960963 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.hold_last_forecast.figure_eight.mean` |
| baselines.const_velocity_forecast.figure_eight.mean | 2.61761 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.const_velocity_forecast.figure_eight.mean` |
| three_shape_model.figure_eight.forecast_rmse.mean | 1.11737 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.figure_eight.forecast_rmse.mean` |
| results.orig_baseline.figure_eight.forecast_rmse.mean | 0.299173 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.figure_eight.forecast_rmse.mean` |
| results.A.figure_eight.forecast_rmse.mean | 0.473974 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.figure_eight.forecast_rmse.mean` |
| results.B.figure_eight.forecast_rmse.mean | 0.45109 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.figure_eight.forecast_rmse.mean` |
| results.C.figure_eight.forecast_rmse.mean | 0.43099 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.figure_eight.forecast_rmse.mean` |
| results.D.figure_eight.forecast_rmse.mean | 0.484519 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.figure_eight.forecast_rmse.mean` |
| baselines.hold_last_forecast.sinusoid.mean | 0.624795 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.hold_last_forecast.sinusoid.mean` |
| baselines.const_velocity_forecast.sinusoid.mean | 1.3898 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.const_velocity_forecast.sinusoid.mean` |
| three_shape_model.sinusoid.forecast_rmse.mean | 0.973432 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.sinusoid.forecast_rmse.mean` |
| results.orig_baseline.sinusoid.forecast_rmse.mean | 0.386051 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.sinusoid.forecast_rmse.mean` |
| results.A.sinusoid.forecast_rmse.mean | 0.374383 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.sinusoid.forecast_rmse.mean` |
| results.B.sinusoid.forecast_rmse.mean | 0.860213 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.sinusoid.forecast_rmse.mean` |
| results.C.sinusoid.forecast_rmse.mean | 0.575063 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.sinusoid.forecast_rmse.mean` |
| results.D.sinusoid.forecast_rmse.mean | 0.780814 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.sinusoid.forecast_rmse.mean` |
| baselines.hold_last_forecast.overall.mean | 1.02382 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.hold_last_forecast.overall.mean` |
| baselines.const_velocity_forecast.overall.mean | 2.14735 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.const_velocity_forecast.overall.mean` |
| three_shape_model.overall.forecast_rmse.mean | 0.542051 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.overall.forecast_rmse.mean` |
| results.orig_baseline.overall.forecast_rmse.mean | 0.326509 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.overall.forecast_rmse.mean` |
| results.A.overall.forecast_rmse.mean | 0.399792 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.overall.forecast_rmse.mean` |
| results.B.overall.forecast_rmse.mean | 0.519943 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.overall.forecast_rmse.mean` |
| results.C.overall.forecast_rmse.mean | 0.424839 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.overall.forecast_rmse.mean` |
| results.D.overall.forecast_rmse.mean | 0.593612 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.overall.forecast_rmse.mean` |
| circle.seed_level.recon_rmse.unified.mean | 0.0531726 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.seed_level.recon_rmse.unified.mean` |
| circle.seed_level.recon_rmse.specialist.mean | 0.0457315 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.seed_level.recon_rmse.specialist.mean` |
| circle.seed_level.recon_rmse.welch_t_p | 0.0473354 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.seed_level.recon_rmse.welch_t_p` |
| circle.paired_seed42.recon_rmse.wilcoxon_p | 0.000403706 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.paired_seed42.recon_rmse.wilcoxon_p` |
| circle.paired_seed42.recon_rmse.relative_diff_pct | 12.2864 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.paired_seed42.recon_rmse.relative_diff_pct` |
| circle.seed_level.forecast_rmse.unified.mean | 0.160566 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.seed_level.forecast_rmse.unified.mean` |
| circle.seed_level.forecast_rmse.specialist.mean | 0.0868448 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.seed_level.forecast_rmse.specialist.mean` |
| circle.seed_level.forecast_rmse.welch_t_p | 0.0098777 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.seed_level.forecast_rmse.welch_t_p` |
| circle.paired_seed42.forecast_rmse.wilcoxon_p | 4.7879e-15 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.paired_seed42.forecast_rmse.wilcoxon_p` |
| circle.paired_seed42.forecast_rmse.relative_diff_pct | 75.6165 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.paired_seed42.forecast_rmse.relative_diff_pct` |
| spiral.seed_level.recon_rmse.unified.mean | 0.0680014 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.seed_level.recon_rmse.unified.mean` |
| spiral.seed_level.recon_rmse.specialist.mean | 0.0632334 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.seed_level.recon_rmse.specialist.mean` |
| spiral.seed_level.recon_rmse.welch_t_p | 0.356397 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.seed_level.recon_rmse.welch_t_p` |
| spiral.paired_seed42.recon_rmse.wilcoxon_p | 0.638356 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.paired_seed42.recon_rmse.wilcoxon_p` |
| spiral.paired_seed42.recon_rmse.relative_diff_pct | -2.80606 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.paired_seed42.recon_rmse.relative_diff_pct` |
| spiral.seed_level.forecast_rmse.unified.mean | 0.284081 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.seed_level.forecast_rmse.unified.mean` |
| spiral.seed_level.forecast_rmse.specialist.mean | 0.198305 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.seed_level.forecast_rmse.specialist.mean` |
| spiral.seed_level.forecast_rmse.welch_t_p | 0.0449903 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.seed_level.forecast_rmse.welch_t_p` |
| spiral.paired_seed42.forecast_rmse.wilcoxon_p | 7.92018e-07 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.paired_seed42.forecast_rmse.wilcoxon_p` |
| spiral.paired_seed42.forecast_rmse.relative_diff_pct | 26.8695 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.paired_seed42.forecast_rmse.relative_diff_pct` |
| B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.three_shape | 0.135818 | `results/validation/three_shapes/three_shapes_and_ood.json :: B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.three_shape` |
| B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.two_shape_seed42 | 0.141974 | `results/validation/three_shapes/three_shapes_and_ood.json :: B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.two_shape_seed42` |
| B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.two_shape_across_seeds.mean | 0.157088 | `results/validation/three_shapes/three_shapes_and_ood.json :: B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.two_shape_across_seeds.mean` |
| B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.two_shape_across_seeds.std | 0.0373541 | `results/validation/three_shapes/three_shapes_and_ood.json :: B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.two_shape_across_seeds.std` |
| C_out_of_distribution_ellipses.three_shape_model_on_same_ellipses (in-distribution reference).recon_rmse.mean | 0.0887434 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.three_shape_model_on_same_ellipses (in-distribution reference).recon_rmse.mean` |
| C_out_of_distribution_ellipses.three_shape_model_on_same_ellipses (in-distribution reference).forecast_rmse.mean | 0.184998 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.three_shape_model_on_same_ellipses (in-distribution reference).forecast_rmse.mean` |
| C_out_of_distribution_ellipses.const_pos_on_same_ellipses.recon_rmse.mean | 0.0625047 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.const_pos_on_same_ellipses.recon_rmse.mean` |
| C_out_of_distribution_ellipses.const_pos_on_same_ellipses.forecast_rmse.mean | 1.12126 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.const_pos_on_same_ellipses.forecast_rmse.mean` |
| C_out_of_distribution_ellipses.const_vel_on_same_ellipses.forecast_rmse.mean | 2.20821 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.const_vel_on_same_ellipses.forecast_rmse.mean` |
| C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).forecast_rotation_direction_accuracy | 1 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).forecast_rotation_direction_accuracy` |
| C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_corr | 0.184563 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_corr` |
| C_out_of_distribution_ellipses.ellipse_geometry_three_shape (ID).axis_ratio_corr | 0.863904 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.ellipse_geometry_three_shape (ID).axis_ratio_corr` |
| five_shape_model.circle.forecast_rmse.mean | 0.25942 | `results/five_shapes/compare_three_vs_five.json :: five_shape_model.circle.forecast_rmse.mean` |
| paired.circle/forecast_rmse.relative_change_of_mean | 0.851324 | `results/five_shapes/compare_three_vs_five.json :: paired.circle/forecast_rmse.relative_change_of_mean` |
| five_shape_model.circle.recon_rmse.mean | 0.0700578 | `results/five_shapes/compare_three_vs_five.json :: five_shape_model.circle.recon_rmse.mean` |
| five_shape_model.spiral.forecast_rmse.mean | 0.427031 | `results/five_shapes/compare_three_vs_five.json :: five_shape_model.spiral.forecast_rmse.mean` |
| paired.spiral/forecast_rmse.relative_change_of_mean | 0.468493 | `results/five_shapes/compare_three_vs_five.json :: paired.spiral/forecast_rmse.relative_change_of_mean` |
| five_shape_model.spiral.recon_rmse.mean | 0.113143 | `results/five_shapes/compare_three_vs_five.json :: five_shape_model.spiral.recon_rmse.mean` |
| five_shape_model.ellipse.forecast_rmse.mean | 0.26087 | `results/five_shapes/compare_three_vs_five.json :: five_shape_model.ellipse.forecast_rmse.mean` |
| paired.ellipse/forecast_rmse.relative_change_of_mean | 0.383672 | `results/five_shapes/compare_three_vs_five.json :: paired.ellipse/forecast_rmse.relative_change_of_mean` |
| five_shape_model.ellipse.recon_rmse.mean | 0.0973143 | `results/five_shapes/compare_three_vs_five.json :: five_shape_model.ellipse.recon_rmse.mean` |
| models.A.optimizer_steps | 33000 | `results/ablation_2x2/ablation_2x2_results.json :: models.A.optimizer_steps` |
| models.A.latent_dim | 8 | `results/ablation_2x2/ablation_2x2_results.json :: models.A.latent_dim` |
| models.A.encoder_hidden | 64 | `results/ablation_2x2/ablation_2x2_results.json :: models.A.encoder_hidden` |
| models.A.n_parameters | 24602 | `results/ablation_2x2/ablation_2x2_results.json :: models.A.n_parameters` |
| models.B.latent_dim | 16 | `results/ablation_2x2/ablation_2x2_results.json :: models.B.latent_dim` |
| models.B.encoder_hidden | 64 | `results/ablation_2x2/ablation_2x2_results.json :: models.B.encoder_hidden` |
| models.B.n_parameters | 27186 | `results/ablation_2x2/ablation_2x2_results.json :: models.B.n_parameters` |
| models.C.latent_dim | 8 | `results/ablation_2x2/ablation_2x2_results.json :: models.C.latent_dim` |
| models.C.encoder_hidden | 128 | `results/ablation_2x2/ablation_2x2_results.json :: models.C.encoder_hidden` |
| models.C.n_parameters | 63642 | `results/ablation_2x2/ablation_2x2_results.json :: models.C.n_parameters` |
| models.D.latent_dim | 16 | `results/ablation_2x2/ablation_2x2_results.json :: models.D.latent_dim` |
| models.D.encoder_hidden | 128 | `results/ablation_2x2/ablation_2x2_results.json :: models.D.encoder_hidden` |
| models.D.n_parameters | 67250 | `results/ablation_2x2/ablation_2x2_results.json :: models.D.n_parameters` |
| fraction_of_trajectories_better_than_A.B.overall.forecast_rmse | 0.350667 | `results/ablation_2x2/ablation_2x2_results.json :: fraction_of_trajectories_better_than_A.B.overall.forecast_rmse` |
| fraction_of_trajectories_better_than_A.C.overall.forecast_rmse | 0.477333 | `results/ablation_2x2/ablation_2x2_results.json :: fraction_of_trajectories_better_than_A.C.overall.forecast_rmse` |
| fraction_of_trajectories_better_than_A.D.overall.forecast_rmse | 0.269333 | `results/ablation_2x2/ablation_2x2_results.json :: fraction_of_trajectories_better_than_A.D.overall.forecast_rmse` |
| factorial.overall/forecast_rmse/interaction ((D-C) - (B-A)).mean | 0.0486225 | `results/ablation_2x2/ablation_2x2_results.json :: factorial.overall/forecast_rmse/interaction ((D-C) - (B-A)).mean` |
| reproducibility_pairs.A_vs_orig_baseline.overall.forecast_rmse.relative_difference | 0.224444 | `results/ablation_2x2/ablation_2x2_results.json :: reproducibility_pairs.A_vs_orig_baseline.overall.forecast_rmse.relative_difference` |
| reproducibility_pairs.D_vs_orig_medium.overall.forecast_rmse.relative_difference | 0.206856 | `results/ablation_2x2/ablation_2x2_results.json :: reproducibility_pairs.D_vs_orig_medium.overall.forecast_rmse.relative_difference` |
| gap_to_specialists.circle.specialist_forecast_rmse | 0.0868448 | `results/conditioning/conditioning_study.json :: gap_to_specialists.circle.specialist_forecast_rmse` |
| gap_to_specialists.spiral.specialist_forecast_rmse | 0.198305 | `results/conditioning/conditioning_study.json :: gap_to_specialists.spiral.specialist_forecast_rmse` |
| seed_level.M1.circle.forecast_rmse.mean | 0.145087 | `results/conditioning/conditioning_study.json :: seed_level.M1.circle.forecast_rmse.mean` |
| seed_level.M1.circle.forecast_rmse.std | 0.0070338 | `results/conditioning/conditioning_study.json :: seed_level.M1.circle.forecast_rmse.std` |
| seed_level.M1.spiral.forecast_rmse.mean | 0.263523 | `results/conditioning/conditioning_study.json :: seed_level.M1.spiral.forecast_rmse.mean` |
| seed_level.M1.spiral.forecast_rmse.std | 0.0356318 | `results/conditioning/conditioning_study.json :: seed_level.M1.spiral.forecast_rmse.std` |
| seed_level.M2.circle.forecast_rmse.mean | 0.171908 | `results/conditioning/conditioning_study.json :: seed_level.M2.circle.forecast_rmse.mean` |
| seed_level.M2.circle.forecast_rmse.std | 0.0959773 | `results/conditioning/conditioning_study.json :: seed_level.M2.circle.forecast_rmse.std` |
| seed_level.M2.spiral.forecast_rmse.mean | 0.269946 | `results/conditioning/conditioning_study.json :: seed_level.M2.spiral.forecast_rmse.mean` |
| seed_level.M2.spiral.forecast_rmse.std | 0.112926 | `results/conditioning/conditioning_study.json :: seed_level.M2.spiral.forecast_rmse.std` |
| seed_level.M3.circle.forecast_rmse.mean | 0.175902 | `results/conditioning/conditioning_study.json :: seed_level.M3.circle.forecast_rmse.mean` |
| seed_level.M3.circle.forecast_rmse.std | 0.0664882 | `results/conditioning/conditioning_study.json :: seed_level.M3.circle.forecast_rmse.std` |
| seed_level.M3.spiral.forecast_rmse.mean | 0.298538 | `results/conditioning/conditioning_study.json :: seed_level.M3.spiral.forecast_rmse.mean` |
| seed_level.M3.spiral.forecast_rmse.std | 0.0680876 | `results/conditioning/conditioning_study.json :: seed_level.M3.spiral.forecast_rmse.std` |
| seed_level.M2aug.circle.forecast_rmse.mean | 0.300634 | `results/conditioning/conditioning_study.json :: seed_level.M2aug.circle.forecast_rmse.mean` |
| seed_level.M2aug.circle.forecast_rmse.std | 0.107892 | `results/conditioning/conditioning_study.json :: seed_level.M2aug.circle.forecast_rmse.std` |
| seed_level.M2aug.spiral.forecast_rmse.mean | 0.434983 | `results/conditioning/conditioning_study.json :: seed_level.M2aug.spiral.forecast_rmse.mean` |
| seed_level.M2aug.spiral.forecast_rmse.std | 0.0948737 | `results/conditioning/conditioning_study.json :: seed_level.M2aug.spiral.forecast_rmse.std` |
| interpolation.30.linear.median_ci95_user_bootstrap.0 | 5.79753 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.linear.median_ci95_user_bootstrap.0` |
| interpolation.30.linear.median_ci95_user_bootstrap.1 | 8.47659 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.linear.median_ci95_user_bootstrap.1` |
| interpolation.30.linear.median | 7.09944 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.linear.median` |
| interpolation.30.linear.geometric_mean_ratio_vs_linear | 1 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.linear.geometric_mean_ratio_vs_linear` |
| interpolation.30.cubic.median_ci95_user_bootstrap.0 | 4.64313 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.cubic.median_ci95_user_bootstrap.0` |
| interpolation.30.cubic.median_ci95_user_bootstrap.1 | 6.90234 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.cubic.median_ci95_user_bootstrap.1` |
| interpolation.30.cubic.median | 5.56143 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.cubic.median` |
| interpolation.30.cubic.geometric_mean_ratio_vs_linear | 0.799206 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.cubic.geometric_mean_ratio_vs_linear` |
| interpolation.30.pchip.median_ci95_user_bootstrap.0 | 4.53771 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.pchip.median_ci95_user_bootstrap.0` |
| interpolation.30.pchip.median_ci95_user_bootstrap.1 | 6.68928 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.pchip.median_ci95_user_bootstrap.1` |
| interpolation.30.pchip.median | 5.61723 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.pchip.median` |
| interpolation.30.pchip.geometric_mean_ratio_vs_linear | 0.777574 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.pchip.geometric_mean_ratio_vs_linear` |
| interpolation.30.frozen_three_shapes.median_ci95_user_bootstrap.0 | 110.938 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.frozen_three_shapes.median_ci95_user_bootstrap.0` |
| interpolation.30.frozen_three_shapes.median_ci95_user_bootstrap.1 | 147.94 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.frozen_three_shapes.median_ci95_user_bootstrap.1` |
| interpolation.30.frozen_three_shapes.median | 126.403 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.frozen_three_shapes.median` |
| interpolation.30.frozen_three_shapes.geometric_mean_ratio_vs_linear | 17.2783 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.frozen_three_shapes.geometric_mean_ratio_vs_linear` |
| interpolation.30.geolife_scratch.median_ci95_user_bootstrap.0 | 51.5761 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.geolife_scratch.median_ci95_user_bootstrap.0` |
| interpolation.30.geolife_scratch.median_ci95_user_bootstrap.1 | 71.6603 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.geolife_scratch.median_ci95_user_bootstrap.1` |
| interpolation.30.geolife_scratch.median | 60.6185 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.geolife_scratch.median` |
| interpolation.30.geolife_scratch.geometric_mean_ratio_vs_linear | 8.22152 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.geolife_scratch.geometric_mean_ratio_vs_linear` |
| interpolation.30.geolife_finetuned_from_three_shapes.median_ci95_user_bootstrap.0 | 42.2398 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.geolife_finetuned_from_three_shapes.median_ci95_user_bootstrap.0` |
| interpolation.30.geolife_finetuned_from_three_shapes.median_ci95_user_bootstrap.1 | 54.6273 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.geolife_finetuned_from_three_shapes.median_ci95_user_bootstrap.1` |
| interpolation.30.geolife_finetuned_from_three_shapes.median | 47.1083 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.geolife_finetuned_from_three_shapes.median` |
| interpolation.30.geolife_finetuned_from_three_shapes.geometric_mean_ratio_vs_linear | 6.41475 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.geolife_finetuned_from_three_shapes.geometric_mean_ratio_vs_linear` |
| forecast_n30.A.geolife_scratch.median | 322.255 | `results/geolife/main_evaluation/geolife_trained_summary.json :: forecast_n30.A.geolife_scratch.median` |
| forecast_n30.A.geolife_finetuned_from_three_shapes.median | 370.578 | `results/geolife/main_evaluation/geolife_trained_summary.json :: forecast_n30.A.geolife_finetuned_from_three_shapes.median` |
| forecast_n30.A.hold_last.median | 228.92 | `results/geolife/main_evaluation/geolife_trained_summary.json :: forecast_n30.A.hold_last.median` |
| forecast_n30.A.const_velocity.median | 151.434 | `results/geolife/main_evaluation/geolife_trained_summary.json :: forecast_n30.A.const_velocity.median` |
| models.baseline.n_parameters | 24602 | `results/capacity_scaling/capacity_scaling_results.json :: models.baseline.n_parameters` |
| models.medium.n_parameters | 67250 | `results/capacity_scaling/capacity_scaling_results.json :: models.medium.n_parameters` |
| models.large.n_parameters | 74466 | `results/capacity_scaling/capacity_scaling_results.json :: models.large.n_parameters` |
| results.baseline.overall.recon_rmse.mean | 0.0907628 | `results/capacity_scaling/capacity_scaling_results.json :: results.baseline.overall.recon_rmse.mean` |
| results.baseline.overall.forecast_rmse.mean | 0.326509 | `results/capacity_scaling/capacity_scaling_results.json :: results.baseline.overall.forecast_rmse.mean` |
| results.medium.overall.recon_rmse.mean | 0.0398668 | `results/capacity_scaling/capacity_scaling_results.json :: results.medium.overall.recon_rmse.mean` |
| results.medium.overall.forecast_rmse.mean | 0.491866 | `results/capacity_scaling/capacity_scaling_results.json :: results.medium.overall.forecast_rmse.mean` |
| results.large.overall.recon_rmse.mean | 0.0432374 | `results/capacity_scaling/capacity_scaling_results.json :: results.large.overall.recon_rmse.mean` |
| results.large.overall.forecast_rmse.mean | 0.59749 | `results/capacity_scaling/capacity_scaling_results.json :: results.large.overall.forecast_rmse.mean` |
| results.large.sinusoid.forecast_rmse.mean | 1.20741 | `results/capacity_scaling/capacity_scaling_results.json :: results.large.sinusoid.forecast_rmse.mean` |
| five_shape_model.figure_eight.forecast_rmse.mean | 0.299173 | `results/five_shapes/compare_three_vs_five.json :: five_shape_model.figure_eight.forecast_rmse.mean` |
| five_shape_model.sinusoid.forecast_rmse.mean | 0.386051 | `results/five_shapes/compare_three_vs_five.json :: five_shape_model.sinusoid.forecast_rmse.mean` |
