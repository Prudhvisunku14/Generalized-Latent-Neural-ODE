# Controlled 2x2 ablation: latent dimension x GRU encoder width (five-family dataset)

> **Status: corrected version (2026-11-21).** This file replaces an earlier version that contained wrong reconstruction numbers and several unsupported
> interpretations. The uncorrected original is preserved byte-for-byte as `ABLATION_2X2_REPORT.original_uncorrected.md` (read-only). The list of
> corrections is in Appendix A. Every number below is read from `ablation_2x2_results.json`, `predictions_test.npz`, `checkpoint_selection_audit.json`,
> the per-epoch history files and `results/validation/seed_study/seed_study.json` by `audit/build_corrected_ablation_report.py`; the table values were then
> re-checked against an independent recomputation (`audit/verify_5_corrected_documents.py`).

## 1. What this report is, and is not

- **The project is an adapted re-implementation and experimental extension of Latent Neural ODEs** (Chen et al., 2018): same model family, different data, solver settings and objective details. It is **not an exact replication** of the paper, and the paper's numbers were not reproduced.
- **It has not produced an improved forecasting model.** No method to improve forecasting has been developed or tested. This ablation studies how model size affects an existing model.
- **Every result here comes from ONE training run per cell (seed 42). Nothing in this report has been confirmed across training seeds.** The only multi-seed evidence in the project (5 seeds, two-shape data, a different experiment) shows that the same architecture varies a lot between seeds (section 8).
- All findings below are therefore **exploratory, single-seed observations**. Statements about causes are hypotheses, not conclusions.

## 2. Design and what was verified

Question: how do latent dimension and GRU encoder width, separately and together, affect reconstruction and forecasting of the same model on the five-family data?

| Cell | Latent | GRU | Parameters | Selected epoch | Validation -ELBO | Training steps |
|---|---:|---:|---:|---:|---:|---:|
| A (latent 8, GRU 64) | 8 | 64 | 24,602 | 294 | 46.80 | 33,000 |
| B (latent 16, GRU 64) | 16 | 64 | 27,186 | 299 | -22.97 | 33,000 |
| C (latent 8, GRU 128) | 8 | 128 | 63,642 | 299 | -5.47 | 33,000 |
| D (latent 16, GRU 128) | 16 | 128 | 67,250 | 296 | -26.55 | 33,000 |

**Verified controls** (see `PROJECT_COMPLETION_AUDIT.md`): all four runs completed 300 epochs with no early stopping and no non-finite values; training arguments differ only in latent size and GRU width; seed 42 and `data_seed` 42 for every cell (identical batch order and identical training observation subsets, so these no longer differ between architectures); 110 optimiser steps per epoch, 33,000 in total; identical validation observation draws; identical test observations (750 trajectories, 150 per family) in a fresh evaluation; checkpoints chosen by validation -ELBO only, never by test results.

**Not controlled:** initial weights and the reparameterisation noise differ between architectures (the networks have different shapes); there is one training run per cell; the ODE field and decoder widths (64 units, 2 layers) are held fixed, so most added parameters are in the encoder.

## 3. Overall results (mean per-trajectory RMSE over the 750 test trajectories)

Mean (median; SD) of per-trajectory RMSE in data units. "Better than A" is the share of the same 750 trajectories on which the cell has lower error than cell A.

| Cell | Reconstruction | vs A | better than A | Forecast (7 < t <= 10) | vs A | better than A |
|---|---|---:|---:|---|---:|---:|
| A (latent 8, GRU 64) | 0.0882 (0.0738; 0.0527) | n/a | n/a | 0.3998 (0.3500; 0.2425) | n/a | n/a |
| B (latent 16, GRU 64) | 0.0462 (0.0402; 0.0250) | -47.6% | 95% | 0.5199 (0.4523; 0.3051) | +30.1% | 35% |
| C (latent 8, GRU 128) | 0.0619 (0.0554; 0.0288) | -29.8% | 77% | 0.4248 (0.3599; 0.2759) | +6.3% | 48% |
| D (latent 16, GRU 128) | 0.0412 (0.0361; 0.0208) | -53.2% | 97% | 0.5936 (0.5529; 0.3006) | +48.5% | 27% |

Reference baselines on the same trajectories: linear interpolation (reconstruction) 0.0672; hold-last (forecast) 1.0238; constant velocity (forecast) 2.1474. Cell A's reconstruction (0.0882) is worse than linear interpolation; B, C and D are better. Overall, all four cells forecast far better than both naive baselines. **On the sinusoid family, however, B (0.860) and D (0.781) are worse than the hold-last baseline (0.625).**

### Per family: forecast RMSE

Mean per-trajectory forecast RMSE; for B, C and D the change relative to A and the Holm-adjusted Wilcoxon p-value for the paired difference (72 cell-contrast tests in total). "ns" = not significant after correction.

| Family | A | B (vs A) | C (vs A) | D (vs A) |
|---|---:|---|---|---|
| circle | 0.2676 | 0.3782 (+41.4%; p=3.4e-10) | 0.2882 (+7.7%; ns) | 0.5182 (+93.7%; p=2e-17) |
| spiral | 0.4915 | 0.5353 (+8.9%; ns) | 0.4987 (+1.5%; ns) | 0.6605 (+34.4%; p=0.001) |
| ellipse | 0.3916 | 0.3749 (-4.2%; ns) | 0.3312 (-15.4%; p=0.013) | 0.5240 (+33.8%; p=6.7e-09) |
| figure_eight | 0.4740 | 0.4511 (-4.8%; ns) | 0.4310 (-9.1%; ns) | 0.4845 (+2.2%; ns) |
| sinusoid | 0.3744 | 0.8602 (+129.8%; p=1.5e-24) | 0.5751 (+53.6%; p=4.9e-18) | 0.7808 (+108.6%; p=1.7e-24) |
| overall | 0.3998 | 0.5199 (+30.1%; p=1.9e-23) | 0.4248 (+6.3%; ns) | 0.5936 (+48.5%; p=1.2e-50) |

### Per family: reconstruction RMSE

| Family | A | B (vs A) | C (vs A) | D (vs A) |
|---|---:|---|---|---|
| circle | 0.0607 | 0.0416 (-31.4%; p=1.8e-19) | 0.0514 (-15.3%; p=8.3e-06) | 0.0369 (-39.1%; p=4.4e-22) |
| spiral | 0.0883 | 0.0466 (-47.3%; p=3.1e-23) | 0.0711 (-19.5%; p=4.8e-10) | 0.0434 (-50.9%; p=1.8e-24) |
| ellipse | 0.0946 | 0.0436 (-53.9%; p=1.6e-24) | 0.0668 (-29.4%; p=3.2e-15) | 0.0408 (-56.9%; p=1.5e-24) |
| figure_eight | 0.1364 | 0.0654 (-52.1%; p=1.6e-23) | 0.0754 (-44.8%; p=5.1e-22) | 0.0568 (-58.4%; p=1.3e-23) |
| sinusoid | 0.0607 | 0.0336 (-44.7%; p=9.1e-24) | 0.0450 (-25.9%; p=4.5e-13) | 0.0282 (-53.6%; p=1.6e-24) |
| overall | 0.0882 | 0.0462 (-47.6%; p=8.4e-116) | 0.0619 (-29.8%; p=1.5e-67) | 0.0412 (-53.2%; p=1.1e-119) |

Reconstruction: B, C and D have lower mean error than A in all five families, and every one of these differences is significant after correction.

## 4. Separate effects of latent dimension and GRU width, and their interaction

Effects are computed per trajectory from the four cells (positive = higher error). Intervals are 95% bootstrap intervals over test trajectories; p-values are Holm-adjusted Wilcoxon tests over 36 factorial tests. **These quantify test-set sampling only, not training-seed variability** (section 8).

| Group | Metric | Latent 8 to 16 | GRU 64 to 128 | Interaction (D-C)-(B-A) |
|---|---|---|---|---|
| overall | forecast | +0.1445 [+0.127, +0.161] (p=6.6e-47) | +0.0494 [+0.036, +0.062] (p=2.3e-11) | +0.0486 [+0.025, +0.073] (p=0.0018) |
| circle | forecast | +0.1703 [+0.141, +0.199] (p=6.8e-16) | +0.0803 [+0.054, +0.106] (p=8e-07) | +0.1193 [+0.087, +0.152] (p=6.5e-09) |
| spiral | forecast | +0.1028 [+0.055, +0.152] (p=0.1) | +0.0662 [+0.033, +0.100] (p=0.005) | +0.1181 [+0.070, +0.170] (p=0.00078) |
| ellipse | forecast | +0.0881 [+0.056, +0.119] (p=8.6e-07) | +0.0444 [+0.018, +0.070] (p=0.005) | +0.2094 [+0.173, +0.247] (p=1.3e-16) |
| figure_eight | forecast | +0.0153 [-0.015, +0.045] (p=0.55) | -0.0048 [-0.040, +0.031] (p=0.65) | +0.0764 [+0.008, +0.143] (p=0.1) |
| sinusoid | forecast | +0.3458 [+0.330, +0.363] (p=7.4e-25) | +0.0606 [+0.042, +0.078] (p=3e-08) | -0.2801 [-0.307, -0.253] (p=2.5e-23) |
| overall | reconstruction | -0.0313 [-0.033, -0.030] (p=1.1e-118) | -0.0156 [-0.017, -0.014] (p=3.3e-75) | +0.0213 [+0.018, +0.024] (p=8.2e-45) |
| circle | reconstruction | -0.0167 [-0.019, -0.014] (p=7.1e-22) | -0.0070 [-0.009, -0.005] (p=1.6e-09) | +0.0047 [+0.001, +0.008] (p=0.046) |
| spiral | reconstruction | -0.0347 [-0.039, -0.031] (p=4.1e-24) | -0.0102 [-0.013, -0.007] (p=2.3e-10) | +0.0141 [+0.008, +0.020] (p=3.5e-06) |
| ellipse | reconstruction | -0.0385 [-0.042, -0.035] (p=7.4e-25) | -0.0153 [-0.018, -0.012] (p=2.8e-15) | +0.0250 [+0.019, +0.031] (p=3.2e-13) |
| figure_eight | reconstruction | -0.0448 [-0.050, -0.040] (p=1.2e-23) | -0.0349 [-0.039, -0.030] (p=1.3e-22) | +0.0524 [+0.044, +0.061] (p=2.3e-20) |
| sinusoid | reconstruction | -0.0220 [-0.025, -0.020] (p=8.8e-25) | -0.0105 [-0.013, -0.009] (p=4.5e-17) | +0.0103 [+0.007, +0.014] (p=9e-06) |

### Encoder width alone did not produce a clearly distinguishable overall forecasting difference

Cell C (GRU 128, latent 8) versus the baseline A: the mean forecast RMSE differs by +0.0250 (+6.3%; bootstrap 95% interval [+0.0031, +0.0461], which only just excludes zero), the median difference is +0.0109, C is better than A on 48% of trajectories, and the Holm-adjusted Wilcoxon p-value is 0.18 (not significant). The small mean difference comes mostly from the sinusoid family (+0.201, p=4.9e-18), while ellipse is better (-0.060, p=0.013); circle, spiral and figure-eight show no distinguishable difference. The size of this difference is also well inside the run-to-run variation seen between identically configured runs (section 8). In short, widening the GRU on its own did not clearly change forecasting in this single-seed experiment.

### Latent dimension and the combined cell

- Latent-only cell B versus A: overall forecast +30.1% (p=1.9e-23), better than A on 35% of trajectories. The difference is significant for circle (+0.111) and sinusoid (+0.486); for spiral, ellipse and figure-eight it is not distinguishable from zero.
- Combined cell D versus A: overall forecast +48.5% (p=1.2e-50), better on 27% of trajectories. D has a higher mean than A on all five families, significantly so for circle, spiral, ellipse and sinusoid; figure-eight (+0.011) is not distinguishable.
- The overall main effects are +0.144 (latent) and +0.049 (GRU width). The GRU main effect is an average over both latent sizes; it is carried mainly by D versus B (+0.074), not by C versus A (+0.025). Describing both factors as independently worsening forecasting would therefore overstate the evidence.
- **Interaction.** The overall forecast interaction is +0.049 (p=0.0018): raising the latent size changes forecast error more at GRU 128 (D minus C = +0.169) than at GRU 64 (B minus A = +0.120), so the two changes reinforce each other overall. The sign is the opposite for sinusoid (-0.280), where they offset, so the pattern differs between families and an overall average should not be over-interpreted.

## 5. The forecasting difference depends on the horizon

The 7 to 10 window is not uniform. Mean per-trajectory RMSE inside sub-windows (derived from the saved predictions):

| Window | A | B | C | D | B / C / D better than A on |
|---|---:|---:|---:|---:|---|
| 7 < t <= 8 | 0.2630 | 0.2174 | 0.1755 | 0.2332 | 59% / 71% / 52% of trajectories |
| 8 < t <= 9 | 0.3940 | 0.4926 | 0.3649 | 0.5684 | 37% / 55% / 27% of trajectories |
| 9 < t <= 10 | 0.4818 | 0.7037 | 0.5951 | 0.8101 | 31% / 41% / 24% of trajectories |

Per family (mean per-trajectory RMSE in the window):

| Family | Window | A | B | C | D |
|---|---|---:|---:|---:|---:|
| circle | 7 < t <= 8 | 0.170 | 0.162 | 0.106 | 0.207 |
| circle | 8 < t <= 9 | 0.259 | 0.354 | 0.218 | 0.511 |
| circle | 9 < t <= 10 | 0.332 | 0.513 | 0.424 | 0.702 |
| spiral | 7 < t <= 8 | 0.302 | 0.222 | 0.210 | 0.252 |
| spiral | 8 < t <= 9 | 0.486 | 0.492 | 0.431 | 0.625 |
| spiral | 9 < t <= 10 | 0.620 | 0.739 | 0.700 | 0.915 |
| ellipse | 7 < t <= 8 | 0.297 | 0.162 | 0.152 | 0.203 |
| ellipse | 8 < t <= 9 | 0.402 | 0.359 | 0.273 | 0.510 |
| ellipse | 9 < t <= 10 | 0.425 | 0.497 | 0.463 | 0.708 |
| figure_eight | 7 < t <= 8 | 0.308 | 0.259 | 0.195 | 0.260 |
| figure_eight | 8 < t <= 9 | 0.453 | 0.459 | 0.397 | 0.492 |
| figure_eight | 9 < t <= 10 | 0.582 | 0.552 | 0.564 | 0.600 |
| sinusoid | 7 < t <= 8 | 0.238 | 0.282 | 0.215 | 0.244 |
| sinusoid | 8 < t <= 9 | 0.370 | 0.801 | 0.505 | 0.704 |
| sinusoid | 9 < t <= 10 | 0.450 | 1.217 | 0.825 | 1.126 |

Using the across-trajectory RMSE at each time (`error_vs_time` in the JSON, Figure 3): the larger cells are better than A just after t = 7 and then cross above it: B first exceeds A at t = 8.09 (stays above from t = 8.09), C at t = 8.64 (from 8.64), D at t = 7.94 (from 7.94). At t = 10 the across-trajectory RMSE is A 0.623, B 0.912, C 0.828, D 1.012. The full-window forecast difference in section 3 is therefore a **late-horizon effect (faster error growth)**; a shorter forecast window would give the opposite sign for B and D overall. Any statement about "worse forecasting" has to name the horizon.

## 6. Training histories and checkpoint selection

Checkpoints are chosen by validation -ELBO only (negative log-likelihood of the 30 observed noisy points plus KL). Forecast RMSE is logged each epoch but never used. Intermediate checkpoints were not saved, so the table below uses the **validation** split only; it cannot be extended to the test set.

| Cell | Selected epoch | Val. forecast RMSE at selected epoch (pooled) | Lowest val. forecast RMSE over training (epoch) | Gap | Train NLL / val NLL at selected epoch | Mean val. -ELBO, epochs 251-275 / 276-300 |
|---|---:|---:|---|---:|---|---|
| A (latent 8, GRU 64) | 294 | 0.4989 | 0.4976 (270) | +0.3% | 8.5 / 16.9 | 52.0 / 47.9 |
| B (latent 16, GRU 64) | 299 | 0.5909 | 0.5758 (224) | +2.6% | -63.8 / -69.3 | -20.8 / -22.6 |
| C (latent 8, GRU 128) | 299 | 0.4950 | 0.4446 (190) | +11.3% | -42.7 / -39.3 | 1.3 / -4.3 |
| D (latent 16, GRU 128) | 296 | 0.6487 | 0.6265 (152) | +3.5% | -67.0 / -72.3 | -24.1 / -26.2 |

- Selection by validation -ELBO picks epochs 294-299 of 300, so the cells were still (slowly) improving on the training objective when the cosine schedule ended. Training had not fully converged.
- Selection could matter modestly: selecting by lowest validation forecast RMSE instead would lower C's validation forecast error by 11.3%, but it would not change the picture for B and D (their lowest validation forecast errors, 0.5758 and 0.6265, remain well above A's 0.4976). Selection is therefore **not** a sufficient explanation of the forecasting differences. (This is a post-hoc diagnostic on validation data, not a proposed selection rule.)
- Validation and test agree on the ordering of the cells: pooled test forecast RMSE A 0.468, C 0.506, B 0.603, D 0.665; validation A 0.499, C 0.495, B 0.591, D 0.649 (A and C swap places; they are close).

## 7. Is this overfitting?

- **Classical overfitting (validation loss rising while training loss falls) was not observed.** Validation NLL and validation context RMSE fall throughout training (Figure 4), and training and validation NLL at the selected epoch differ by at most 8.4 nats for every cell (table above). Training NLL uses a sampled latent state and validation uses the posterior mean, so the two are not strictly like-for-like.
- That does **not** rule out other explanations. The training loss only supervises the 30 observed points at t <= 7 (verified in `batch_objective`); no loss term involves t > 7. A model can therefore fit the observed interval very well while its extrapolation is left unconstrained. This is a **plausible hypothesis**, not a tested one: the `full_clean` supervision option that would test it has not been run.

## 8. Two different kinds of uncertainty

- **Test-trajectory uncertainty** (what every interval and p-value in this report measures): small, because 750 trajectories are compared pairwise.
- **Training-seed uncertainty: not measured here.** Each cell is one training run. Reference scale from existing results:
  - The same architecture trained twice with different data streams differs in mean forecast RMSE by +22.4% (cell A versus the earlier baseline: 0.3998 versus 0.3265) and +20.7% (cell D versus the earlier medium run: 0.5936 versus 0.4919). Two pairs only, so this is an indication of scale, not a variance estimate, and it does not say which run is the more representative.
  - In the five-seed two-shape study (a different experiment) the forecast RMSE of one architecture ranges from 0.169 to 0.283 (mean 0.219, SD 0.050, coefficient of variation 23%).
  - The forecast differences in this ablation (B +30.1%, C +6.3%, D +48.5% versus A) are of the same order as this run-to-run variation. A very small p-value means the two *fixed trained models* differ on the test set; it does not show that the architecture change causes the difference in new training runs.
- Reconstruction improvements are much larger (-29.8% to -53.2%) and consistent across all five families, but they are also single-seed.

## 9. Interpretation, separated by strength of evidence

**Observed (single seed, this protocol)**
1. Reconstruction of withheld context points improves for every larger cell in every family (overall -47.6%, -29.8%, -53.2% for B, C, D versus A; cell A is worse than linear interpolation, B, C and D are better).
2. Over the full 7 to 10 window the mean forecast error is higher for B (+30.1%) and D (+48.5%) than for A; for C it is +6.3% and the difference is not significant overall.
3. The latent-dimension change is associated with the larger forecast change; the encoder-width change alone is not clearly distinguishable from the baseline overall.
4. The forecast differences appear late in the horizon: larger cells are better than A in the first part of the window and worse after about t = 8 to 8.6.
5. The sinusoid family contributes most of the overall forecast difference for B, C and D; circle contributes for B and D; figure-eight shows no distinguishable difference for any cell.

**Plausible explanations (not tested)**
- The training objective supervises only observed points at t <= 7, so extrapolation is unconstrained and may depend on details of the learned latent dynamics that differ between sizes.
- Larger latent spaces may produce faster error growth in the learned ODE (consistent with the late-horizon pattern).
- Differences in optimisation (the cells were still improving at epoch 300) or in initialisation.

**Not established**
- That capacity worsens forecasting in general, or that the effect would reappear with other seeds.
- Why forecasting degrades; any causal mechanism.
- That latent dimension and GRU width have separable, independent effects (the interaction is significant and family-dependent).
- That any of these models beats a stronger baseline than hold-last or constant velocity (none was tested).

## 10. Limitations
- One seed per cell; no multi-seed confirmation of any ablation result.
- Single synthetic dataset, one observation protocol (30 points, noise 0.05, forecast window 7 to 10).
- Only naive forecast baselines (hold-last, constant velocity); no parametric or sequence-model baseline.
- Only validation-split diagnostics are available for epochs other than the selected one.
- `torchdiffeq` fixed-step `rk4` is the 3/8-rule fourth-order method; predictions at query times between solver steps are linearly interpolated. Measured effect on RMSE is about 2e-4 (see the audit), negligible here.

## 11. Recommended next steps (not executed)
1. Multi-seed confirmation of at least cells A and D (seeds 123, 456, 789, 2026, with `--seed` and `--data-seed` set per seed); B and C as well if affordable. About 81 minutes and 0.3 MB per run.
2. Report effects as mean +/- SD across seeds, and count in how many seeds each sign reproduces.
3. If degradation reproduces: test the observed-only objective hypothesis (`--supervision full_clean`, a labelled diagnostic) before proposing any fix.
4. Add stronger baselines (per-family parametric fit extrapolation; a sequence-model baseline) before claiming any forecasting benefit.

## 12. How to present this to a research supervisor

Suggested wording: "In an exploratory, single-seed 2x2 ablation on the five-family dataset (latent 8 or 16, GRU 64 or 128; identical data streams), larger encoders reconstruct withheld points much better (-29.8% to -53.2%). Over the full 7 to 10 forecast window the largest model and the latent-only variant forecast worse on average (+48.5% and +30.1%), whereas widening the GRU alone made no clearly distinguishable overall difference (+6.3%, not significant). The effect appears only after about t = 8; early in the window the larger models are better. These differences are of the same order as the seed-to-seed variation seen in our earlier five-seed study, so we have not shown that capacity changes forecasting; the next step is multi-seed confirmation."

Do **not** say: that capacity "hurts" forecasting in general; that the controlled run shows the "true" architectural effect; that this is overfitting; that latent and GRU effects are independent; that the work replicates the paper or has produced an improved forecasting model.

## 13. Files and figures

Files in `results/ablation_2x2/`: `ablation_2x2_results.json` (summary statistics per model and family, paired contrasts, factorial effects, error versus time; it does **not** store per-trajectory RMSE), `predictions_test.npz` (test truth, observations, and predictions of all seven models), `checkpoint_selection_audit.md/.json`, `tables/` (5 Markdown tables), `figures/` (6 figures), `ABLATION_2X2_REPORT.original_uncorrected.md` (backup).

| Figure | Content |
|---|---|
| `fig1_cells_by_family.png` | Mean forecast and reconstruction RMSE per family for cells A-D with baselines (error bars: standard error over test trajectories) |
| `fig2_interaction.png` | 2x2 interaction plots per family and overall, forecast and reconstruction |
| `fig3_error_vs_time.png` | Across-trajectory RMSE versus time on the **test** split (before t = 7 it includes the observed timestamps) |
| `fig4_training_histories.png` | Validation NLL, training NLL, validation context RMSE and validation forecast RMSE per epoch (validation split only) |
| `fig5_percent_improved.png` | Share of the same test trajectories on which B, C, D beat A |
| `fig6_examples.png` | First test trajectory of each family; illustration only, not selected as typical |

Known cosmetic issues (figures were left unchanged to preserve the original results): in Figure 5 (right panel) the legend overlaps some bars; Figure 3 curves before t = 7 include the observed timestamps, so they are not forecast errors.

## Appendix A. Corrections relative to the original report

| # | Original statement | Corrected |
|---|---|---|
| 1 | Overall reconstruction RMSE A/B/C/D = 0.0796/0.0533/0.0563/0.0404; changes vs A -33.0%/-29.3%/-49.2% | 0.0882/0.0462/0.0619/0.0412; changes -47.6%/-29.8%/-53.2% (forecast numbers in the original were correct) |
| 2 | Holm correction "within the 90 factorial tests" | 36 factorial tests (and 72 cell-contrast tests in a separate correction family) |
| 3 | JSON "includes per-trajectory RMSE for all test trajectories" | It does not; it holds summaries, paired contrasts, factorial effects and error versus time |
| 4 | "Both factors independently worsened forecast accuracy" | Latent-only B: significant on circle and sinusoid only; GRU-only C: not distinguishable overall (section 4); the GRU main effect is carried by D versus B |
| 5 | Interaction described as "partially cancel but reinforce" | Overall forecast interaction is positive (effects reinforce); it offsets only for sinusoid |
| 6 | "Latent main effect is negative for forecast" | Sign convention was confused: positive means higher error (worse) |
| 7 | "latent-16 and latent-32 cells" | The 2x2 has no latent-32 cell (latent 8 and 16 only) |
| 8 | "Robust to checkpoint-selection bias (validation -ELBO and test forecast RMSE are well-correlated)" | Unsupported as stated. What is shown: selection is unlikely to explain B and D, and validation and test order the cells similarly (section 6) |
| 9 | Figure 3 described as validation and test; Figure 4 as including -ELBO | Figure 3 is test only; Figure 4 shows validation NLL, training NLL, validation context RMSE, validation forecast RMSE |
| 10 | Controlled run shows the "true" architecture effect; uncontrolled runs favoured forecast "by chance" | Removed: the 21-22% differences show run-to-run variability, not which run is right (section 8) |
| 11 | "Classic reconstruction-forecast trade-off confirmed" | Not established from one seed |
| 12 | (chat summary) "95 tests pass" | The suite has 106 tests, all passing |
| 13 | Omitted | Forecast difference is a late-horizon effect (section 5) |
