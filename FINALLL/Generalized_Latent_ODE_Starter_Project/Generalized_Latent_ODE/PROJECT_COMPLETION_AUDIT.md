# Project completion audit: Latent Neural ODEs, cross-family generalisation and forecasting

Date: 2026-11-21. Written for: the project owner, ahead of a supervisor meeting.
Nothing was trained, retrained, overwritten or pushed during this audit. Verification scripts are in `audit/` (run from the project root; they only read).

**How to read the labels.** *Verified* = I checked it against code, checkpoints, logs, predictions or result files in this audit. *Not verified* = a document says so, but I did not re-derive it. *Not done* = no evidence it exists.

---

## 1. Executive summary

- **What is solid.** The Latent ODE implementation, the five-family datasets and splits, the training runs, the evaluation pipeline and the 2x2 ablation *statistics* are correct and reproducible. Every number in `ablation_2x2_results.json` was recomputed independently from the saved predictions and matched. The test suite passes (106 tests).
- **What the project has found.** (i) One shared model works for circles, spirals and ellipses but forecasts worse than specialists. (ii) A model trained on three families does **not** zero-shot generalise to figure-eight or sinusoid; it is worse than naive baselines on them. (iii) The synthetic-trained model does not transfer to GeoLife. (iv) In a single-seed 2x2 ablation, larger encoders reconstruct better and, overall, forecast worse.
- **What the project has not done.** It has not developed or tested any method that forecasts better. No capacity result has been repeated across seeds. No stronger baseline (parametric fit, sequence model) exists. No leave-one-family-out experiment exists. The mechanism behind the forecasting degradation is untested.
- **Corrections needed before presenting.** My own `results/ablation_2x2/ABLATION_2X2_REPORT.md` (written earlier in this session) contains wrong reconstruction numbers and several over-stated interpretations (section 10). Do not present it as written.
- **Verdict.** Objective A (systematic investigation): **PARTIALLY COMPLETE**. Objective B (an improved forecasting model): **NOT COMPLETE**. We have **not** demonstrated a better forecasting model.

---

## 2. Implementation verification (Phase 1)

Reference: Chen et al. 2018. Paper facts below come from the arXiv v6 HTML, Sections 5 and 5.1 and Table 2, read by a helper agent. Appendix E (exact ELBO form) was only available as a summary and is **not verified**.

| Question | Finding | Evidence |
|---|---|---|
| 1. Architecture correct? | **Verified.** Encoder to q(z0), autonomous latent ODE, decoder, ELBO objective. | `src/glo/model.py`, `src/glo/pipeline.py`; `tests/test_model.py`, `tests/test_objective.py` |
| 2. GRU encoder working? | **Verified.** Observations are flipped in time, the GRU consumes [x, y, t/T, dt/T], the last hidden state goes through a linear layer to mean and log-variance (clamped to [-8, 5]). Gradients reach the encoder (test). | `model.py` `RecognitionGRU` |
| 3. Initial latent state? | **Verified.** z0 = mu + sigma*eps in training, z0 = mu at evaluation. KL to N(0, I), weight 1. | `model.py` `LatentODE.forward`, `kl_standard_normal` |
| 4. Continuous-time latent dynamics? | **Verified by construction.** `LatentDynamics.forward(t, z)` ignores t (time-invariant, as in the paper); states are evaluated at arbitrary query times. Training used 100 query points, evaluation 200, and a test confirms outputs do not depend on the requested times. Whether the learned field is *meaningful* dynamics is not shown by this check. | `model.py`; `tests/test_inference.py::test_output_times_do_not_change_predictions` |
| 5. RK4 correct? | **Verified with a caveat.** torchdiffeq 0.2.5 `rk4` is the Runge-Kutta 3/8 rule (fourth order), not the classical 1/6 formula. Measured order on a harmonic oscillator: error ratio 16.1 per step halving (fourth order) at grid-aligned times. But the solver steps on a uniform 0.1 grid and **linearly interpolates** to off-grid query times, so outputs at the 200 evaluation points are second-order accurate (error ratio about 4 per halving; 2.7e-3 at step 0.1 on that test problem). Effect on the trained models is small: per-trajectory RMSE changes by 1.4e-4 to 2.6e-4 on average (max 1.2e-3) between rk4 at 0.1, rk4 at 0.025 and tight dopri5. | `audit/verify_2_solver.py`; torchdiffeq source |
| 6. Decoder? | **Verified.** tanh MLP from z(t) to (x, y) in normalised coordinates; predictions are de-normalised with the training-set mean and scale before any metric. | `model.py`, `evaluation.py::collect` |
| 7. Objective? | **Verified.** -ELBO = Gaussian NLL with fixed sigma = 0.05 (data units) summed over the 30 observed points and 2 coordinates, plus KL. Hand-computed unit test passes. | `pipeline.py::batch_objective`; `tests/test_objective.py` |
| 8. Irregular observations? | **Verified.** 30 observation times per trajectory drawn without replacement from the 140 grid times with t <= 7, the first always t = 0. Across the 750 test trajectories there are 750 distinct time sets, gaps from 0.05 to 2.16. Times are multiples of the 0.05 grid, so irregular but discretised. The noise realisation is fixed per trajectory (stored), only the subset varies between epochs. | `audit/verify_3_ablation_statistics.py`; `pipeline.py::prepare_batch` |

**Is this a replication?** **No.** It is an **adapted re-implementation with an extension**. Exact replication would require the paper's data and numbers, which are not reproduced (the README already says so).

| | Chen et al. 2018 (spiral experiment, as read) | This project |
|---|---|---|
| Data | 1,000 two-dimensional spirals, half clockwise, 100 equally spaced steps; noise std not stated | Synthetic circle / spiral / ellipse (+ figure-eight, sinusoid), 1,000 each, 200 points on [0, 10], noise 0.05 |
| Observations | Subsampled without replacement, n = 30, 50, 100 | 30 irregular points at t <= 7 (t = 0 always included) |
| Evaluated on | RMSE on 100 time points beyond the training window (Table 2: latent ODE 0.164 / 0.150 / 0.135; RNN 0.394 / 0.320 / 0.181) | Reconstruction of withheld context points and forecast over 7 < t <= 10 |
| Encoder | RNN, backwards in time, 25 units, cell type not stated | GRU, backwards, inputs include t/T and dt/T; 64 or 128 units in the later work |
| Latent / dynamics / decoder | latent 4; one hidden layer of 20 units each | latent 8 or 16; 64 units, 2 layers (Model D) |
| Solver | Adjoint method described generically; for sequence modelling "the default value of 1.5e-8" tolerance is mentioned | Fixed-step rk4, step 0.1, backpropagation through the solver, no adjoint |
| Objective | VAE-style ELBO; details in Appendix E (not verified) | Fixed-sigma Gaussian NLL + KL, beta = 1 |
| Families | Two directions of one spiral type | Up to five families, one shared model, no shape labels |

The README's "original-style" run (latent 4, GRU 25, MLP 20) is an architecture-size re-implementation on this project's data, not a numerical reproduction. The paper's Table 2 numbers are not comparable with ours.

---

## 3. Dataset verification (Phase 2)

All **verified** by `audit/verify_1_datasets.py` and `tests/test_five_shapes.py`.

| Check | two_shapes | three_shapes | five_shapes |
|---|---|---|---|
| Train / val / test sizes | 1,400 / 300 / 300 | 2,100 / 450 / 450 | 3,500 / 750 / 750 |
| Per family (train / val / test) | 700 / 150 / 150 | 700 / 150 / 150 | 700 / 150 / 150 |
| Trajectory-id overlap between splits | 0 | 0 | 0 |
| Identical-path overlap between splits | 0 | 0 | 0 |
| Observation times all <= 7; t = 0 always observed; 30 per trajectory | yes | yes | yes |
| Measured noise SD | 0.0500 | 0.0501 | 0.0501 |
| Time grid / forecast start | 200 points on [0, 10], 7 | same | same |

- `five_shapes` contains the `three_shapes` trajectories unchanged, with unchanged splits (checked by id and array equality).
- Nearest test-to-train path distance per family is the same size as the distance between training trajectories, so there are no near-duplicates.
- Observed and forecast intervals are separated: the observation mask is zero for t > 7 in every split, and the evaluation code draws observations only from grid times <= 7.
- Shape checks for figure-eight and sinusoid pass (`results/five_shapes/dataset_checks.json`, no failures).
- No shape label is given to the model (`cond_dim = 0`; encoder input is observations and times only). The five-shape model was trained on all five families (checkpoint `experiment = five_shapes`, 700 per family).

**Does the work demonstrate multi-family learning?** Yes, in the sense that one model with no label fits several families. **Unseen-family (zero-shot) evidence is narrower than the project title suggests:**

| Setting | Zero-shot? | Result |
|---|---|---|
| Three-shape model on circle / spiral / ellipse | No (trained on them) | Good fit |
| Five-shape model on all five families | **No (trained on all)** | Not a zero-shot result |
| Frozen three-shape model on figure-eight and sinusoid | **Yes** | Worse than naive baselines (below) |
| Two-shape models on ellipses (README section 4) | Yes | Partial: direction transfers, shape does not (not re-verified here) |

There is no leave-one-family-out experiment, and the zero-shot results come from one model and one seed.

---

## 4. Experimental verification (Phase 3)

**Independently verified**
- Ablation JSON: 252 recomputed means / medians / SDs of per-trajectory RMSE and the pooled RMSE all match (worst difference 6e-8); 750 unique test ids, all from the `five_shapes` test split; all four cells received identical observations in a fresh evaluation.
- Cross-file consistency: forecast means for the earlier five-shape baseline and the frozen three-shape model are identical to six decimals in `results/five_shapes/compare_three_vs_five.json`, `results/capacity_scaling/capacity_scaling_results.json` and `results/ablation_2x2/ablation_2x2_results.json`.
- README seed-study numbers match `results/validation/seed_study/seed_study.json`: reconstruction 0.0599 +/- 0.0069, forecast 0.2194 +/- 0.0502 (two-shape test, 5 seeds). **Per-seed forecast values range from 0.169 to 0.283 for the same architecture and data.**
- Figures 1, 3 and 4 of the ablation agree with the numbers.

**Five-family results (mean per-trajectory forecast RMSE, from the saved JSON)**

| Family | Three-shape model | Five-shape model | Hold-last baseline |
|---|---|---|---|
| circle | 0.140 | 0.259 | 1.137 |
| spiral | 0.291 | 0.427 | 1.276 |
| ellipse | 0.189 | 0.261 | 1.120 |
| figure-eight (unseen by three-shape) | 1.117 | 0.299 | 0.961 |
| sinusoid (unseen by three-shape) | 0.973 | 0.386 | 0.625 |

Reading: on the three original families the five-shape model forecasts **worse** than the three-shape model; on the two new families the three-shape model is **worse than hold-last**, so it does not generalise to them. Reconstruction of unseen families by the three-shape model is far worse than linear interpolation (figure-eight 0.732 vs 0.092).

**Not verified or not done**
- Paired statistics in the capacity JSON and the cell-contrast tables in the ablation JSON were not recomputed (the factorial effects were).
- The validation, conditioning and GeoLife reports were not re-derived beyond the seed-study spot check.
- `reports/` has **no five-shape report** (the planned 10-section report was never written).
- GeoLife-trained model results exist (`results/geolife/main_evaluation/`, `tables/table5_...`) but report sections 9 and 11, README section 8 and the fine-tuned vs scratch analysis were never finished, so the README still says those runs are "in progress".
- `results/ablation_2x2/ABLATION_2X2_REPORT.md` does **not** match the saved results in several places (section 10).

---

## 5. Controlled ablation findings (Phase 4)

**Verified setup.** Cells A (8/64), B (16/64), C (8/128), D (16/128) have 24,602 / 27,186 / 63,642 / 67,250 parameters. All four completed 300 epochs with no traceback, no early stop and no non-finite values. Their training arguments differ **only** in latent size and GRU width. All used seed 42 and `data_seed` 42, 110 steps per epoch, 33,000 steps in total. Checkpoint selection is validation -ELBO only (`scripts/02_train_model.py`; selected epoch equals the history argmin for all four). Validation observation draws come from a fixed generator seeded `seed + 999`, so they are identical across cells. Test observations are identical (fresh evaluation, 750 trajectories).

What the identical `data_seed` guarantees: identical batch order and identical training observation subsets (checked by code path and a unit test; it cannot be observed after the fact). What it does **not** control: the initial weights and the reparameterisation noise, which necessarily differ between architectures.

**Results (mean per-trajectory RMSE over 750 test trajectories; recomputed)**

| Cell | Recon | Recon vs A | Forecast | Forecast vs A | Trajectories with lower forecast error than A |
|---|---|---|---|---|---|
| A | 0.0882 | n/a | 0.3998 | n/a | n/a |
| B | 0.0462 | -47.6% | 0.5199 | +30.1% | 35% |
| C | 0.0619 | -29.8% | 0.4248 | +6.3% | 48% |
| D | 0.0412 | -53.2% | 0.5936 | +48.5% | 27% |

Linear interpolation scores 0.0672 on reconstruction and hold-last 1.024 on forecasting.

**Factorial effects, overall (positive = higher error; recomputed; intervals resample test trajectories only)**

| Effect | Forecast | Reconstruction |
|---|---|---|
| Latent 8 to 16 | +0.1445 [+0.127, +0.161], Holm p 6.6e-47 | -0.0313 [-0.033, -0.030] |
| GRU 64 to 128 | +0.0494 [+0.036, +0.062], Holm p 2.3e-11 | -0.0156 [-0.017, -0.014] |
| Interaction (D-C)-(B-A) | +0.0486 [+0.025, +0.073], Holm p 0.0018 | +0.0213 [+0.019, +0.024] |

Statistics are correct: means, bootstrap intervals (independent resampling agrees within 0.003) and Holm-adjusted p-values (36 tests) all match.

**Does the evidence support "larger latent dimension and GRU width improved reconstruction but worsened forecasting in the controlled single-seed experiment"?**
- **Reconstruction improves: supported.** B, C and D beat A on reconstruction in all five families.
- **Forecasting worsens: supported only in a weaker, qualified form.**
  - It holds clearly for D (worse than A on all five families) and for latent-only B overall (+30%, driven by circle +41%, sinusoid +130% and spiral +9%; B is slightly better on ellipse and figure-eight).
  - **GRU width alone (C) is not distinguishable from A overall** (+6.3%, better on 48% of trajectories, better on ellipse and figure-eight). The overall GRU main effect (+0.049) is carried by D versus B, i.e. by the interaction.
  - The overall interaction is positive: raising latent size hurts more at GRU 128 than at GRU 64 (the effects reinforce). It is negative only for sinusoid (-0.28).
  - **It is a late-horizon effect.** In 7 < t <= 8 the larger cells have *lower* error than A (mean difference -0.07 to -0.10); they cross above A at t = 8.09 (B), 8.64 (C) and 7.94 (D) and are worse by t = 10 (A 0.62, B 0.91, C 0.83, D 1.01). A shorter horizon would reverse the sign.
  - All four cells were still improving when training ended (selected epochs 294-299 of 300).

**Two kinds of uncertainty, kept apart**
- *Test-trajectory uncertainty* is what every interval and p-value above measures. It is small (750 paired trajectories).
- *Training-seed uncertainty* is **not measured at all**: each cell is one training run. Reference scale: the same architecture trained twice with different data streams differs by +22.4% (A versus the earlier baseline) and +20.7% (D versus the earlier medium run), and the five-seed two-shape study shows a forecast CV of about 23% (range 0.169 to 0.283). The ablation's forecast differences (+6%, +30%, +48%) are of the same order as this run-to-run variation. A very small p-value here says the test-set difference between two fixed trained models is real; it does not say the architecture causes it.

---

## 6. Scientific contributions (Phase 6)

**A. Implementation (verified).** Latent ODE with GRU encoder and RK4; synthetic generators for five families with leakage-free splits; paired, per-trajectory evaluation with bootstrap and Holm statistics; baselines; packaged inference with checksums; GeoLife pipeline; 106 passing tests.

**B. Experimental, beyond the 2018 spiral experiment.** The paper (as read) uses two directions of one spiral type, with no other families. This project adds: a unified-versus-specialist comparison over 5 seeds; a dynamics-conditioning study (hypothesis not supported at this scale); zero-shot tests on unseen families (negative); a five-family model; single-seed capacity and 2x2 studies; real-data transfer to GeoLife (negative). Findings are largely negative or exploratory.

**C. Methodological improvement.** **None has been developed and validated.** "Model D" is a larger, longer-trained configuration of the same architecture and improved validation scores over the original-size run on two-shape data (context 0.335 to 0.059, forecast 0.815 to 0.262, validation split, README section 1). That is a hyperparameter change. The conditioning variants did not improve forecasting. The capacity and ablation work identified a limitation; it did not fix it. The multi-family model is **worse** than specialists on forecasting.

**Limitation of the 2018 experiment versus a research gap.** That the paper's spiral experiment covers two directions of one spiral type is a verified limitation of that experiment. Whether *cross-family forecasting* is an open research gap relative to later work (Latent ODE-RNN, Neural ODE Processes, GOKU-nets, CoDA, which the README cites) requires a literature review that this audit did not do. **Not verified.** The README correctly makes no novelty claim.

**What can legitimately be claimed:** a verified reimplementation and evaluation framework; negative results on zero-shot cross-family generalisation and on GeoLife transfer; a documented cost of sharing one model across families; a single-seed observation that larger encoders improve reconstruction and, over the full 7 to 10 horizon, tend to forecast worse in this setup.

---

## 7. Remaining research gaps

- No forecasting improvement has been proposed or tested.
- Mechanism of the forecasting degradation is untested. The training loss supervises only the 30 observed points at t <= 7 (verified in `batch_objective`); the `full_clean` supervision option exists but was never run. A link between this and poor extrapolation is a hypothesis.
- No multi-seed result for any five-family, capacity or ablation experiment.
- No classical baseline (for example a per-family least-squares fit extrapolated forward) and no neural sequence baseline like the paper's RNN. The only baselines are hold-last, constant velocity and linear interpolation.
- No leave-one-family-out generalisation experiment.
- Observation protocol is narrow: always 30 points, always t = 0, noise 0.05, a single horizon (7 to 10). Sensitivity to these is largely untested for the five-family models.
- Documents are stale or missing: no five-shape report, README omits the five-family, capacity and ablation work and still lists GeoLife training as in progress, GeoLife report sections 9 and 11 unfinished.
- Version control: the new work is uncommitted and unpushed (28 changed or untracked paths at audit time; the push is blocked by authentication).

---

## 8. Project completion checklist

| Item | Status | Evidence |
|---|---|---|
| Latent ODE implementation | COMPLETE | model.py, tests (106 pass) |
| Solver correctness | COMPLETE with caveat (3/8-rule RK4; linear interpolation at query times) | audit/verify_2_solver.py |
| Five-family datasets and splits | COMPLETE | audit/verify_1_datasets.py |
| Shared-model training on the intended families | COMPLETE | checkpoints, logs |
| Reconstruction and forecast evaluation pipeline | COMPLETE | evaluation.py; recomputation matched |
| Controlled 2x2 ablation (single seed) | COMPLETE (statistics verified); report needs corrections | audit/verify_3 and verify_4 |
| Multi-seed confirmation (five-family, capacity) | NOT COMPLETE | none exists |
| Forecast-degradation mechanism | NOT COMPLETE | untested hypotheses |
| Cross-family generalisation (zero-shot) | PARTIALLY COMPLETE | single model, single seed, no leave-one-out |
| Stronger baselines | NOT COMPLETE | only naive baselines |
| Improved forecasting method | NOT COMPLETE | none developed |
| Five-shape research report | NOT COMPLETE | no file in reports/ |
| GeoLife wrap-up (report 9/11, README 8, pretraining-benefit stats) | PARTIALLY COMPLETE | results exist, documents not updated |
| README and documentation current | PARTIALLY COMPLETE | omits five-family work |
| Literature gap review | NOT VERIFIED | not performed here |
| Claims in the written ablation report | NOT VERIFIED as written (errors found) | section 10 |

---

## 9. Minimum next steps (recommended, not executed)

| Priority | Task | Why / missing evidence | Training? | Obj. A | Obj. B |
|---|---|---|---|---|---|
| 1 | Correct the written claims (ablation report; write the five-shape report; update README and GeoLife sections) | Presenting wrong numbers is the main risk | No | Essential | Essential |
| 2 | Horizon-resolved forecast analysis from existing predictions (RMSE for 7-8, 8-9, 9-10, per cell and family) | Shows the degradation is late-horizon; costs nothing | No | Essential | Essential |
| 3 | Multi-seed confirmation of the 2x2: at least A and D (and B and C if affordable) with 4 more seeds (123, 456, 789, 2026), setting `--seed` and `--data-seed` per seed | Without this, no capacity claim stands against about 22% run-to-run variation | Yes: 8 runs (A, D) to 16 runs (all four), about 81 min each, 0.3 MB each; with 8 concurrent 4-thread runs roughly 3 to 6 hours wall clock (concurrency of 8 not yet measured; 4 concurrent was verified) | Essential for any capacity claim | Essential |
| 4 | Stronger baselines: per-family parametric least-squares fit extrapolated forward; a GRU sequence-to-sequence or ODE-RNN trained on the same data | The model is only compared with naive baselines | Parametric: no. Neural baselines: yes | Recommended | Essential |
| 5 | Leave-one-family-out (or leave-two-out) training with at least 3 seeds | The "cross-family generalisation" claim currently rests on one frozen model | Yes | Essential if the title keeps "cross-family generalisation" | Optional |
| 6 | Mechanism test: train A and D with `--supervision full_clean` (clean dense targets, a supervised diagnostic that must be labelled as such) | Tests whether the observed-only objective limits extrapolation | Yes (2 to 4 runs) | Optional | Essential before proposing a fix |
| 7 | Evaluate one candidate forecasting improvement chosen from step 6, against the same baselines, multi-seed | Only then can "improved model" be claimed | Yes | Not needed | Essential |
| 8 | Final report and presentation | Deliverable | No | Essential | Essential |

---

## 10. Claims to correct before presenting

Errors in `results/ablation_2x2/ABLATION_2X2_REPORT.md` (written by me, earlier in this session; I did not modify it because you asked me not to overwrite reports):

1. **Section 2 reconstruction numbers are wrong.** The report gives overall reconstruction 0.0796 / 0.0533 / 0.0563 / 0.0404 for A to D and changes of -33.0% / -29.3% / -49.2%. The saved results give 0.0882 / 0.0462 / 0.0619 / 0.0412 and -47.6% / -29.8% / -53.2%. The forecast numbers in that table are correct.
2. "Holm-corrected within the **90** factorial tests": there are **36**.
3. "`ablation_2x2_results.json` includes per-trajectory RMSE for all test trajectories": it does not (summaries, paired statistics, error-vs-time only).
4. "Both factors independently worsened forecast accuracy": over-stated. GRU-only (C) is +6.3% and not distinguishable from A; B and C are better than A on ellipse and figure-eight.
5. The interaction is described as the factors "partially cancel but reinforce degradation". The overall forecast interaction is positive, meaning the effects **reinforce**; it cancels only for sinusoid.
6. "Latent main effect is negative for forecast" confuses the sign convention (positive = higher error = worse).
7. "No overfitting signature (train and validation NLL close for latent-16 and latent-32 cells)": the 2x2 has no latent-32 cell.
8. "The effect is robust to checkpoint-selection bias (validation -ELBO and test forecast RMSE are well-correlated)": the correlations were within-run on the validation split. What is supported is that validation and test rank the four cells almost identically (test pooled forecast A 0.468, C 0.507, B 0.603, D 0.665; validation 0.499 / 0.495 / 0.591 / 0.649, with only A and C swapping).
9. Figure 3 is described as "validation and test"; it is test only. Figure 4 is described as including -ELBO; its panels are validation NLL, training NLL, validation context RMSE and validation forecast RMSE.
10. "The controlled runs show the 'true' architecture effect" and "uncontrolled runs favoured forecast by chance": not supported. The +22% and +21% differences show run-to-run variability; they do not show which run is right.
11. "Classic reconstruction-forecast trade-off confirmed": not established.
12. My earlier chat message said 95 tests pass; the suite now has **106**, all passing.
13. The report omits the key qualification that the degradation appears only later in the horizon.

Claims in the earlier capacity report: I replaced its unsupported statistics earlier in this session (correction notice at its top); no further discrepancies were found against the saved JSON.

Other wording to avoid: "replication" of the paper; "RK4" without saying it is the 3/8 rule with linear interpolation to query times; "overfitting"; "capacity hurts forecasting" as a general statement; "improved model" for Model D without saying it is a larger configuration; "zero-shot" for the five-shape model on any of its five families.

---

## 11. Supervisor presentation readiness

**Can be presented with confidence now:**
- The verified implementation and datasets, including the leakage and separation checks.
- One shared model learns several families without labels but forecasts worse than specialists (5-seed result for two families).
- Negative results, stated as such: zero-shot transfer to unseen families fails; synthetic-to-real transfer to GeoLife fails; conditioning did not help.
- The 2x2 ablation as an **exploratory, single-seed** study: reconstruction improves with capacity; full-horizon forecast is worse for the largest cell, and the effect appears only after about t = 8; the seed-to-seed variability scale is of the same order as the effects.

**Present only as a hypothesis or plan:** why forecasting degrades, any improvement method, any capacity "law".

**Not ready:** a final research report (missing seeds, baselines, five-shape report) and any claim of a better forecasting model.

---

## 12. Final verdict

**"Is our Latent Neural ODE research project complete?"** **No.**

- **Objective A (systematic investigation of reconstruction, forecasting and cross-family generalisation): PARTIALLY COMPLETE.** The infrastructure, datasets, five-seed studies for two families, single-seed five-family and capacity studies, and the negative generalisation results are done and verified. Multi-seed confirmation of the five-family and capacity findings, a leave-one-family-out test and stronger baselines are missing. Sufficient for a research presentation (with the corrections above); **not** yet sufficient for a final report.
- **Objective B (an improved Latent ODE with better forecasting across families): NOT COMPLETE.** No improved method has been developed or tested, and the larger-capacity variants did not forecast better overall. Not sufficient for a presentation as an achieved result or for a final report.

**Have we demonstrated a better forecasting model? No.**

---

## Addendum (2026-11-21): refinements found while correcting the documents

The corrections requested after this audit were made (see section 13). Checking the new text against the saved data refined four statements above; the verdicts and conclusions are unchanged.

1. **"Worse than naive baselines" (sections 1 and 3) is imprecise.** On figure-eight and sinusoid the frozen three-shape model is worse than **hold-last** for forecasting and worse than **linear interpolation** for reconstruction, but it is *better* than constant-velocity extrapolation, which is itself a poor baseline on these data (forecast RMSE 2.6 and 1.4).
2. **Per-family significance in section 5.** After Holm correction: latent-only cell B is significantly worse than A only on circle and sinusoid; the combined cell D is significantly worse on circle, spiral, ellipse and sinusoid (figure-eight is not distinguishable); GRU-only cell C is significantly *worse* on sinusoid and significantly *better* on ellipse, with no distinguishable difference elsewhere or overall (Holm p = 0.18). Statements above that B or C are "better on ellipse and figure-eight" are point estimates only (C on ellipse is the one significant case).
3. **Naive baselines versus the larger cells.** Overall all four ablation cells forecast far better than both naive baselines, but on the sinusoid family B (0.860) and D (0.781) are worse than hold-last (0.625).
4. **Five-shape baseline reconstruction.** Its reconstruction is not better than linear interpolation on any family (essentially tied on circle, +0.2%, and worse on the other four).

## 13. Follow-up: documents corrected after this audit

- `results/ablation_2x2/ABLATION_2X2_REPORT.md` was rewritten from the saved results (original preserved read-only as `ABLATION_2X2_REPORT.original_uncorrected.md`); the 13 corrections are listed in its Appendix A.
- `reports/five_shapes_report.md` was written from existing results only.
- `README.md` was updated (status, objectives, five-family data, results 8 to 10, limitations, structure).
- `audit/verify_5_corrected_documents.py` re-derived 259 numbers from the raw saved predictions and found 0 mismatches with the written tables (a deliberately altered copy was detected).
