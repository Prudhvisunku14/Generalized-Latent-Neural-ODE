# Does dynamics conditioning reduce the unified-model forecasting gap? Technical report

Date: 2026-11-15. Written for: the research team deciding whether to pursue a dynamics-conditioned
Latent ODE as the basis for a multi-family trajectory model.

Evidence: `results/conditioning/` (figures, `conditioning_study.json`, `comparison_table.md`,
`probe_scores.csv`), checkpoints and per-epoch histories in `checkpoints/conditioning/`, logs in
`logs/conditioning/`. Earlier results are untouched. The validation-study report is
`reports/validation_study_report.md`.

## 1. Answer

**The hypothesis is not supported at this scale.** Across 5 seeds with identical training data, the
dynamics-conditioned model M3 does not forecast better than the unified baseline M1 or the parameter-matched control
M2, on either family.

| Forecast RMSE (test, mean ± SD over 5 seeds) | Circle | Spiral |
|---|---|---|
| Specialists (validation study, 3 seeds) | 0.087 ± 0.006 | 0.198 ± 0.009 |
| **M1** unified Latent ODE | 0.145 ± 0.007 | 0.264 ± 0.036 |
| **M2** unconditioned, parameter-matched | 0.172 ± 0.096 (median 0.127) | 0.270 ± 0.113 (median 0.227) |
| **M3** dynamics-conditioned | 0.176 ± 0.067 (median 0.147) | 0.299 ± 0.068 (median 0.277) |
| M2aug ablation (3 seeds) | 0.301 ± 0.108 | 0.435 ± 0.095 |

- Seed-paired difference **M3 − M1** in forecast RMSE: circle +0.031, 95% CI [−0.050, +0.112]; spiral +0.035,
  95% CI [−0.063, +0.133]. M3 is better on 2 of 5 seeds for circles and 1 of 5 for spirals. Closing the specialist gap would need
  −0.058 (circle) and −0.065 (spiral), which is at or beyond the lower edge of these intervals. The point estimates go the wrong way.
- Averaged over seeds per trajectory (identical observations), M3 forecasts are 21% (circle) and 13% (spiral) worse than M1.
  Bootstrap CIs are [+0.024, +0.038] and [+0.025, +0.046].
- M3 reconstructs slightly *better* than M1 (circle −0.002, spiral −0.004 RMSE; trajectory-level CIs exclude 0;
  seed-level not significant). This is a small effect.
- The learned code **is used and does control radial growth** (§6). Conditioning changes what the model represents
  without improving forecasts here.

## 2. Prior work and what is (not) new

Conditioning a neural ODE's vector field on an inferred, trajectory- or environment-specific variable is established:

- **Neural ODE Processes** (Norcliffe, Bodnar, Day, Moss, Liò, ICLR 2021) encode a context set into an initial-state
  variable L0 and a separate dynamics variable D that conditions the ODE. This is essentially M3's factorisation; NDPs make
  D stochastic, while M3's c is deterministic.
- **GOKU-nets** (Linial, Ravid, Eytan, Shalit, CHIL 2021) infer both the initial state and the parameters of a known ODE
  from observations with a VAE.
- **CoDA** (Kirchmeyer et al., ICML 2022) conditions a dynamics model on per-environment context vectors via a
  hypernetwork. **LEADS** (Yin et al., NeurIPS 2021) splits dynamics into shared and environment-specific components.
- Background: Latent ODE (Chen et al., NeurIPS 2018; Rubanova et al., NeurIPS 2019), Augmented Neural ODEs (Dupont et al.,
  NeurIPS 2019; relevant to the M2aug ablation), static-vs-dynamic latent factorisation in sequence VAEs (Li & Mandt, ICML 2018).

The first four were checked by web search on 2026-11-15. The background references are cited from memory and should be
checked before publication. This was not a systematic literature review.

**Genuinely new here:** not the architecture. The candidate contribution is methodological. It is a controlled,
parameter-matched, data-identical, multi-seed comparison of conditioned vs unconditioned latent ODEs on sparse, irregular, noisy
2-D trajectories, with causal tests of the code (ablation, swap, directional sensitivity) and a negative result on the
interference hypothesis at this scale. That is a modest contribution and would need broader experiments (§9) to be publishable.

## 3. Models

| | M1 unified | M2 parameter-matched | M3 dynamics-conditioned | M2aug (ablation) |
|---|---|---|---|---|
| Latent z | 8 | 8 | 8 | 12 |
| Code c | – | – | 4, c = tanh(W_c h) | – |
| Encoder | GRU(4→64), heads 64→16 | same | same + 64→4 code head | GRU(4→64), 64→24 |
| Dynamics f | MLP 8→64→64→8 | MLP 8→67→67→8 | MLP 12→64→64→8 | MLP 12→64→64→12 |
| Decoder g | MLP 8→64→64→2 | same | same | MLP 12→64→64→2 |
| **Parameters** | **24,602** | **25,049** | **25,118** | 25,894 |

Activations are tanh. M2 is within 0.3% of M3's parameter count. M1 is itself within 2.1%, so the parameter control mainly
rules out a capacity explanation. M2aug has the same ODE state size as M3 (12), but its extra 4 dimensions evolve freely and are
KL-regularised, which isolates the effect of c being *constant*. Architecture diagram: `fig1_architecture_m3.png`.

## 4. Mathematical formulation (M3)

Observations {(x_i, t_i)}, i = 1..30, with 0 = t_1 < … < t_30 ≤ 7, are normalised with training-set statistics.

- Encoder: h = GRU over the reversed sequence of (x_i, t_i/T, Δt_i/T), with T = 10.
  (μ, log σ²) = W h ∈ R⁸ × R⁸, and **c = tanh(W_c h) ∈ [−1, 1]⁴**, a deterministic point estimate.
- Latent initial state: z₀ = μ + σ ⊙ ε with ε ~ N(0, I) during training; z₀ = μ at evaluation.
- Dynamics: **dz/dt = f_θ(z, c)** and dc/dt = 0, integrated from t = 0 to 10 with RK4 (step 0.1).
- Decoder: x̂(t) = g_ψ(z(t)).
- Objective (per trajectory):
  L = Σ_i Σ_d [ (x_id − x̂_d(t_i))² / (2s²) + log s + ½ log 2π ] + KL( N(μ, σ²) ‖ N(0, I) ),
  with s = 0.05 / scale. This is −ELBO for z₀; **c carries no KL term**.

M1, M2 and M2aug are identical without c. Family labels and future coordinates are never inputs; `tests/test_conditioning.py`
checks the forward signature and the encoder input size.

## 5. Training and evaluation protocol

- Every model uses `configs/model_d.yaml` unchanged except for the architecture key (`configs/conditioning/*.yaml`).
  Settings: Adam 3e-3 with cosine schedule to 1e-5, 300 epochs, batch 32, gradient clipping at 1, 100-point training grid,
  30 observations, KL weight 1, patience 60. Seeds 42, 123, 2024, 2025, 2026 for M1, M2 and M3; 42, 123, 2024 for M2aug.
- **Identical training data**: `--data-seed = seed` gives batch order and training observation subsets their own generators,
  so all models at a given seed see byte-identical batches. Verified directly.
- Checkpoint = minimum validation −ELBO (z₀ = μ). Best epochs ranged from 267 to 300; all runs were still improving near the end.
- The hyperparameters were tuned for M1 in the earlier study. **M3 received no tuning of its own**, which may favour M1.
- Test split (300 trajectories) used once, with identical observation subsets for every model (seed 123) and posterior-mean z₀.
  Statistics: seed-level mean ± SD with a Student-t 95% CI; seed-paired differences with a t-test; trajectory-paired
  differences on seed-averaged errors with a bootstrap 95% CI and Wilcoxon test.
- Code probes are linear (standardised ridge or logistic, with internal CV), **fit on the validation split and scored on test**.

## 6. Results

### 6.1 Full comparison (mean ± SD over seeds; `comparison_table.md`)

| Model | Family | Recon RMSE | Forecast RMSE | ADE | FDE | Chamfer | radius drift @t=10 | ω error (t>7) | growth error |
|---|---|---|---|---|---|---|---|---|---|
| M1 | circle | 0.0524 ± 0.0052 | 0.145 ± 0.007 | 0.193 ± 0.010 | 0.284 ± 0.016 | 0.0094 ± 0.0020 | 0.139 ± 0.009 | 0.069 ± 0.009 | – |
| M1 | spiral | 0.0665 ± 0.0063 | 0.264 ± 0.036 | 0.349 ± 0.049 | 0.536 ± 0.070 | 0.0545 ± 0.0118 | 0.351 ± 0.096 | 0.101 ± 0.013 | 0.033 ± 0.007 |
| M2 | circle | 0.0516 ± 0.0056 | 0.172 ± 0.096 | 0.224 ± 0.116 | 0.353 ± 0.228 | 0.0122 ± 0.0078 | 0.173 ± 0.068 | 0.101 ± 0.109 | – |
| M2 | spiral | 0.0646 ± 0.0073 | 0.270 ± 0.113 | 0.354 ± 0.139 | 0.560 ± 0.271 | 0.0529 ± 0.0262 | 0.320 ± 0.115 | 0.108 ± 0.073 | 0.031 ± 0.009 |
| M3 | circle | 0.0502 ± 0.0039 | 0.176 ± 0.067 | 0.228 ± 0.077 | 0.361 ± 0.170 | 0.0099 ± 0.0064 | 0.139 ± 0.081 | 0.114 ± 0.077 | – |
| M3 | spiral | 0.0626 ± 0.0054 | 0.299 ± 0.068 | 0.392 ± 0.084 | 0.614 ± 0.154 | 0.0624 ± 0.0119 | 0.381 ± 0.172 | 0.131 ± 0.064 | 0.033 ± 0.009 |
| M2aug | circle | 0.0524 ± 0.0042 | 0.301 ± 0.108 | 0.363 ± 0.119 | 0.704 ± 0.264 | 0.0268 ± 0.0136 | 0.322 ± 0.118 | 0.295 ± 0.172 | – |
| M2aug | spiral | 0.0678 ± 0.0067 | 0.435 ± 0.095 | 0.547 ± 0.108 | 0.978 ± 0.224 | 0.0889 ± 0.0093 | 0.590 ± 0.319 | 0.263 ± 0.102 | 0.047 ± 0.011 |

Radius drift is |r̂ − r|/r at t = 10 about the true centre; ω error is |ω̂ − ω| fitted on t > 7; growth error is |ĝ − g| for spirals.

**Per-seed forecast RMSE** (seeds 42, 123, 2024, 2025, 2026):

- M1: circle 0.153, 0.148, 0.144, 0.146, 0.134; spiral 0.318, 0.269, 0.265, 0.240, 0.225
- M2: circle **0.343**, 0.138, 0.127, 0.123, 0.127; spiral **0.471**, 0.210, 0.231, 0.210, 0.227
- M3: circle 0.136, **0.292**, 0.171, 0.147, 0.134; spiral 0.238, **0.412**, 0.307, 0.277, 0.259

M2 and M3 each have one failed seed (bold). Excluding failures, M2 is the best model by median (circle 0.127, spiral 0.227).
M1 is the most consistent. With 5 seeds and occasional failures, rankings between M1, M2 and M3 are not reliable,
but no reading of the data shows M3 ahead.

### 6.2 Paired comparisons (forecast RMSE)

| Comparison | Family | Seed-paired mean diff [95% CI] | Seeds M3 better | Trajectory-paired diff [bootstrap 95% CI] |
|---|---|---|---|---|
| M3 − M1 | circle | +0.031 [−0.050, +0.112], p = 0.35 | 2/5 | +0.031 [+0.024, +0.038] |
| M3 − M1 | spiral | +0.035 [−0.063, +0.133], p = 0.38 | 1/5 | +0.035 [+0.025, +0.046] |
| M3 − M2 | circle | +0.004 [−0.159, +0.167], p = 0.95 | 1/5 | +0.004 [−0.004, +0.012] |
| M3 − M2 | spiral | +0.029 [−0.170, +0.227], p = 0.71 | 1/5 | +0.029 [+0.018, +0.040] |
| M3 − M2aug | circle | −0.101 (3 seeds), p = 0.20 | 3/3 | −0.101 [−0.116, −0.086] |
| M3 − M2aug | spiral | −0.116 (3 seeds), p = 0.21 | 3/3 | −0.116 [−0.136, −0.098] |

Trajectory-paired intervals treat the 150 trajectories as the unit and do not include seed variance. Seed-paired
intervals are the more conservative evidence.

### 6.3 Error vs horizon (`fig6_error_vs_horizon.png`; mean over seeds)

| Pointwise error | t = 7 | 8 | 9 | 10 |
|---|---|---|---|---|
| circle M1 / M2 / M3 | 0.102 / 0.103 / 0.100 | 0.160 / 0.177 / 0.174 | 0.222 / 0.265 / 0.276 | 0.284 / 0.353 / 0.361 |
| spiral M1 / M2 / M3 | 0.168 / 0.162 / 0.158 | 0.276 / 0.278 / 0.303 | 0.411 / 0.416 / 0.475 | 0.536 / 0.560 / 0.614 |

M3 is slightly best at the end of the context window (t = 7), consistent with its better reconstruction. It falls behind
over the forecast horizon.

### 6.4 Ablation: constant code vs extra free dimensions

M2aug has the same ODE state size as M3 and forecasts much worse (all 3 seeds; −0.10 to −0.12 RMSE in M3's favour).
At this budget, four extra KL-regularised state dimensions that evolve freely hurt extrapolation. Making the
extra dimensions **constant** (M3) avoids that harm, but does not improve on simply not having them (M1 and M2).

## 7. What the dynamics code learns (`fig8_dynamics_code.png`, `probe_scores.csv`)

**c is used.** With z₀ kept fixed, mean forecast RMSE over the 5 seeds was:

| Code intervention | Circle | Spiral |
|---|---|---|
| Posterior c (baseline) | 0.176 | 0.299 |
| c replaced by mean code | 0.408 | 0.643 |
| c shuffled within family | 0.501 | 0.761 |
| c shuffled across all trajectories | 0.535 | 0.738 |

**c controls radial growth, causally** (seed 42):
- Code swap: 60 test circles decoded with their own z₀ and the mean code of *expanding* spirals all became expanding
  (100% positive fitted growth, mean +0.052). With the code of *contracting* spirals, 98% contracted (mean −0.087).
  With their own code, mean growth was −0.010.
- Moving c along the growth-probe direction (fit on validation) changes predicted growth monotonically: −0.19 → −0.11 → −0.01 →
  +0.07 → +0.08 for circles at −2…+2 code SDs. **The same move also changes |ω̂|** (0.92 → 1.18), so growth and angular speed are
  **not disentangled** in c.

**Linear probes** (fit on val, R² or accuracy on test; mean [min, max] over 5 seeds):

| Target | c (M3) | z₀ (M3) | [z₀, c] (M3) | z₀ (M1) |
|---|---|---|---|---|
| spiral growth rate (R²) | **0.53** [−0.01, 0.83] | 0.27 [0.00, 0.60] | 0.62 [−0.08, 0.92] | 0.43 [0.13, 0.61] |
| spiral expanding vs contracting (acc.) | **0.83** [0.47, 0.97] | 0.73 [0.55, 0.89] | 0.88 [0.57, 0.99] | 0.83 [0.73, 0.91] |
| signed ω (R²) | 0.07 [−0.02, 0.40] | 0.08 [−0.06, 0.43] | 0.42 [0.15, 0.79] | 0.10 [−0.03, 0.29] |
| \|ω\| (R²) | 0.04 [−0.01, 0.09] | 0.12 [−0.05, 0.37] | 0.29 [−0.03, 0.47] | −0.00 [−0.09, 0.10] |
| rotation direction (acc.) | 0.57 [0.50, 0.81] | 0.63 [0.53, 0.85] | 0.83 [0.69, 0.99] | 0.63 [0.55, 0.74] |
| radius r0 (R²) | 0.18 [−0.00, 0.27] | 0.45 [0.00, 0.68] | 0.51 [0.06, 0.73] | 0.36 [0.25, 0.44] |
| family, circle vs spiral (acc.) | 0.48 [0.47, 0.49] | 0.50 [0.48, 0.54] | 0.53 [0.49, 0.59] | 0.52 [0.48, 0.56] |

- In 4 of 5 seeds, c carries the spiral growth rate (R² 0.40–0.83) and its sign (83–97%) more linearly than z₀. In seed 123,
  c encodes neither (R² −0.01, accuracy 0.47), **and seed 123 is M3's worst forecasting run** (0.292 and 0.412). This is
  correlational evidence, from n = 5, that forecast quality depends on c capturing growth.
- Angular velocity, rotation direction and radius are **not** linearly concentrated in c. They are spread across z₀ and c,
  with the joint representation decoding them best.
- Family is not linearly decodable from any representation. That does not mean family information is absent: a circle is the
  growth = 0 case of the spiral family, which lies *between* expanding and contracting spirals, so a linear classifier cannot separate
  it. The growth probe is the better test.
- PCA of c (seed 42) is bimodal on PC1 (64% of variance) because tanh saturates near ±1, and the split does not align with
  family or ω.

**Conclusion on the code:** c is a functional dynamics variable that mainly encodes radial growth, and that is causally
verified. It is **not** a disentangled representation of the generating parameters.

## 8. Why conditioning may not help here (hypotheses, not tested)

1. **No real interference to remove.** Circles and spirals are one continuous family (growth g, with g = 0 for circles). An
   autonomous latent ODE can already represent g as an invariant direction of z. M1's z₀ already decodes growth
   (R² 0.43), so a separate code adds little.
2. **Optimisation, not representation, may limit the unified model.** Specialists are stable across seeds (SD 0.006–0.009).
   The retrained M1 is equally stable on circles (0.007) but not on spirals (0.036), and M2 and M3 each had a failed seed.
   Specialists' best epochs were 581–596 of 600. The gap may reflect training difficulty or budget more than interference.
3. **c is unregularised** (no KL), so it can absorb trajectory-specific detail that helps reconstruction (M3 has the best recon)
   but generalises worse over the horizon, a familiar overfitting pattern.
4. M3 had no hyperparameter tuning of its own.

## 9. Limitations

- Five seeds per model, with one failed seed each for M2 and M3. Seed-level CIs are wide (±0.08–0.20 forecast RMSE for M2 and M3).
- One dataset: two closely related families of smooth planar rotations. The interference hypothesis may matter more for
  qualitatively different dynamics, which this study did not test.
- One code size (4), deterministic code, no KL or other regulariser on c, no hypernetwork conditioning, and no tuning for M3.
- The earlier validation-study limitations still apply to every model here: the encoder only works near 30 observations, and
  forecasts show a phase lead and radius drift.
- Specialist numbers come from the validation study. They use the old global-RNG data order rather than `--data-seed`, so they are
  not data-identical to M1, M2 and M3, though they are protocol-identical.
- Note: the retrained M1 (with `--data-seed`) gives circle 0.145 ± 0.007 and spiral 0.264 ± 0.036. Earlier M1 runs gave 0.157 ± 0.037
  and 0.282 ± 0.067. The difference comes only from training data order.

## 10. Recommended next steps (not run)

1. Make c variational, with a KL to N(0, I) like NDPs, and compare against deterministic c. This tests explanation 3.
2. Test whether the gap is an optimisation issue (explanation 2): train M1 longer (600 epochs), or use a lower learning rate for stability, before
   adding architecture.
3. If conditioning is pursued further, test it on families with genuinely different vector fields (e.g. adding a damped
   oscillator or limit cycle), where interference is more plausible.

## 11. Figures (`results/conditioning/`)

1. `fig1_architecture_m3.png`: architecture
2. `fig2_loss_curves.png`: train/val −ELBO, val KL, val forecast RMSE (median and min–max over seeds)
3. `fig3_circle_reconstruction_M3.png`
4. `fig4_spiral_reconstruction_M3.png`
5. `fig5_model_comparison.png`: M1, M2, M3 on identical trajectories (seed 42). Illustrative only: seed 42 is M2's failed run.
6. `fig6_error_vs_horizon.png`
7. `fig7_radius_angular_growth_errors.png`
8. `fig8_dynamics_code.png`: PCA by family, ω and growth, plus directional sensitivity
9. `fig9_best_median_worst_M3.png`

## 12. Reproduction

```
bash scripts/run_conditioning_study.sh        # 18 training runs (CPU, 1 thread each, ~45 min in parallel)
python scripts/12_conditioning_study.py       # evaluation, statistics, probes, figures
python -m pytest -q tests                     # 19 tests
```
