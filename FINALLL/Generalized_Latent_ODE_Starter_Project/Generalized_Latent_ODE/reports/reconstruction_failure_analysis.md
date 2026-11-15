# Reconstruction failure analysis — original two-shape checkpoint

Date: 2026-11-15. Checkpoint under audit: `checkpoints/two_shapes_cond0_best.pt`
(unchanged; a copy is in `backups/original_2026-11-15/`).

## Summary

The collapsed circle/spiral predictions are **not** caused by a bug in the encoder, the ODE
integration, the decoder or the loss. The confirmed causes are:

1. **The checkpoint is from a run that was stopped after about 3 epochs.** It was never a completed training run (main cause).
2. **No `torchdiffeq` installed.** The run silently used the offline RK4 fallback, but the checkpoint
   still records `solver='dopri5'` (a metadata/reproducibility defect).
3. **`latent_dim = 4` is structurally too small** for this dataset (a capacity limit, secondary).
4. Tooling gaps hid problem 1. The trainer logged neither NLL/KL separately nor RMSE, saved no history, and
   ran no reference baselines. The evaluator overwrote `results/figures/` on every run.

KL-driven posterior collapse was investigated and **ruled out**.

## Evidence

### E1. The checkpoint is from epoch 3 of a nominal 100-epoch run
`torch.load(...)['epoch'] == 3`, `val_loss == 7536.2`, config `epochs=100`, `max_train_batches=0`.

I reran the *unchanged* original script and config for 15 epochs (`scripts/02_train_model.py --epochs 15`):

| epoch | 1 | 3 | 5 | 10 | 15 |
|---|---|---|---|---|---|
| val loss | 7953 | **7465** | 6844 | 5485 | 4906 |

Validation loss decreases **monotonically**. If the original run had continued, the best-epoch checkpoint would
have been replaced at every epoch. A best epoch of 3 (val 7536, close to the reproduced 7465) therefore means
the run ended after about 3–4 epochs. File times agree: checkpoint 12:31, evaluation 12:38. Without
`torchdiffeq`, every epoch ran the pure-Python RK4 fallback, which makes interruption more likely.

### E2. The model predicts approximately the dataset mean (`diagnostics/diagnose_checkpoint.py`)

| quantity (300 test trajectories, normalised units unless stated) | value |
|---|---|
| RMSE, posterior-mean prediction (orig. units) | 0.852 |
| RMSE of predicting the constant 0 (= training mean) | 0.867 |
| prediction coordinate std vs truth std | 0.085 / 0.98 (x), 0.19 / 0.97 (y) |
| prediction temporal-variance ratio vs truth (full eval) | **≈ 1 %** |
| linear interpolation of the 30 observations, recon RMSE | **0.066** vs model 0.756 |

That matches the figures: small curves near the origin. Circle and spiral reconstruction errors are almost
identical (0.755 vs 0.756), which is expected because the model ignores the family.

### E3. KL / posterior collapse is not the cause
- Per-trajectory KL = **3.7 nats** vs Gaussian NLL = **7475 nats**. With σ_obs = 0.05 (0.056 normalised), each
  unit of squared error costs about 160 nats per coordinate, so the reconstruction term dominates by more than 3 orders
  of magnitude.
- Posterior means differ between trajectories (std of μ across trajectories 0.8–1.25 per dimension), and
  posterior σ ≈ 0.7. That is not collapse onto the prior (which would give μ≈0, σ≈1, KL≈0).
- Posterior-mean vs sampled-posterior RMSE: 0.852 vs 0.855. Both are bad, for the same reason (undertraining).

So KL annealing cannot be the fix. It was still kept as an option and tested (see the results report).

### E4. Encoder, time handling, ODE and decoder are correct
- Encoder: observations and timestamps are flipped together; the GRU reads the newest observation first, so the
  final hidden state summarises the sequence at the first observation, which is always index 0 (t = 0). Inputs are `(x, y, t/T, Δt/T)`;
  the hidden state starts at zeros. Integration starts at `query_t[0] = 0`, which matches the time of z0. ✔
- Reparameterisation `z0 = μ + exp(½ logvar) ε`, and the KL is the analytic Gaussian KL. ✔ (Same as
  `torchdiffeq/examples/latent_ode.py`: backward RNN, N(0,I) prior, fixed-σ Gaussian NLL summed per
  trajectory, KL weight 1.)
- No future leakage: `prepare_batch` samples observations only from `t ≤ 7` (existing test plus a dataset validator).
- New tests: the analytic rotation ODE is integrated on [0,10] with dopri5 and RK4 (error < 1e-3); integration
  starts at `times[0]` on a shifted grid; gradients are non-zero for **every** parameter through dopri5.
- **Single-trajectory overfitting (decisive):** with full clean supervision the *original architecture* reconstructs
  one clean circle with RMSE **0.0053** and one clean spiral with RMSE **0.026**, both including the t > 7 forecast.
  The whole pipeline can learn rotation and radial growth. See `results/debug_overfit/`.
- De-normalisation (`pred·scale + mean`) is applied before all metrics and plots, and the reconstruction and forecast
  intervals are split at t = 7 on the query grid. ✔

### E5. Latent dimension 4 is structurally insufficient
The latent ODE is autonomous, so a trajectory is a deterministic function of z0 ∈ ℝ^d. The set of
trajectories the model can produce is therefore at most d-dimensional. A spiral in this dataset has
**6** free parameters (r0, ω including its sign, phase, cx, cy, growth), and a circle has 5. With d = 4 the
model cannot represent the family exactly, however well it is trained. This motivates latent_dim ≥ 6 (8 used). It is
a capacity limit, not a code bug, and it is tested empirically in Phase 2C and the full runs.

### E6. Smaller defects found during the audit
| defect | effect | fix |
|---|---|---|
| `torchdiffeq` missing; silent RK4 fallback while the checkpoint records `dopri5` | irreproducible metadata, slow training | installed `torchdiffeq 0.2.5`; checkpoint now stores `solver_backend`; `--require-torchdiffeq` flag |
| trainer logged only the total loss | collapse and undertraining invisible | per-epoch CSV with NLL, KL, β, LR, gradient norms, val context/forecast RMSE |
| no early stopping, schedule or annealing options | — | `--patience`, `--lr-schedule cosine`, `--kl-anneal-epochs` |
| `03_evaluate_model.py` always overwrote `results/figures/*` | figures from different checkpoints mixed up | `--figures-dir` option |
| no reference baselines | no sanity floor | constant-position / constant-velocity / interpolation baselines in `06_full_evaluation.py` |

Gradient clipping at norm 1.0 is always active (pre-clip norms are about 10³–10⁴ because the NLL is summed).
With Adam this only changes the step scale between batches. It is not a correctness bug, and it was kept.
