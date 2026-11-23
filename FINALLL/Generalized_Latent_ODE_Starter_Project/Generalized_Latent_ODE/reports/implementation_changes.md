# Implementation changes (2026-11-15)

All changes are backward compatible. The original checkpoint loads and evaluates unchanged
(evaluation reproduces recon RMSE 0.7556 and forecast RMSE 0.8428 exactly). Originals of `checkpoints/`, `results/` and `configs/` are copied to
`backups/original_2026-11-15/`.

Changes fall into three groups. **Bug/defect fixes** correct something that was wrong. **Tooling** adds
diagnostics. **Architecture** is a deliberate model change, kept separate from the fixes.

## Environment (fix)
- Installed `torchdiffeq==0.2.5` (already listed in `requirements.txt`). The original run had silently used
  the offline RK4 fallback.

## `src/glo/model.py`
- *Architecture (opt-in):* `mlp()` helper plus `dynamics_layers` and `decoder_layers` arguments. With `n=1` the module layout
  and `state_dict` keys are identical to the original (checked by a test).
- *Tooling:* `rtol`, `atol` and `step_size` are stored on the model and passed to the solver, so a checkpoint
  records exactly how it was integrated.
- The encoder, reparameterisation, KL and decoder maths are **unchanged**. The audit found them correct
  (`reports/reconstruction_failure_analysis.md`, E4).

## `src/glo/solver.py`
- `integrate(..., step_size=None)` passes `options={'step_size': ...}` to fixed-step torchdiffeq methods,
  and to the offline RK4 fallback as `max_step`.

## `scripts/02_train_model.py` (defaults reproduce the original behaviour)
- *Fix:* the checkpoint config now stores `solver_backend` (the real backend) and `n_parameters`. The
  `--require-torchdiffeq` flag refuses to fall back silently.
- *Tooling:* per-epoch CSV history with β, LR, train NLL, KL and loss, validation ELBO, NLL and KL, validation
  context and forecast RMSE (clean **validation** targets, never test), mean and max pre-clip gradient norm,
  NFE per batch and wall time.
- *Training options:* `--kl-anneal-epochs` (linear β warm-up), `--lr-schedule cosine`, `--min-lr`,
  `--patience` (early stopping), `--grad-clip`, `--dynamics-layers`, `--decoder-layers`, `--solver rk4
  --step-size`, `--rtol/--atol`, `--device`, `--threads`.
- Checkpoint selection uses the validation **full ELBO (β = kl_weight)** with posterior-mean z0 and fixed
  observation subsets, so it is comparable across annealing epochs.

## `scripts/03_evaluate_model.py`
- *Fix:* `--figures-dir` option. It previously always overwrote `results/figures/`.

## New: `scripts/06_full_evaluation.py`
Full test-split evaluation comparing the improved and original checkpoints on **identical observation subsets**:
per-family metrics, constant-position, constant-velocity and interpolation baselines, posterior mean vs K sampled
posteriors, prediction vs truth variance, geometric checks against the known generating parameters,
an RK4 vs tight-dopri5 solver cross-check, and all figures.

## New diagnostics
- `diagnostics/diagnose_checkpoint.py`: posterior statistics, KL vs NLL, variance collapse, dynamics magnitude.
- `diagnostics/overfit_tests.py`: Phase 2 single-trajectory and 10+10 overfitting tests.

## Tests (`tests/test_model.py`, 3 added; 10/10 pass)
- `test_harmonic_oscillator_analytic_dopri5_and_rk4`: exact rotation on [0,10]; integration starts at `times[0]`.
- `test_gradients_reach_every_module_through_dopri5`: non-zero gradient for every parameter.
- `test_depth_one_matches_legacy_layout`: old checkpoints remain loadable.

## Solver choice for training
Training uses RK4 with step 0.1 (400 function evaluations per solve). Adaptive dopri5 needed a growing number of
evaluations as the dynamics sharpened, which made a 3000-iteration 20-trajectory run take more than 1 h. RK4's
accuracy is checked twice: by the analytic test, and by re-integrating the trained model with dopri5
(rtol 1e-6) in `06_full_evaluation.py`. For these small networks CPU (4–6 threads) was about 20× faster than a shared GPU.

---

# Validation study changes (2026-11-15, second session)

The Model D architecture and its training procedure are **unchanged**. Every validation run uses
`configs/model_d.yaml`, which matches the stored arguments of the Model D checkpoint except for the
CPU thread count (4 vs 6). Only experiment keys (seed, families, dataset, epochs, checkpoint path) are set on
the command line, in `scripts/run_validation_experiments.sh`.

## Training script (`scripts/02_train_model.py`)
- `--config FILE.yaml`: YAML defaults with key validation. CLI flags override it.
- `--families ...`: train and validate on a subset of families (for the specialist models).
- `--overwrite`: the trainer now **refuses to overwrite any existing checkpoint** without it. This protects
  the original and the completed two-shape checkpoints.
- `--n-obs-min/--n-obs-max`: optional random number of observations per *training* batch (validation
  keeps `--n-obs`). Used **only** in the clearly labelled ablation
  `checkpoints/validation/ablation_variable_nobs_seed42.pt`, not in Model D or any requested phase. It was
  motivated by a validation-split sweep showing the encoder only works near n_obs = 30.

## `src/glo/pipeline.py`
- `TrajectoryDataset(..., families=None)`: optional family filter. Normalisation constants are unchanged.

## New `src/glo/evaluation.py`
- `collect()`: per-trajectory predictions and metrics. It **always normalises with the model's training-dataset
  statistics**, which is required for out-of-distribution (two-shape model on ellipses) and cross-dataset comparisons.
- `BaselineModel`: constant position and constant velocity with linear interpolation inside the context. These run through
  the same pipeline as the models. *Definition change vs `06_full_evaluation.py`:* extrapolation now starts at the last
  observation time instead of t = 7, so the few context points after the last observation are extrapolated, not
  held. Constant-velocity recon RMSE changes from 0.0662 to 0.0659; forecast metrics are identical.
- `t_ci95` (Student-t interval over runs), `geometry`, `horizon_error`.

## Figures
- `06_full_evaluation.py`: the suptitle overlap is fixed (`tight_layout(rect=...)`). `results/improved_two_shapes/` was
  regenerated. `evaluation.json`, `baseline_comparison.json` and `per_trajectory_metrics.csv` are **byte-identical**
  to the backup in `backups/improved_two_shapes_before_title_fix/`.

## New scripts
`07_seed_study.py`, `08_forecast_analysis.py`, `09_shared_vs_separate.py`, `10_three_shapes_and_ood.py`.

## Tests (16 pass)
- `tests/test_objective.py`: the objective equals −ELBO computed independently with `torch.distributions`;
  lower is better; it can be legitimately negative.
- `tests/test_evaluation.py`: family filter, baseline correctness, t-interval vs scipy, and that `collect` normalises with the
  model's training statistics.

---

# Conditioning study changes (2026-11-15, third session)

- **No new architecture code was needed for M3.** `LatentODE(cond_dim=4)` already implements
  dz/dt = f_θ(z, c). The encoder emits c = tanh(W h) from the same GRU summary as (μ, log σ²), and the solver
  integrates [z, c] with dc/dt = 0 (verified by `tests/test_conditioning.py`). c is deterministic and has no KL term.
- `create_loader(..., generator=None)` and `02_train_model.py --data-seed`: dedicated generators for batch
  order and training observation subsets, so models with different parameter counts see **identical training
  data** at the same seed (verified: byte-identical batches and observation indices). The old runs did not use it and
  are unchanged.
- Configs `configs/conditioning/{m1_unified,m2_param_matched,m3_conditioned,m2aug_latent12}.yaml`. Each differs from
  `configs/model_d.yaml` only in the shown key plus `threads: 1`. Runner: `scripts/run_conditioning_study.sh`.
- `glo.evaluation.encode/decode`: obtain (μ, log σ², c) with the same observation subsets as `collect`, and decode from
  chosen (z0, c), for the code-swap and sensitivity analyses. Verified: `decode(encode(...))` equals `collect()` exactly.
- New `scripts/12_conditioning_study.py`; tests `tests/test_conditioning.py` (19 tests pass in total).

---

# Pretrained inference system (2026-11-15, fourth session)

No model, training code or checkpoint was changed. All 44 checkpoint files have the same SHA-256 before and after (checked by
`scripts/validate_inference_system.py`).

- `scripts/package_models.py`: finds candidates from stored configuration, scores them on the validation split with identical observation subsets,
  copies the winner into `saved_models/` (SHA-256 checked identical), writes `model_config.json`, `preprocessing.json`, `metadata.json`,
  `selection_report.json`, `verification_report.json`. Selected: `two_shapes_seed2024` (not the earlier seed-42 "Model D"), `three_shapes_seed42`,
  `circle_only_seed42`, `spiral_only_seed42`, `m3_conditioned_seed2025`, plus the variable-count ablation as an extra experimental model.
- `src/glo/inference.py`: loading with checksum/state-dict/parameter checks (`torch.load(weights_only=True)`), CSV validation, deterministic prediction under
  `inference_mode`, metrics (reuses `pipeline.infer_test_metrics`), figures. It calls `model.encoder` directly with the **training value T = 10**.
  `LatentODE.forward` derives T from the last query time, so using it with another `--forecast-end` would silently change the encoder input. The solver
  grid is anchored at t = 0 and t = 10, so a prediction at a given time does not depend on the other requested times (tested).
- `scripts/predict_trajectory.py` (CLI), `scripts/make_example_inputs.py`, `scripts/validate_inference_system.py`.
- Tests: `tests/test_inference.py` (32 tests; 51 in total). A mutation check confirmed the consistency test fails for a wrong T (2.5), a 1% scale error (0.036) or a shifted mean (0.008),
  whereas correct inference differs from the evaluation pipeline by about 3e-6.
- Docs: `saved_models/README.md`, `docs/INFERENCE_GUIDE.md`.

---

# GeoLife synthetic-to-real transfer (2026-11-15, fifth session)

No pre-existing checkpoint, saved model, synthetic dataset or earlier result was modified. Every stage of `scripts/21` and `scripts/23`
checks the SHA-256 of the 44 pre-existing checkpoint files (plus `saved_models/*/best_model.pt`) before and after it runs.

- New data module `src/glo/geolife.py`: `.plt` parsing (format verified on all 18,670 files), GMT timestamps cross-checked with the day-number field,
  invalid-coordinate and time-order cleaning, local azimuthal equidistant projection (WGS-84 Gaussian radius), run splitting (gap > 60 s,
  speed > 70 m/s), 200-record segments, sparse-observation tasks, and per-segment similarity transforms fitted on the observations only.
- New `src/glo/geolife_evaluation.py`: linear / not-a-knot cubic / PCHIP baselines (extrapolation refused), hold-last and constant-velocity
  forecast baselines, metrics in metres (the pipeline's RMSE definition), trajectory descriptors, and a user-cluster bootstrap. The Latent ODE is called only
  through the unchanged `glo.inference.predict`.
- `src/glo/inference.py`: added `load_model_folder()` (the same verification as `load_packaged_model`, for folders outside `MODEL_REGISTRY`);
  `load_packaged_model` now delegates to it. The registry and the behaviour for existing models are unchanged, and all existing tests pass.
- `scripts/02_train_model.py`: opt-in `--data-root` and `--init-checkpoint` (fine-tuning; the architecture must match) and the experiment name
  `geolife_regular`. Defaults are unchanged.
- `tests/test_inference.py::test_cli_batch_mode` now batches copies of the three example files in a temporary folder. User-added files in
  `input_trajectories/` (e.g. a 200-point ground-truth CSV with t > 7, which the CLI correctly rejects) had made it fail. The files were left in place.
- New scripts `20_inspect_geolife.py`, `21_evaluate_geolife_transfer.py`, `22_geolife_training_data.py`, `23_geolife_trained_models.py`;
  config `configs/geolife.yaml`; tests `tests/test_geolife.py` (20 tests, plus a user-split disjointness check).
- New checkpoints (Phase 11, new files only): `checkpoints/geolife/geolife_scratch_seed42.pt`,
  `checkpoints/geolife/geolife_finetune_from_three_shapes_seed42.pt`; packaged copies in `saved_models/geolife_trained/`.
- `.gitignore`: GeoLife archive and GeoLife-derived data and per-segment CSV files (the MSR-LA licence forbids redistribution).
- Reports: `reports/geolife_dataset_audit.md`, `reports/geolife_real_world_interpolation_report.md`.
