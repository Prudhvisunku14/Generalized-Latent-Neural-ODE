# Pretrained models

Frozen, verified copies of the trained Latent Neural ODE checkpoints. Nothing here is retrained; every
`best_model.pt` is **byte-identical (SHA-256)** to the original file in `checkpoints/`, which is left untouched.
Use them with `scripts/predict_trajectory.py` (see `docs/INFERENCE_GUIDE.md`).

## Available models

| CLI name (`--model`) | Folder | Trained on | Parameters | Seed | Epoch | Original checkpoint |
|---|---|---|---|---|---|---|
| `three_shapes_unified` (default) | `three_shapes_unified/` | circles, spirals, ellipses | 24,602 | 42 | 298 | `checkpoints/validation/three_shapes_seed42.pt` |
| `two_shapes_unified` (M1) | `two_shapes_unified/` | circles, spirals | 24,602 | 2024 | 288 | `checkpoints/validation/two_shapes_seed2024.pt` |
| `circle_specialist` | `specialists/circle/` | circles only | 24,602 | 42 | 596 | `checkpoints/validation/circle_only_seed42.pt` |
| `spiral_specialist` | `specialists/spiral/` | spirals only | 24,602 | 42 | 585 | `checkpoints/validation/spiral_only_seed42.pt` |
| `dynamics_conditioned` (M3, experimental) | `experimental/dynamics_conditioned/` | circles, spirals | 25,118 | 2025 | 299 | `checkpoints/conditioning/m3_conditioned_seed2025.pt` |
| `variable_nobs_experimental` (experimental) | `experimental/variable_nobs/` | circles, spirals; random 10–50 observations per batch | 24,602 | 42 | 299 | `checkpoints/validation/ablation_variable_nobs_seed42.pt` |
| `five_shapes_unified` | `five_shapes_unified/` | circles, spirals, ellipses, figure-eights, sinusoids | 24,602 | 42 | 297 | `checkpoints/five_shapes/five_shapes_seed42.pt` |
| `ablation_2x2_A` (2×2 ablation) | `ablation_2x2/A_*/` | five families, `--data-seed 42` | 24,602 | 42 | 294 | `checkpoints/ablation_2x2/five_shapes_A_lat8_gru64_seed42.pt` |
| `ablation_2x2_B` (2×2 ablation) | `ablation_2x2/B_*/` | five families, `--data-seed 42` | 27,186 | 42 | 299 | `checkpoints/ablation_2x2/five_shapes_B_lat16_gru64_seed42.pt` |
| `ablation_2x2_C` (2×2 ablation) | `ablation_2x2/C_*/` | five families, `--data-seed 42` | 63,642 | 42 | 299 | `checkpoints/ablation_2x2/five_shapes_C_lat8_gru128_seed42.pt` |
| `ablation_2x2_D` (2×2 ablation) | `ablation_2x2/D_*/` | five families, `--data-seed 42` | 67,250 | 42 | 296 | `checkpoints/ablation_2x2/five_shapes_D_lat16_gru128_seed42.pt` |

Each folder contains `best_model.pt`, `model_config.json` (architecture), `preprocessing.json` (normalisation and
time conventions) and `metadata.json` (provenance, checksum, validation and test performance). Everything was found; no
checkpoint is missing and nothing had to be retrained. `five_shapes_unified` and the four `ablation_2x2_*` models were added later by `scripts/package_five_shapes_model.py` (byte-identical copies; single training seed; each passed a load + CLI smoke test with finite 200-point output on a held-out example; those smoke tests are not accuracy evaluations). `selection_report.json` lists every candidate that was scored and
every checkpoint that was excluded (with the reason), and `verification_report.json` records the load checks.

### How the checkpoint for each model was chosen
Several checkpoints exist for most models (different seeds, studies). Candidates were identified from their **stored
configuration**, not file names, and scored on the **validation split** with identical observation subsets; the lowest
validation −ELBO won. This is why the two-shape model is the seed-2024 run, **not** the seed-42 "Model D" reported earlier
(its seed-42 sibling scored worse on validation: −18.6 vs −24.3). The stored training-time validation losses were not
used because each seed drew different observation subsets. Where the −ELBO winner is not also the best validation
*forecast* RMSE in its pool, `metadata.json` says so (`spiral_specialist` and `dynamics_conditioned`). The three-shape and
variable-count models each have a single checkpoint, so nothing was selected for them. Excluded: the epoch-1/3 smoke or
under-trained originals, the original-architecture and other hyper-parameter runs, and byte-identical duplicates.

## Architecture (all unified/specialist models)
`GRU encoder (4→64, reads the time-reversed sequence)` → `Linear 64→16 = (μ, log σ²) of z₀ ∈ ℝ⁸` → `dz/dt = f_θ(z)`,
tanh MLP 8→64→64→8 → `RK4, step 0.1, from t = 0` → `decoder`, tanh MLP 8→64→64→2. The conditioned model adds a
64→4 `tanh` head giving a constant dynamics code c and uses f_θ(z, c) (MLP 12→64→64→8). Parameter counts are checked on
every load. Layer-by-layer detail is in each `model_config.json`.

## Preprocessing (also in each `preprocessing.json`)
* Input columns `t, x, y`. Coordinates: `(xy − mean) / scale`, a single scalar `scale` (RMS deviation) with `mean` and `scale` computed on the
  **training split** of the model's training dataset (two-shape: scale 0.8903; three-shape: 0.8526). Outputs are mapped back with `xy·scale + mean`.
  A model must always use its own constants; the loader does this automatically.
* Encoder features per observation: `[x_norm, y_norm, t/T, Δt/T]`, T = 10, processed newest-to-oldest, hidden state initialised to zero.
  On the reversed sequence the gap is `t_rev[i−1] − t_rev[i]` (positive), with gap 0 for the first element.
* The latent state is the **posterior mean** z₀ = μ; the ODE is integrated from **t = 0** (the time of z₀; every training
  trajectory had an observation at t = 0).
* Time: context (observations) 0 ≤ t ≤ 7; trained and validated output range 0 ≤ t ≤ 10 (forecast 7 < t ≤ 10).

## Training assumptions and limitations (read before relying on a prediction)
* **Exactly 30 observations were used in training** (the experimental variable-count model used a random 10–50).
  Variable-length input is *technically* supported (the GRU accepts any length), but accuracy collapses away from 30. On held-out test
  trajectories with the three-shape model (original units):

  | n observations | 10 | 20 | **30** | 40 | 50 |
  |---|---|---|---|---|---|
  | reconstruction RMSE | 0.997 | 0.418 | **0.081** | 0.294 | 0.481 |
  | forecast RMSE | 1.544 | 0.895 | **0.203** | 0.666 | 1.050 |
  | simple baseline: interpolate / hold last point (recon, forecast) | 0.267, 1.241 | 0.107, 1.200 | 0.064, 1.174 | 0.048, 1.157 | 0.045, 1.149 |

  Away from 30 observations these models are **several times worse than a trivial interpolation for reconstruction**, and the forecast is worse than
  holding the last point at 10 observations. Even at 30, the three-shape model's reconstruction (0.081) is not better than interpolation (0.064); its
  value is the forecast. Use `variable_nobs_experimental` if you must use other counts, accepting about twice the forecast error at 30 (0.386 vs 0.165–0.203).
* Forecast accuracy decays with horizon (see the validation reports); expanding or large spirals are hardest. The model under-estimates spiral growth.
* **No shape classification.** The unified models infer dynamics implicitly from coordinates; they output no shape name and none should be
  inferred from them. A specialist is only meaningful for its own family.
* The first observation should be at t = 0; observations after t = 7 are rejected (they would be forecast-interval data).
* Not validated: coordinate scales/units different from training (spiral radius up to ≈ 4.7), more than one trajectory per file, 3-D data,
  `--forecast-end` beyond 10 (warned), and shapes other than circles, spirals and ellipses.
* The three-shape and variable-count models are single-seed; seed-to-seed spread of the unified model is large (forecast RMSE SD ≈ 0.05
  over 5 seeds; see `reports/validation_study_report.md`).

## Performance on the held-out test split (30 observations; recorded after selection, not used for it)
Original coordinate units, posterior mean. Full tables are in each `metadata.json`.

| Model | Recon RMSE | Forecast RMSE | Forecast FDE |
|---|---|---|---|
| `three_shapes_unified` (450 test: circle / spiral / ellipse) | 0.082 (0.058 / 0.094 / 0.093) | 0.211 (0.137 / 0.298 / 0.198) | 0.372 |
| `two_shapes_unified` (300 test) | 0.055 | 0.169 | 0.332 |
| `circle_specialist` (150 circles) | 0.045 | 0.085 | 0.147 |
| `spiral_specialist` (150 spirals) | 0.063 | 0.194 | 0.376 |
| `dynamics_conditioned` (300 test) | 0.051 | 0.212 | 0.426 |
| `variable_nobs_experimental` (300 test) | 0.057 | 0.389 | 0.932 |

These are single checkpoints; do not read them as seed-averaged performance.

## Integrity and re-building
`metadata.json → sha256` is checked on every load; a mismatch refuses to load. To rebuild this folder from `checkpoints/`:
`python scripts/package_models.py --force`.

## Inventory: what is and is not published on GitHub
| Models | Published? |
|---|---|
| `three_shapes_unified`, `two_shapes_unified`, `circle_specialist`, `spiral_specialist`, `dynamics_conditioned`, `variable_nobs_experimental`, `five_shapes_unified`, `ablation_2x2_A` to `D` (10 models) | **Yes**, ~0.1 to 0.3 MB each; no Git LFS needed |
| Capacity-scaling `medium`/`large` models | Training checkpoints only (`checkpoints/capacity_scaling/`); no packaged inference folder |
| GeoLife-trained models (`saved_models/geolife_trained/`: from scratch, fine-tuned) | **No.** Kept locally. They are derived from the GeoLife dataset (Microsoft Research non-commercial licence) and the redistribution question is open. Recommended distribution: retrain locally from your own GeoLife download with `scripts/22_geolife_training_data.py` and `scripts/23_geolife_trained_models.py`, or ask the licence holder before releasing weights. |

The inference CLI uses these folders; the repository keeps this folder name (`saved_models/`) instead of `models/` so that every script, test and checksum record stays valid.
