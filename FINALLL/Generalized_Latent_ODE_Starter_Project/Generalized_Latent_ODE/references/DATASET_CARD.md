# Dataset card: 2-D continuous-time synthetic trajectories

## Purpose

A controlled research dataset for learning irregularly observed continuous-time trajectories with a single Latent Neural ODE, initially from circles and logarithmic spirals and then adding ellipses.

## Sizes and splits

- Two-shape: 2000 examples (1400 train, 300 validation, 300 test).
- Three-shape: 3000 examples (2100 train, 450 validation, 450 test).
- Equal 700/150/150 split within each family.
- 200 equally spaced **ground-truth reference timestamps** between `t=0` and `t=10`.
- Default observations: 30 randomly selected timestamps in `[0,7]` (always including `t=0`), with unequal gaps and Gaussian coordinate noise (`sigma=0.05` in raw coordinate units).
- The stored `default_obs_mask` supplies one reproducible observation subset; training scripts generate additional irregular masks on the shared query grid.
- Seed 42 and deterministic per-family RNG streams. Per-trajectory IDs and exact sampled parameters saved.

## Parameter ranges (from code)

- All families: `omega` randomly positive/negative with absolute value from 0.75 to 1.45; `phase` in `[-pi,pi]`, `cx,cy` in `[-0.35,0.35]`.
- Circle: radius in `[0.55,1.5]`.
- Logarithmic spiral: initial radius in `[0.55,1.5]`, growth `k` sampled from `[0.055,0.12]` in magnitude with random sign; radius evolves as `r0*exp(k*t)`.
- Ellipse: major semiaxis `a` in `[0.85,1.65]`, minor semiaxis `b/a` in `[0.4,0.78]`, rotation angle in `[0,pi]`.

## Stored arrays

Each NPZ includes `time (200,)`, `xy (N,200,2)`, `noisy_xy (N,200,2)`, `family (N,)`, `ids (N,)`, `params (N,9)` and `default_obs_mask (N,200)`.

Do not use `params` or `family` as inputs to the baseline model: doing so would provide labels unavailable to the intended inference protocol. The model must infer dynamics from noisy timestamped positions.

## Known limitations

- All trajectories are synthetic, analytically generated, smoothly sampled, and two-dimensional.
- These are **not** the exact 2018 paper's source spiral curves or its training protocol.
- The training families are limited to three and are controlled by simple parameter distributions. Good interpolation within known families is **not** evidence of generalization to unseen families.
- When testing forecasting, no future values are supplied to the encoder; if `--supervision full_clean` is used, synthetic future targets are included in the **training loss**, and such experiments must be described as supervised.
- Dataset generation uses equal-time *reference targets*; the model's observation masks are sparse/irregular. This is intentional to permit unambiguous time-aligned evaluation.

## Reproducibility

- Generator: `src/glo/data.py`.
- Script: `scripts/01_generate_dataset.py`.
- Validator: `scripts/04_validate_dataset.py`.
- Source and methodological cautions: `references/SOURCES_AND_METHODS.md`.
