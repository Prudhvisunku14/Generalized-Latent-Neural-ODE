# Pretrained inference guide

Give the system sparse, noisy, time-stamped 2-D coordinates of an unknown trajectory. A frozen pretrained Latent Neural ODE
reconstructs the trajectory for 0 ≤ t ≤ 7 and forecasts it for 7 < t ≤ 10. No training happens, and you never state a shape.

All commands run from the project root in the VS Code terminal (`Generalized_Latent_ODE/`) with the project's Python environment
(`torch`, `torchdiffeq`, `numpy`, `matplotlib`).

## 1. Prepare the trajectory CSV

A header row `t,x,y` and one observation per row:

```csv
t,x,y
0,-0.50082618,1.33941448
0.402010053,-0.874465644,0.841090143
0.703517616,-1.06693256,0.379560024
...
```

A complete valid 30-observation file is `input_trajectories/example_circle.csv` (also `example_spiral.csv` and `example_ellipse.csv`).

| Requirement | What happens if it is not met |
|---|---|
| Columns `t`, `x`, `y` (case-insensitive, any order) | error naming the missing columns |
| Every value numeric and finite | error with the line number and column |
| Timestamps unique | error |
| Timestamps ≥ 0 and **≤ 7** (the context interval) | error. Data after t = 7 is forecast-interval data and is never given to the model |
| Rows sorted by time | rows are sorted automatically (warning); x, y stay attached to their t |
| At least 2 observations | error; fewer than 10 gives a warning |
| An observation at t = 0 | warning if missing (every training trajectory had one) |
| 30 observations | warning if different: see §6 |
| Coordinates in the training units/scale | warning if they exceed 1.5× the training range |
| Extra columns (e.g. `family`) | ignored, with a warning. They are never read |

Use the same units as the training data (a unit-scale 2-D plane; circles/spirals/ellipses with radius ≈ 0.4–1.7, noise σ ≈ 0.05).
Observation times need not be regular.

## 2. Choose a pretrained model

```bash
python scripts/predict_trajectory.py --list-models
```

| `--model` | Use it when |
|---|---|
| `three_shapes_unified` (default) | You do not know the shape. Trained on circles, spirals and ellipses |
| `two_shapes_unified` | Circles/spirals only (slightly more accurate on those than the three-shape model) |
| `circle_specialist`, `spiral_specialist` | You know the family; valid only for that family |
| `dynamics_conditioned`, `variable_nobs_experimental` | Experiments; see `saved_models/README.md` |

If the requested model is not packaged, the program stops with an error. It never silently substitutes another model.

## 3. Run inference

```bash
python scripts/predict_trajectory.py \
    --model three_shapes_unified \
    --input input_trajectories/example_circle.csv \
    --n-query 200 \
    --forecast-end 10 \
    --output inference_outputs/circle_example
```

Options (`--help` explains each):

| Option | Meaning |
|---|---|
| `--model` | pretrained model (default `three_shapes_unified`) |
| `--input` | one or more CSV files, or a directory (every `*.csv` is processed: batch mode) |
| `--output` | output directory (default `inference_outputs/<csv name>`; in batch mode one sub-folder per CSV) |
| `--n-query` | number of evenly spaced output times on [0, forecast-end] (default 200) |
| `--forecast-end` | last output time (default 10). Values above 10 extrapolate beyond the trained horizon and warn |
| `--ground-truth` | optional dense `t,x,y` CSV for scoring (one input file) |
| `--ground-truth-dir` | batch version: a folder with files named like the inputs |
| `--device`, `--threads` | default CPU, 1 thread (bit-reproducible) |
| `--no-plots`, `--quiet`, `--list-models` | as named |

Batch example, scoring against the ground truth of the three provided examples:

```bash
python scripts/predict_trajectory.py --input input_trajectories/example_circle.csv input_trajectories/example_spiral.csv \
    input_trajectories/example_ellipse.csv --output inference_outputs/batch --ground-truth-dir input_trajectories/ground_truth
# (--input also accepts a directory; every *.csv in it must then be a valid observation file with t <= 7)
```

## 4. Outputs

In the output folder:

| File | Content |
|---|---|
| `predicted_trajectory.csv` | `t,predicted_x,predicted_y,region` with `region` = `reconstruction` (t ≤ 7) or `forecast` (t > 7) |
| `reconstruction_forecast.png` | x–y plot, equal axes. Black dots: observations, blue: reconstruction, red: forecast, green: ground truth (only if supplied) |
| `x_vs_time.png`, `y_vs_time.png` | coordinate against time, forecast boundary marked at t = 7 |
| `run_summary.json` | model, checkpoint SHA-256, observation count, latent posterior mean, warnings, `shape_label_used: false` |
| `metrics.json` | **only if a ground truth was supplied**: reconstruction RMSE, forecast RMSE, MAE, ADE, FDE, Chamfer distance |

Open the PNGs in VS Code (click the file), or `xdg-open inference_outputs/circle_example/reconstruction_forecast.png`.

Delivered example results (three-shape model, held-out test trajectories; ground truth in `input_trajectories/ground_truth/`):

| Example | Reconstruction RMSE | Forecast RMSE | Forecast FDE |
|---|---|---|---|
| `circle_example` | 0.046 | 0.082 | 0.159 |
| `ellipse_example` | 0.034 | 0.098 | 0.110 |
| `spiral_example` | 0.133 | 0.332 | 0.560 |

The spiral is a deliberately unlucky case (an expanding spiral, whose growth the model under-estimates). It is one trajectory, not a benchmark:
in the observation-count study over the whole three-shape test split (450 trajectories, 30 random observations each) the mean forecast RMSE is 0.131 for circles,
0.293 for spirals and 0.184 for ellipses.

## 5. Interpreting reconstruction and forecast

* **Reconstruction (blue, t ≤ 7)** is a smooth model of the trajectory, not an interpolant: it does not pass through every noisy dot by design.
* **Forecast (red, t > 7)** continues the motion implied by the inferred dynamics. Its error grows with the horizon
  (for the two-shape models in the validation study: pointwise error about 0.10 at t = 7 rising to about 0.3 at t = 10 for circles, about 0.15 to 0.5 for spirals) and
  typically appears as a phase lead or a drifting radius.
  Compare the red line against the `x_vs_time` plot to see whether an error is in timing or in amplitude.
* The model outputs no shape name, and none should be read from it. Do not interpret the latent values in `run_summary.json` as a classification.
* Metrics, when a ground truth is supplied, use the project's evaluation definitions (reconstruction RMSE excludes observed times;
  forecast metrics use t > 7; units are the input coordinate units). The ground truth is read only after the prediction is made and never reaches the model.

## 6. Variable numbers of observations: supported, but not accurate

The GRU encoder accepts any sequence length, so files with 10, 20, 40, 50 or more rows run without error. **That is technical support only.** The unified
models were trained with exactly 30 observations and their accuracy degrades sharply otherwise. Held-out test trajectories, three-shape model, original units
(from `results/inference_validation/`, figure `observation_count_accuracy.png`):

| Observations | Reconstruction RMSE | Forecast RMSE | Baseline: interpolation / last-point hold (recon, forecast) |
|---|---|---|---|
| 10 | 0.997 | 1.544 | 0.267, 1.241 |
| 20 | 0.418 | 0.895 | 0.107, 1.200 |
| **30** | **0.081** | **0.203** | 0.064, 1.174 |
| 40 | 0.294 | 0.666 | 0.048, 1.157 |
| 50 | 0.481 | 1.050 | 0.045, 1.149 |

* At any count other than 30 the reconstruction is **several times worse than simply interpolating the observations**. The forecast is worse than holding the
  last point at 10 observations (1.544 vs 1.241) and only about 9% better at 50 (1.050 vs 1.149). The two-shape model behaves the same way.
* Even at 30 observations the three-shape model's reconstruction (0.081) is not better than interpolating the noisy points (0.064); the two-shape model's (0.055) is
  slightly better. The model's real advantage is the **forecast** (0.203 vs 1.174 for holding the last point), which interpolation cannot provide.
* If your data does not have 30 points, either **subsample or select 30 observations spread over 0 ≤ t ≤ 7**, or use `--model variable_nobs_experimental`.
  It reconstructs well at every count (recon 0.055–0.161 from 40 down to 10 observations) but its forecasts are about twice as inaccurate at 30 observations
  (RMSE 0.386, against 0.203 for the three-shape model) and it has been validated for 10–50 observations only.
* A variable-count training strategy that matches the fixed-count accuracy has *not* been developed. It would be a separate experiment, and no frozen model here was changed.

## 7. Reproducing the validation

```bash
python scripts/package_models.py --force         # rebuild saved_models/ from checkpoints/ (selection on validation data)
python scripts/make_example_inputs.py            # regenerate the example CSVs from held-out test trajectories
python scripts/validate_inference_system.py      # tests 1-8 (about 3 minutes)
python -m pytest -q tests                        # 51 tests
```

`results/inference_validation/validation_report.json` records: the three held-out examples; unknown-family-label runs (neutral file names, a `family` column
that is correct, wrong or absent gives identical predictions); the observation-count study; two identical runs (byte-identical outputs); consistency with the existing evaluation
pipeline for all six packaged models (maximum prediction difference 3.8 × 10⁻⁶, i.e. float32 rounding; the tests fail if it exceeds 10⁻⁴); and unchanged SHA-256 of all 44 checkpoint files
before and after every run.

## 8. Troubleshooting

| Message | Cause and fix |
|---|---|
| `missing required column(s)` | the header must contain `t`, `x`, `y` |
| `observation(s) after the context end t=7` | remove rows with t > 7; supply them via `--ground-truth` only if you want them scored |
| `duplicate ... timestamps` | each time may appear once |
| `Model '...' is not available ... NOT substituted` | the model is not in `saved_models/`; run `--list-models` or `python scripts/package_models.py` |
| `N observations supplied but this model was trained with 30` | warning only; see §6 |
| `--forecast-end ... exceeds the trained horizon` | warning only; predictions beyond t = 10 are unvalidated |
| `Checksum mismatch` | the weights file was altered; restore it from `checkpoints/` (`package_models.py --force`) |
