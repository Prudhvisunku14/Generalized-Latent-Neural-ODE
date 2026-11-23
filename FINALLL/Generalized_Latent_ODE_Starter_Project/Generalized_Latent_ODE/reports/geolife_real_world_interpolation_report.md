# Synthetic-to-real transfer of the Generalized Latent Neural ODE: interpolating real GeoLife GPS trajectories

**Result in one paragraph.** The pretrained three-shape Latent ODE (trained only on synthetic circles, spirals and ellipses) runs on
real GPS input without error, but it **does not transfer**. On 1,000 held-out GeoLife segments from 97 users, with 30 observations each,
its median interpolation error is **126.4 m**. Linear interpolation of the same 30 points gives **7.1 m**, a cubic spline **5.6 m** and
PCHIP **5.6 m**. Per segment, the Latent ODE error is on average 17.3× the linear error (95% user-cluster bootstrap CI 15.4–19.3) and
it beats linear interpolation on 1 segment out of 1,000. The failure holds for every trajectory type, speed, observation gap, transport
mode and observation count (10–50), and for every coordinate-normalisation strategy tried. Forecasting is also worse than a
constant-velocity extrapolation. Whether training the same architecture on real GPS helps (Phase 11, §9) is **not yet known**: those runs were unfinished when this report was published.

---

## 1. Research motivation

Earlier work in this project showed that one Latent Neural ODE can reconstruct and forecast synthetic circles, spirals and ellipses from
30 sparse noisy points, without being told the shape. Those families are produced by smooth, autonomous, mostly rotational dynamics.
Real movement is different: long straight stretches, abrupt turns at intersections, stops, speed changes, irregular sampling and GPS
noise. The question here is whether the dynamics learned on synthetic geometry are a useful prior for real movement, i.e. whether the
model can fill in missing GPS positions **without retraining**. We separate four claims: (1) the model *runs* on real input; (2) it
*interpolates* withheld positions accurately; (3) it *forecasts* future positions; (4) it does so *across* many trajectories and users.
A smooth output curve is evidence of none of (2)–(4).

## 2. Dataset

GeoLife GPS Trajectories 1.3 (Microsoft Research Asia; 182 users, 2007–2012, mostly Beijing), read in place from the original archive
(SHA-256 verified). The full audit is in `reports/geolife_dataset_audit.md`. In short:

| Quantity | Value |
|---|---|
| Users / trajectory files / data lines | 182 / 18,670 / 24,876,978 |
| Malformed lines / invalid coordinates / duplicate timestamps / backward timestamps | 0 / 1 / 698,900 / 0 |
| Clean records | 24,178,077 |
| Sampling interval 5th / 50th / 95th / 99th percentile | 1 / 2 / 5 / 20 s (96.4% of steps ≤ 5 s) |
| Eligible 200-record segments (rules in §3.1) | 75,100 from 141 users |
| Eligible segment duration / path length (median) | 6.6 min / 2.6 km |
| Development split (users / segments) | 42 / 300 |
| Held-out test split (users / segments / source files) | 97 / 1,000 / 699 |

Licence: MSR-LA, non-commercial, no redistribution of the data or derived data. All GeoLife-derived data files are kept out of git.

## 3. Experimental methodology

### 3.1 Preprocessing (Table 2: `results/geolife/tables/table2_preprocessing.md`)
* **Parsing and cleaning.** The 7-field records are parsed with GMT timestamps taken from the date and time strings (cross-checked against the
  fractional-day field: 0 disagreements). Invalid coordinates are dropped. A record is kept only if its time is strictly later than the previous kept record.
* **Continuous runs.** A run is broken at any gap > 60 s or step speed > 70 m/s (thresholds fixed from the audit before any model was run).
* **Segments.** 200 consecutive clean records inside a run, non-overlapping. A segment is rejected if no record is ≥ 100 m from its start
  (stationary), or if it has fewer than 60 context or 10 future records.
* **Projection.** Latitude/longitude are converted to local metres with a spherical azimuthal equidistant projection centred on the segment's
  first record, which is always an observation. The sphere radius is the WGS-84 Gaussian radius of curvature at that latitude
  (√(MN) = 6,374.4 km at Beijing). Distances from the origin are exact on that sphere. Compared with ellipsoidal distances, the
  sphere stretches north–south and shrinks east–west distances by up to ≈ 0.2% at 10 km (checked numerically; ≈ 0.005% on the diagonal). Every method's errors are
  measured in the same frame, so this changes all errors by at most 0.2% and cannot change any comparison. Forward/inverse round trip and distances are unit-tested.
* **Time (protocol A, primary).** t = 10 (t_real − t_start)/(t_end − t_start), keeping the irregular spacing. Context is t ≤ 7 and forecast is
  7 < t ≤ 10. Knowing t_end, the end of the planned window, is an **experimental assumption**. It reveals no position; it only fixes the time unit.
* **Time (protocol B, forecasting).** t = 7 (t_real − t_start)/(t_last_obs − t_start). No knowledge of the segment end is used.

### 3.2 Sparse-observation tasks
For each segment, 30 records are observed: the first record (t = 0, as in every training trajectory) plus 29 drawn uniformly from the other
context records (seed = 100000 + 1000003·N + segment index; identical for every method). **Interpolation targets** are all withheld
context records between the first and last observation: 105,037 recorded GPS positions over the test set. The 3,478 withheld context records
after the last observation would require extrapolation, so they are excluded from the interpolation score and reported here. **Forecast targets** are
the records with 7 < t ≤ 10 (protocol A), or every record after the last observation up to t = 10 (protocol B). Every method is scored at
the **recorded timestamps** of these records against the **recorded GPS positions**; no reference curve is ever interpolated.

### 3.3 Coordinate normalisation without leakage
The model expects synthetic coordinates: trajectories centred near the origin with a context RMS radius whose median is 1.00 (5th–95th
percentile 0.57–1.67; measured on the synthetic training split). Each GPS segment is mapped by a similarity transform fitted on the
**30 observed positions only**: centre = mean of the observations, scale = their RMS distance from it mapped to a target size. Predictions are mapped
back to metres with the exact inverse. Withheld and future records cannot influence the transform; a test corrupts every withheld record and
checks that all predictions are unchanged.

Five strategies were compared on the **development users only**, with the criterion (lowest median interpolation RMSE of the Latent ODE) fixed
in advance:

| Strategy | Centre | Spread → target | Dev median RMSE (m) | Dev win rate vs linear |
|---|---|---|---|---|
| S1 (a-priori default) | mean of observations | RMS → 1.00 (synthetic median) | 202.4 | 0% |
| S2 | mean of observations | RMS → 0.57 (synthetic 5th percentile) | 233.9 | 0% |
| **S3 (selected)** | mean of observations | RMS → 1.67 (synthetic 95th percentile) | **178.0** | 0% |
| S4 | first observation | RMS → 1.00 | 302.0 | 0% |
| S5 | mean of observations | max distance → 1.29 (synthetic median) | 216.7 | 0% |
| *linear interpolation (reference)* | | | *5.9* | |

The choice of strategy changes the error by less than a factor of 2, while the gap to linear interpolation is about 30×. Normalisation is not the cause of the failure.

**Sensitivity (development users, S3).** An interpolator that respected the geometry would give the same metric error under rotation,
translation and rescaling. Linear interpolation does exactly that. The Latent ODE does not:

* rotating the input by 45°, 90°, 180° or 270° changes a segment's error by a median 31%, 37%, 9% and 38%;
* translating the normalised input by 0.5, 1 or 2 units changes it by a median 6%, 17% and 69% (the median error rises from 178 to 303 m at 2 units);
* scaling the target size by 0.25, 0.5, 2 or 4× changes it by a median 57%, 20%, 35% and 75%. The selected size is the best of these.

Errors scale with segment size. Over path-length terciles the median error is 47 m, 174 m and 518 m, against 4.0 m, 7.6 m and 15.0 m for linear interpolation.
The relative error stays at about 0.07–0.08 of the path length in each tercile.

### 3.4 Methods compared (identical observations, identical target timestamps)
* **Linear**: x(t) and y(t) interpolated separately on elapsed seconds.
* **Cubic spline**: SciPy `CubicSpline` (not-a-knot) on the observations. It is never evaluated outside the observed time range; a query outside it raises an error.
* **PCHIP**: SciPy `PchipInterpolator`, a shape-preserving cubic.
* **Latent ODE**: `saved_models/three_shapes_unified/best_model.pt`, frozen (`eval()`, `inference_mode`, no optimiser, weights checksummed before
  and after every stage). It is called only through the existing verified `glo.inference.predict` with features [x, y, t/T, Δt/T], T = 10, and
  the posterior mean z₀, then queried at the actual withheld timestamps.
* **Forecast baselines** (secondary experiment only): hold the last observation; constant velocity from the last two observations.

### 3.5 Statistics
Errors are in metres: RMSE_xy = √(Σ(Δx² + Δy²)/(2N)), the project's definition, plus coordinate MAE, ADE (mean Euclidean error), and the
median, 90th-percentile and maximum Euclidean error. Segments of one user are not independent, so confidence intervals come from a
**user-cluster bootstrap** (2,000 resamples of users). Paired comparisons use per-segment RMSE: the geometric-mean ratio with a cluster-bootstrap
CI, the win rate, and a Wilcoxon signed-rank test with Holm correction over the five comparisons.

### 3.6 Order of work (preventing selection on the test set)
audit → thresholds fixed → user split → pilot (10 development segments, a-priori S1) → strategy selection and sensitivity (development
users) → **one** evaluation on the held-out test users → density study (test) → Phase 11 (§9). Nothing was tuned after looking at test results.

## 4. Model

The frozen `three_shapes_unified` checkpoint (seed 42, epoch 298, 24,602 parameters; SHA-256 recorded in its `metadata.json`) works as follows:
a reverse-time GRU encoder (4 → 64) feeds Linear 64 → 16, which gives the posterior mean and log-variance of z₀ ∈ ℝ⁸;
dz/dt = f(z) is a tanh MLP 8 → 64 → 64 → 8; RK4 integrates it with step 0.1 from t = 0; and a tanh MLP decoder 8 → 64 → 64 → 2 produces x̂(t).
It was trained on 2,100 synthetic trajectories with exactly 30 observations on t ≤ 7. The architecture was not changed.

## 5. Quantitative results: interpolation (primary)

**Pilot** (10 development segments, S1; `results/geolife/pilot/`). The Latent ODE was worse than linear interpolation on all 10 segments,
with errors of 63–595 m against 0.9–35 m. Before scaling up, this failure was checked to be real rather than a pipeline error. Synthetic
test trajectories were rotated, scaled to ~500 m and shifted, then sent through exactly the same GPS code path. They gave a median error of
0.068 synthetic units, against 0.075 for the direct model call, so the transform and inverse transform are correct.

**Main evaluation** (Table 3; 1,000 held-out segments, 97 users, 105,037 targets, 30 observations, strategy S3):

| Method | RMSE median [95% CI] (m) | RMSE mean (m) | ADE median (m) | p90 error median (m) | max error median (m) | win rate vs linear | time / segment |
|---|---|---|---|---|---|---|---|
| Latent ODE (frozen) | **126.4** [110.9, 147.9] | 272.7 | 148.8 | 286.6 | 443.7 | **0.1%** | 26 ms |
| Linear | 7.10 [5.80, 8.48] | 13.0 | 6.33 | 15.5 | 32.4 | — | 0.02 ms |
| Cubic spline | **5.56** [4.64, 6.90] | 11.7 | 4.92 | 12.3 | 28.2 | 68.6% | 0.2 ms |
| PCHIP | 5.62 [4.54, 6.69] | **10.1** | 4.97 | 12.4 | 26.7 | 85.9% | 0.2 ms |

| Paired comparison | Geometric-mean RMSE ratio [95% CI] | Win rate (first method lower) | Holm-adjusted Wilcoxon p |
|---|---|---|---|
| Latent ODE vs linear | 17.3 [15.4, 19.3] | 0.1% | 1e-164 |
| Latent ODE vs cubic | 21.6 [18.6, 25.0] | 0.8% | 6e-163 |
| Latent ODE vs PCHIP | 22.2 [19.4, 25.2] | 0.0% | 2e-164 |
| Cubic vs linear | 0.80 [0.77, 0.83] | 68.6% | 1e-38 |
| PCHIP vs linear | 0.78 [0.76, 0.80] | 85.9% | 3e-116 |

In model units (metres divided by the per-segment scale), the Latent ODE's median error is 0.571, against 0.038 for linear and 0.030 for cubic.
On synthetic test data the same model reaches about 0.07. The real-data error is therefore about 8× its synthetic error, and the per-segment
normalisation has not made the problem look synthetic to the model.

Figures: `fig2_gps_with_observations.png` (Figure 2), `fig3_method_comparison.png` (Figure 3: randomly drawn segments), `fig4_best_median_worst.png`
(Figure 4), `fig5_error_distribution.png` (Figure 5). Figure 4's best case (ranked by error in model units) is a slow, curved path at 0.65 m/s. Even
there the Latent ODE reaches 9.2 m against 2.8 m for linear. The median case is a nearly straight path and the worst a stop-and-go path (straightness 0.07).
In both the model draws loops, the motion it learned from circles and spirals.

### 5.1 By trajectory characteristic (Figure 7)
Descriptors are computed from the recorded context records and are used only to stratify the results:

* **Straightness** = net displacement / path length. *Nearly straight* if ≥ 0.9; otherwise *frequent turning* if ≥ 3 heading changes > 45°
  (headings over 25 m chords), else *curved*.
* **Speed bands** (mean speed): < 2.5, 2.5–10, > 10 m/s.

| Stratum (n) | Latent ODE | Linear | Cubic | PCHIP | LODE / linear (geo-mean) |
|---|---|---|---|---|---|
| Nearly straight (365) | 198.5 | 5.27 | 3.49 | 3.62 | 40.3× |
| Curved (339) | 91.0 | 5.80 | 4.64 | 4.39 | 14.4× |
| Frequent turning (296) | 75.0 | 10.67 | 10.88 | 9.25 | 7.5× |
| Slow < 2.5 m/s (311) | 35.5 | 4.29 | 4.15 | 3.79 | 8.6× |
| Medium 2.5–10 m/s (426) | 138.7 | 9.25 | 6.91 | 7.07 | 15.5× |
| Fast > 10 m/s (263) | 414.5 | 9.67 | 5.58 | 6.78 | 47.1× |
| Mean observation gap 4.1–8.7 s (335) | 131.9 | 3.59 | 2.18 | 2.42 | — |
| Mean observation gap 8.7–16.3 s (332) | 87.2 | 6.55 | 5.14 | 5.17 | — |
| Mean observation gap 16.3–33.2 s (333) | 208.9 | 22.05 | 17.42 | 16.62 | — |
| Labelled walk (21) / bike (16) / bus (39) / subway (17) / car (13) | 29 / 55 / 131 / 216 / 360 | 2.5 / 2.8 / 9.1 / 5.8 / 9.1 | | 2.6 / 2.8 / 5.8 / 2.8 / 7.2 | — |

All values are median RMSE in metres. The relative failure is largest on **straight, fast** movement (40–47×), which is the least like the
rotational training data, and smallest on **slow, turning** movement (7.5–8.6×). No stratum has a Latent ODE win rate above 0.4%.

### 5.2 Error versus observation gap (Figure 8)
For the baselines, the error at a target grows steadily with the length of the observation gap that contains it: Spearman ρ = 0.62 (linear),
0.67 (cubic) and 0.63 (PCHIP), from about 0.5 m in 4 s gaps to about 15–18 m in the longest decile. The Latent ODE error is almost unrelated
to the gap (ρ = 0.10) and stays around 100–200 m. Its error is not driven by missing data: it comes from a trajectory-level mismatch between the
dynamics the model generates and the actual path. The model does not use the nearby observations the way an interpolator must.

## 6. Comparison with conventional methods

* **Where the Latent ODE is better:** nowhere measurable. It wins on 1 of 1,000 test segments against linear and on 8 against cubic.
* **Where it is worse:** everywhere. The gap is largest for straight and fast motion (median 199–415 m against 4–10 m) and smallest for slow
  or turning paths (still 7–9×).
* **Among the baselines**, cubic and PCHIP beat linear interpolation (geometric-mean ratio 0.80 and 0.78; win rates 69% and 86%). PCHIP has the
  best mean and tail, because it does not overshoot at stops and sharp turns, where the not-a-knot spline occasionally does. Their median errors of
  about 2.5 m at the median target are close to GPS measurement noise, so the measured truth limits how small any error can be.
* **Cost:** the Latent ODE takes about 26 ms per segment on one CPU thread, against 0.02–0.2 ms for the baselines.

### 6.1 Observation density (Figure 6, Table 4)

| N observations | Latent ODE (fixed-30 model) | Latent ODE (variable-count model) | Linear | Cubic | PCHIP | LODE win rate vs linear |
|---|---|---|---|---|---|---|
| 10 | 190.3 | 137.6 | 20.2 | 20.8 | 17.5 | 0.1% |
| 20 | 129.8 | 143.8 | 10.8 | 9.0 | 8.7 | 0.0% |
| 30 | 126.4 | 148.0 | 7.1 | 5.6 | 5.6 | 0.1% |
| 40 | 146.6 | 155.5 | 4.8 | 3.6 | 3.6 | 0.1% |
| 50 | 176.9 | 156.4 | 3.8 | 2.9 | 2.9 | 0.0% |

(Median interpolation RMSE in metres, the same 1,000 test segments at every level.) The baselines improve steadily with more observations. The
fixed-30 model is best at 30 observations and gets **worse** with more data, repeating its synthetic behaviour. The experimental
variable-count model (trained on circles and spirals with 10–50 observations) is flat at about 140–156 m: it is insensitive to the
count, but it is not accurate. That a model accepts a different number of observations does not make it accurate at that number.

## 7. Forecasting (secondary; never combined with interpolation)

| Protocol | Targets | Median horizon | Latent ODE | Hold last | Constant velocity | LODE win rate vs constant velocity |
|---|---|---|---|---|---|---|
| A (known segment end) | 61,485 | 142 s | 702 m | 229 m | **151 m** | 8.2% |
| B (no end-time knowledge) | 57,556 | 124 s | 661 m | 199 m | **121 m** | 7.5% |

(Median per-segment forecast RMSE.) The Latent ODE forecast is 3–5× worse than the trivial baselines. Its forecasts typically curl back
(Figure 4, dashed), as a circle or spiral would. Removing the knowledge of the segment end (protocol B) does not change the conclusion.
Poor forecasting is reported separately from interpolation. Here both fail, for the same reason.

## 8. Generalisation assessment

| Claim | Verdict | Evidence |
|---|---|---|
| (1) Runs on real GPS input | **Yes** | 1,000 / 1,000 test segments, finite outputs, frozen weights verified unchanged |
| (2) Accurate interpolation of withheld positions | **No** | 126 m median vs 5.6–7.1 m for classical interpolation; 0.1% win rate |
| (3) Accurate forecasting | **No** | 3–5× worse than constant velocity |
| (4) Generalisation across trajectories and users | **No** | the failure is uniform across 97 users, every stratum, every N and every normalisation strategy |

The synthetic-trained model does **not** transfer to real GPS trajectories. The likely mechanism follows from the evidence, but it has not been
tested by an intervention. The encoder maps any 30 points to a latent state of a dynamical system that, by training, produces closed or
spiralling orbits with a smooth angular velocity. The model has no way to represent a straight line, a stop, or a sudden turn, so it produces
the nearest synthetic-looking orbit. That is why the error does not depend on observation spacing (§5.2), why it depends on rotation and
translation (§3.3), and why it is worst on straight, fast segments (§5.1).

## 9. Phase 11: training the same architecture on GeoLife (conditional)

**Not completed at the time of publication.** Training data (`scripts/22`, development users only, 2,398 train / 323 validation regular windows) and two runs (from scratch, and fine-tuned from the three-shape weights, seed 42, same Model D configuration) were started, but had not finished. No result is reported, and no conclusion about real-data training is drawn. They will be evaluated with `scripts/23_geolife_trained_models.py` on the same 1,000 test segments.

## 10. Limitations

* **Normalisation.** A per-segment similarity transform is the only adaptation used. Five strategies and a sensitivity sweep were tried on development
  users, and none changes the conclusion. A learned or rotation-canonicalising transform was not tried.
* **Observation density and sampling.** Results are for 200-record windows of dense GeoLife data (median 2 s sampling), so 30 observations are
  4–33 s apart on average. Sparser real-world sampling would raise baseline errors (Figure 8), but the Latent ODE is far worse at every gap studied.
* **GPS noise.** Targets are noisy measurements (a few metres). This sets a floor of a few metres for every method and compresses the differences
  between the baselines, but it cannot explain a 126 m error.
* **Movement complexity.** Urban movement is piecewise straight with stops. This is precisely the distribution shift tested, and the result is that the
  synthetic prior is wrong for it.
* **Scope.** Mostly Beijing and mostly one dataset. Segment rules (200 records, ≥ 100 m extent, breaks at 60 s / 70 m/s) and the definitions of
  "straight / curved / turning" are documented choices. Protocol A assumes the segment end time is known, though this assumption does not affect the conclusions.
* **Single checkpoints.** The frozen model is one seed. Earlier seed studies showed a seed-to-seed spread far smaller than the 17× gap observed here.
  Phase 11 models are single-seed (§9).

## 11. Conclusions

* **Demonstrated.** The frozen synthetic-trained Latent ODE can be applied to real GPS without code changes and without leakage, but its
  interpolations of withheld positions are about 17–22× less accurate than linear, cubic-spline or PCHIP interpolation of the same 30 points.
  The result is consistent over 1,000 held-out segments from 97 users. Synthetic-to-real transfer for this model and these trajectory families **fails**.
* **Also demonstrated.** Classical shape-preserving interpolation (PCHIP) is the strongest method here, with a median error of 5.6 m at 30 observations
  and 2.9 m at 50. Any learned interpolator for this kind of data has to beat that.
* **Not yet tested.** Whether training on real GPS data fixes the failure (Phase 11).
* **Unresolved.** Whether richer synthetic families (lines, piecewise-linear paths with turns, stops, speed changes), equivariant architectures,
  or larger real-data training would close the gap to classical interpolation. Whether a learned model could still add value where
  interpolation cannot, such as long gaps, forecasting, or uncertainty. None of this has been tested here, and no claim is made about it.

## 12. Reproduction and files

```bash
python scripts/20_inspect_geolife.py                               # audit + segment index (~20 s)
python scripts/21_evaluate_geolife_transfer.py prepare pilot normsel main density figures   # ~12 min
python scripts/22_geolife_training_data.py                         # Phase 11 data (dev users only)
python scripts/02_train_model.py --config configs/model_d.yaml --experiment geolife_regular --data-root data/geolife/processed \
    --seed 42 --data-seed 42 --threads 6 --checkpoint checkpoints/geolife/geolife_scratch_seed42.pt
python scripts/02_train_model.py --config configs/model_d.yaml --experiment geolife_regular --data-root data/geolife/processed \
    --seed 42 --data-seed 42 --threads 6 --lr 3e-4 --init-checkpoint checkpoints/validation/three_shapes_seed42.pt \
    --checkpoint checkpoints/geolife/geolife_finetune_from_three_shapes_seed42.pt
python scripts/23_geolife_trained_models.py package evaluate
python -m pytest -q tests
```

* Code: `src/glo/geolife.py`, `src/glo/geolife_evaluation.py`, `scripts/20_inspect_geolife.py`, `scripts/21_evaluate_geolife_transfer.py`,
  `scripts/22_geolife_training_data.py`, `scripts/23_geolife_trained_models.py`, `tests/test_geolife.py`, `configs/geolife.yaml`.
* Results: `results/geolife/` (`dataset_statistics.*`, `splits_summary.json`, `pilot/`, `main_evaluation/summary.json`,
  `main_evaluation/density_summary.json`, `main_evaluation/geolife_trained_summary.json`, `tables/table1–5*.md`, `tables/normalization_strategies.json`, `figures/fig1–9*.png`).
* GeoLife-derived data (not in git): `data/geolife/processed/`, `data/geolife/splits/`, the per-segment CSV files.
