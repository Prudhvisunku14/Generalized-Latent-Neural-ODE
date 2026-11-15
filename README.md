# Generalized Latent Neural ODE

A research codebase that studies **one shared Latent Neural ODE for several 2-D trajectory families**, and then asks whether
a model trained on synthetic geometry can be used on **real GPS data (GeoLife)**. It is built on the Latent ODE of
Chen et al. (2018) and uses the same model family throughout. **No new architecture is claimed**, and nothing here shows
generalisation to arbitrary trajectories. The honest summary of the findings is in [Results](#results) and
[Known limitations](#known-limitations).

## Status of the work

| Item | Status |
|---|---|
| 1. Original-style Latent ODE (4-D latent, GRU 25, MLP 20) re-implemented and trained on synthetic circles + spirals | **Completed** (it under-performed; see below) |
| 2. Improved unified Latent ODE for circles, spirals, ellipses; 5-seed validation; unified vs specialist comparison | **Completed** |
| 3. Dynamics-conditioning study (does conditioning the ODE on an inferred code close the unified-vs-specialist gap?) | **Completed**: hypothesis not supported at this scale |
| 4. Pretrained inference from a CSV (no shape label, frozen weights) | **Completed** |
| 5. Synthetic-to-real transfer: frozen synthetic model on GeoLife (held-out users) | **Completed**: the model does **not** transfer |
| 6. GeoLife-trained models (from scratch; fine-tuned from the synthetic weights) | **In progress**: training runs started (from scratch; fine-tuned from the three-shape weights) but not finished, so no results, weights or claims are included yet |
| Planned, not started | richer synthetic families (lines, piecewise-linear paths), equivariant / canonicalised inputs, uncertainty, more shapes. Nothing is claimed about these. |

The independent full reproduction of the paper's own experiments is **not** part of this repository.
Item 1 is a re-implementation of the paper's Latent ODE *architecture* (4-D latent, 25-unit GRU, 20-unit MLPs) on this project's synthetic data.
It is not a numerical reproduction of the paper's benchmark (see [`references/SOURCES_AND_METHODS.md`](references/SOURCES_AND_METHODS.md)).

## Research motivation

Neural ODEs model a trajectory as the flow of a learned vector field. Latent ODEs add an encoder that turns sparse, irregular,
noisy observations into an initial latent state, which makes them natural for irregularly sampled data. A long-term goal is a
**single continuous-time latent dynamical model that handles several kinds of movement** without being told which kind it sees.
This repository is a controlled step in that direction. It first checks, on synthetic families with known ground truth, how far one
shared model goes and what it costs. It then tests the hardest question: does what was learned from synthetic geometry say anything useful about real movement?

## Research objectives

1. **Establish a working, validated Latent ODE baseline** for sparse, noisy 2-D trajectories (circles and logarithmic spirals), including diagnosing why
   the first checkpoint failed and demonstrating reproducibility across seeds.
2. **Test unification:** can one model cover circles, spirals and ellipses with no shape label, how does it compare with specialist models, and does
   conditioning the dynamics on an inferred code help?
3. **Test synthetic-to-real transfer:** can the pretrained model interpolate withheld positions in real GeoLife GPS trajectories without retraining, and how does it compare with
   linear and cubic interpolation? If not, what does training on real GPS data change?

## Architecture

### Original Latent ODE (Chen et al., 2018), as re-implemented here
Reverse-time recurrent encoder → Gaussian posterior over the initial latent state z₀ (4-D) → autonomous MLP `dz/dt = f(z)` → ODE solver → decoder to (x, y).
Sizes: GRU 25, dynamics MLP 20, decoder MLP 20 (one hidden layer each). Trained with the negative ELBO (Gaussian likelihood on observed points + KL to N(0, I)).

### Improved unified model used in this project ("Model D")
The **same model family** with larger hyper-parameters and fixed-step RK4, selected on validation ELBO:

```
sparse noisy observations [x, y, t/T, Δt/T]   (T = 10, newest first)
        │  GRU encoder (4 → 64)
        ▼
Linear 64 → 16  = (μ, log σ²) of z₀ ∈ ℝ⁸          training: z₀ = μ + σε    inference: z₀ = μ
        │  dz/dt = f_θ(z)      tanh MLP 8 → 64 → 64 → 8, autonomous
        ▼  RK4, step 0.1, from t = 0
latent path z(t) → decoder tanh MLP 8 → 64 → 64 → 2 → x̂(t), ŷ(t)
```
24,602 parameters. Adam (lr 3·10⁻³, cosine to 10⁻⁵), 300 epochs, batch 32, 30 observations at t ≤ 7, checkpoint = lowest validation −ELBO.
Config: [`configs/model_d.yaml`](configs/model_d.yaml). Two changes against the original configuration were found necessary (latent 4 → 8 and a larger network); see
[`reports/reconstruction_failure_analysis.md`](reports/reconstruction_failure_analysis.md) and [`reports/improved_two_shapes_results.md`](reports/improved_two_shapes_results.md).

An experimental **dynamics-conditioned** variant (M3) adds a constant 4-D code c inferred by the encoder, `dz/dt = f(z, c)`. Conditioning an ODE on an inferred
variable is established in the literature (Neural ODE Processes, GOKU-nets, CoDA, LEADS); this repository makes no novelty claim.

## Supported trajectory families
Synthetic, 2-D, analytically generated (details: [`references/DATASET_CARD.md`](references/DATASET_CARD.md)):
**circle**, **logarithmic spiral** (expanding or contracting) and **ellipse**, with random centre, phase, angular speed and rotation direction.
Real GPS trajectories (GeoLife) are used only in the transfer experiments. The unified models output no shape name and take no shape label.

## Datasets

| Dataset | Content | In this repo |
|---|---|---|
| `data/processed/two_shapes`, `three_shapes` | 2,000 / 3,000 synthetic trajectories (700/150/150 per family train/val/test), 200 points on t ∈ [0, 10], σ = 0.05 noise, seed 42 | **yes** (own data; regenerate with `scripts/01_generate_dataset.py`) |
| GeoLife GPS Trajectories 1.3 | 182 users, 18,670 files, 24.9 M GPS records (Microsoft Research Asia) | **no** (see licence note) |

### GeoLife licence: please read
GeoLife is distributed under the **Microsoft Research License Agreement (non-commercial use only; you may not redistribute the data or derivative works)**.
For that reason this repository contains **no GeoLife data, no per-user trajectory coordinates, no per-segment tables and no figures that draw individual users' paths**
(they are excluded by `.gitignore`). It contains only code, aggregate statistics and aggregate figures. To reproduce the GeoLife experiments, download the dataset
yourself (search for *GeoLife GPS Trajectories 1.3* on the Microsoft Download Center), keep the file name `Geolife Trajectories 1.3.zip`, and place it in the project root.
The archive is read in place and never extracted or modified. Please cite Zheng et al. (2008, 2009, 2010) (see [References](#references)).

## Installation
Python 3.10–3.12. Tested with Python 3.12, PyTorch 2.5.1, torchdiffeq 0.2.5, NumPy 1.26, SciPy 1.13, pandas 2.2, scikit-learn 1.4.

```bash
git clone https://github.com/Prudhvisunku14/Generalized-Latent-Neural-ODE.git
cd Generalized-Latent-Neural-ODE
python -m venv .venv && source .venv/bin/activate
# install PyTorch for your CPU/CUDA first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python -m pytest -q tests          # 71 tests, about 20 s on CPU
```
The pretrained weights are ordinary files (about 100 KB each), so **Git LFS is not required**. The tiny networks train faster on CPU than on a shared GPU.

## Training
```bash
# synthetic data (already included; regenerate deterministically if wanted)
python scripts/01_generate_dataset.py --n-per-family 1000 --n-points 200 --n-obs 30 --seed 42 && python scripts/04_validate_dataset.py

# Model D on circles + spirals; use --experiment three_shapes for circles + spirals + ellipses
python scripts/02_train_model.py --config configs/model_d.yaml --experiment two_shapes --seed 42 \
    --checkpoint checkpoints/my_two_shapes_seed42.pt          # refuses to overwrite existing checkpoints

# the studies (seeds, specialists, conditioning)
bash scripts/run_validation_experiments.sh
bash scripts/run_conditioning_study.sh
```
Training scripts never overwrite an existing checkpoint unless `--overwrite` is passed.

## Pretrained inference
`saved_models/` holds frozen, checksummed models (see [`saved_models/README.md`](saved_models/README.md)). Give a CSV with columns `t,x,y`
(observations at 0 ≤ t ≤ 7, ideally 30 of them) and no shape label:
```bash
python scripts/predict_trajectory.py --list-models
python scripts/predict_trajectory.py --model three_shapes_unified --input input_trajectories/example_circle.csv \
    --output inference_outputs/circle_example --ground-truth input_trajectories/ground_truth/example_circle.csv
```
It writes `predicted_trajectory.csv` (reconstruction for t ≤ 7, forecast for 7 < t ≤ 10), figures and `run_summary.json` (and `metrics.json` if a ground truth is given).
Weights are frozen and the posterior mean is used, so results are deterministic. **The models were trained with exactly 30 observations and are inaccurate at other counts**
(see Known limitations). Full guide: [`docs/PRETRAINED_INFERENCE_GUIDE.md`](docs/PRETRAINED_INFERENCE_GUIDE.md).

## Evaluation methodology
* Observations: 30 noisy points at t ≤ 7 (the first at t = 0), randomly spaced; **no coordinate after t = 7 reaches the encoder**. Targets are the clean trajectory on 200 points over [0, 10].
* Metrics in original coordinate units: reconstruction RMSE (unobserved context points), forecast RMSE / MAE / ADE / FDE (t > 7), Chamfer distance, plus geometric errors (radius drift, angular velocity, growth rate).
* Every compared model sees **identical observation subsets**. Across-seed intervals are Student-t 95% CIs; paired comparisons use trajectory-level bootstrap and Wilcoxon tests.
* Checkpoint selection uses the **validation** split only (−ELBO); the test split was not used to choose hyper-parameters.
* GeoLife: user-disjoint development and test sets; every method receives the same 30 observations and is scored at the **recorded timestamps against the recorded GPS positions** (never against an interpolated reference);
  coordinates are normalised per segment using only the observed points; confidence intervals use a user-cluster bootstrap. Details: [`reports/geolife_real_world_interpolation_report.md`](reports/geolife_real_world_interpolation_report.md).

## Results
All numbers are from runs in this repository. Reports hold the full tables and caveats; all of them are single-dataset, small-scale results.

### 1. The original-style configuration versus the improved model (2 shapes, validation split, seed 42)
| Run | Architecture | Val context RMSE | Val forecast RMSE |
|---|---|---|---|
| A | original (latent 4, hidden 25/20/20) | 0.335 | 0.815 |
| **D** | latent 8, hidden 64, 2 layers, lr 3·10⁻³ | **0.059** | **0.262** |

The first checkpoint that was inspected had been stopped after about 3 epochs; it is not evidence about the architecture. Run A above is a properly trained original-size model.
[`reports/improved_two_shapes_results.md`](reports/improved_two_shapes_results.md)

### 2. Reproducibility of Model D (5 seeds, test split)
Reconstruction RMSE **0.060 ± 0.007**, forecast RMSE **0.219 ± 0.050** (95% CI [0.157, 0.282]). Forecast accuracy varies noticeably with the seed. Training had not fully converged (best epoch 288–297 of 300).
[`reports/validation_study_report.md`](reports/validation_study_report.md)

### 3. Unified vs specialist models (forecast RMSE, test)
| Family | Unified (5 seeds) | Specialist (3 seeds) |
|---|---|---|
| circle | 0.161 ± 0.037 | **0.087 ± 0.006** |
| spiral | 0.284 ± 0.067 | **0.198 ± 0.009** |

One shared model costs forecasting accuracy (reconstruction is close). Specialists are also far more stable across seeds.

### 4. Three families, and unseen ellipses (single seed)
| Model (150 test ellipses) | Recon RMSE | Forecast RMSE |
|---|---|---|
| three-shape model (ellipses in training) | 0.089 | 0.185 |
| two-shape models, ellipses never seen | 0.210 | 0.321 |

Rotation direction transfers; **shape does not** (unseen ellipses are rounded toward circles, and reconstruction is worse than interpolating the observations). Evidence for out-of-family generalisation is weak.

### 5. Dynamics conditioning (5 seeds, forecast RMSE, test)
| Model | Circle | Spiral |
|---|---|---|
| specialists | 0.087 ± 0.006 | 0.198 ± 0.009 |
| M1 unified | 0.145 ± 0.007 | 0.264 ± 0.036 |
| M2 unconditioned, parameter-matched | 0.172 ± 0.096 | 0.270 ± 0.113 |
| M3 dynamics-conditioned | 0.176 ± 0.067 | 0.299 ± 0.068 |

Conditioning did not close the gap at this scale (M3 − M1: circle +0.031, CI [−0.050, +0.112]; spiral +0.035, CI [−0.063, +0.133]). The learned code does control radial growth.
[`reports/conditioning_study_report.md`](reports/conditioning_study_report.md)

### 6. Packaged models on the held-out synthetic test split (30 observations, single checkpoints)
| Model | Recon RMSE | Forecast RMSE |
|---|---|---|
| `three_shapes_unified` | 0.082 | 0.211 |
| `two_shapes_unified` | 0.055 | 0.169 |
| `circle_specialist` | 0.045 | 0.085 |
| `spiral_specialist` | 0.063 | 0.194 |
| `dynamics_conditioned` | 0.051 | 0.212 |
| `variable_nobs_experimental` | 0.057 | 0.389 |

### 7. Synthetic-to-real transfer on GeoLife (frozen three-shape model, 30 observations)
1,000 held-out segments from 97 users; 105,037 withheld GPS positions. Median per-segment interpolation RMSE in metres:

| Method | Median RMSE [95% CI] | Mean | Win rate vs linear |
|---|---|---|---|
| **Latent ODE (frozen, synthetic-trained)** | **126.4** [110.9, 147.9] | 272.7 | 0.1% |
| Linear interpolation | 7.10 [5.80, 8.48] | 13.0 | n/a |
| Cubic spline | 5.56 [4.64, 6.90] | 11.7 | 68.6% |
| PCHIP | 5.62 [4.54, 6.69] | 10.1 | 85.9% |

**The synthetic-trained model does not transfer to real GPS.** It runs on real input, but its interpolations are about 17× worse than linear
interpolation (geometric-mean ratio 17.3, user-bootstrap CI [15.4, 19.3]), for every trajectory type, speed, observation gap, transport mode and observation count (10–50),
and for every normalisation strategy tried. Forecasts are 3–5× worse than a constant-velocity extrapolation. The likely cause is that the learned dynamics can only produce orbit-like curves.
[`reports/geolife_real_world_interpolation_report.md`](reports/geolife_real_world_interpolation_report.md), [`reports/geolife_dataset_audit.md`](reports/geolife_dataset_audit.md)

### 8. GeoLife-trained models
Not available yet. Both training runs (scripts `22_geolife_training_data.py`, `02_train_model.py --experiment geolife_regular`, `23_geolife_trained_models.py`) were still running when this repository was published. Their weights are not included (they are also arguably derivative works of GeoLife under its licence). This section will be filled in only with measured results.

## Known limitations
* **Unified models forecast worse than specialists**, and seed-to-seed variation is large (forecast SD about 0.05 over 5 seeds). Most unified results are single-seed or few-seed.
* **Observation count.** The models were trained with exactly 30 observations and become several times worse than simple interpolation at 10, 20, 40 or 50. A variable-count model (experimental) fixes reconstruction but doubles the forecast error at 30.
* **Forecast accuracy decays** over the 3-unit horizon (phase lead, radius drift); large or expanding spirals are hardest.
* **Out-of-family generalisation is weak**: unseen shapes are mapped toward the nearest learned one.
* **No transfer to real GPS** (above). Classical PCHIP / cubic interpolation is far better on GeoLife with 30 observations; a learned model has to beat 5–6 m to be useful there.
* GeoLife evidence is from one dataset, mostly in Beijing, with 200-record windows of dense data; segment rules and the "straight / curved / turning" categories are documented choices.
* Training solver is fixed-step RK4 (checked against dopri5: mean difference 0.0009); a few runs were still improving at the last epoch.
* No claim is made of a new architecture, of disentangled representations, of universal trajectory generalisation, or of novelty for conditioning.

## Repository structure
```
configs/            training configs (model_d.yaml, conditioning/, geolife.yaml)
src/glo/            model.py, solver.py, pipeline.py, data.py, evaluation.py, inference.py, geolife.py, geolife_evaluation.py
scripts/            01-12 data/train/evaluate/studies, 20-23 GeoLife, predict_trajectory.py, package_models.py, check_publication_readiness.py
saved_models/       frozen pretrained models + README (weights, architecture, preprocessing, metadata, SHA-256)
checkpoints/        all training checkpoints and per-epoch histories (validation/, conditioning/, improved_two_shapes/)
data/processed/     synthetic datasets (two_shapes, three_shapes); data/geolife/ holds only SOURCE.md (GeoLife is not distributed)
input_trajectories/ example CSV inputs;  inference_outputs/  example predictions
reports/            technical reports (failure analysis, improved results, validation, conditioning, GeoLife audit and transfer, implementation changes, publication readiness)
results/            metrics, figures and tables from every experiment (GeoLife: aggregate results only)
docs/               pretrained-inference guide;  tests/  71 unit and integration tests;  logs/  training logs;  backups/  untouched originals
```

## Reproducibility
* Fixed seeds everywhere; each study records its seeds. Inference is bit-reproducible on CPU with one thread (verified: byte-identical outputs).
* Checkpoints are never overwritten by the scripts; `saved_models/*/metadata.json` records the SHA-256 and the original checkpoint path, and every load verifies the checksum.
* Everything in [`reports/`](reports/) is generated by the scripts in `scripts/` from the data and checkpoints in this repository (GeoLife steps need the dataset you download yourself).
* `python scripts/check_publication_readiness.py` re-verifies all packaged models and the repository contents; see [`reports/publication_readiness_report.md`](reports/publication_readiness_report.md).
* GeoLife pipeline: `python scripts/20_inspect_geolife.py && python scripts/21_evaluate_geolife_transfer.py all && python scripts/22_geolife_training_data.py` (then train, then `scripts/23_geolife_trained_models.py package evaluate`; commands in the GeoLife report, §12).

## References
* R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud. *Neural Ordinary Differential Equations.* NeurIPS 2018. arXiv:1806.07366.
* Y. Rubanova, R. T. Q. Chen, D. Duvenaud. *Latent ODEs for Irregularly-Sampled Time Series.* NeurIPS 2019. arXiv:1907.03907.
* `torchdiffeq` (R. T. Q. Chen, MIT licence): https://github.com/rtqichen/torchdiffeq. Used as a dependency; no upstream source is vendored.
* A. Norcliffe et al. *Neural ODE Processes.* ICLR 2021. · V. Linial et al. *Generative ODE modeling with known unknowns (GOKU-nets).* CHIL 2021. · M. Kirchmeyer et al. *Generalizing to new physical systems via context-informed dynamics model (CoDA).* ICML 2022. · Y. Yin et al. *LEADS.* NeurIPS 2021. (Literature notes and caveats: [`reports/conditioning_study_report.md`](reports/conditioning_study_report.md).)
* GeoLife: Y. Zheng, L. Zhang, X. Xie, W.-Y. Ma. *Mining interesting locations and travel sequences from GPS trajectories.* WWW 2009. · Y. Zheng, Q. Li, Y. Chen, X. Xie, W.-Y. Ma. *Understanding mobility based on GPS data.* UbiComp 2008. · Y. Zheng, X. Xie, W.-Y. Ma. *GeoLife: A collaborative social networking service among user, location and trajectory.* IEEE Data Eng. Bull. 33(2), 2010.

## License
No license has been chosen yet; until one is added, all rights are reserved by the authors. The GeoLife dataset is governed by its own Microsoft Research License Agreement (non-commercial use only).
