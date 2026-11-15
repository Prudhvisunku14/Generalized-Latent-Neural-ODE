# Generalized Latent Neural ODE: multi-shape trajectory learning

**Research starter implementation — Phase 1 (circle + spiral), Phase 2 (circle + spiral + ellipse).**

**Status:** Complete synthetic datasets, deterministic train/validation/test split generation, data validation, visualization, a variational Latent ODE model, an optional dynamics-conditioning extension, training/evaluation CLI scripts, and automated tests. A trained research-quality checkpoint is **not** included. Any quick smoke checkpoint is **not** a scientifically trained model.

## Goal

Learn a **single shared continuous-time dynamical model** capable of inferring sparse/noisy 2-D trajectories and predicting unobserved/future coordinates from different *trajectory families*. First test circle+spiral; then retrain and test the same architecture on circle+spiral+ellipse; only then consider figure-eight and other families.

### The generative architecture

- Reverse-time GRU encoder takes sparse noisy prefix points and corresponding timestamps, predicts the mean and log-variance of latent initial state `z0`.
- During training, reparameterization samples `z0`; at validation/test time we decode from the posterior mean for deterministic RMSE comparisons.
- One **autonomous** MLP `dz/dt=f_theta(z)` controls every trajectory in a given experiment.
- A differentiable ODE solver evaluates the latent path at arbitrary query times.
- A decoder maps each latent state to `(x,y)` coordinates.
- Variational training uses Gaussian likelihood on available noisy observations plus Gaussian KL to a `N(0,I)` prior.

The conditional extension adds a GRU-inferred, constant trajectory-specific code `c`, with `dz/dt=f_theta(z,c), dc/dt=0`. This is an experimental **ablation**; one should not assume it is a new research invention without reviewing prior conditional Neural ODE literature.

## Get started on Ubuntu (Python 3.10-3.12 recommended)

```bash
cd Generalized_Latent_ODE
python -m venv .venv
source .venv/bin/activate
# Install PyTorch for your CUDA/CPU first, following https://pytorch.org/get-started/locally/
pip install -r requirements.txt

# Datasets are already included; rerun to regenerate with fixed seed:
python scripts/01_generate_dataset.py --n-per-family 1000 --n-points 200 --n-obs 30 --seed 42
python scripts/04_validate_dataset.py
python -m pytest -q
```

### Phase 1: train one model on circle + spiral

```bash
python scripts/02_train_model.py --experiment two_shapes --epochs 100 --n-obs 30 --n-query 100
python scripts/03_evaluate_model.py --checkpoint checkpoints/two_shapes_cond0_best.pt \
    --n-obs 30 --n-query 200 --output results/phase1_evaluation.json
```

### Phase 2: use the same architecture on three shapes

```bash
python scripts/02_train_model.py --experiment three_shapes --epochs 100 --n-obs 30 --n-query 100
python scripts/03_evaluate_model.py --checkpoint checkpoints/three_shapes_cond0_best.pt \
    --n-obs 30 --n-query 200 --output results/phase2_evaluation.json
```

### Optional: dynamics-conditioned extension

```bash
python scripts/02_train_model.py --experiment two_shapes --epochs 100 --cond-dim 4
python scripts/03_evaluate_model.py --checkpoint checkpoints/two_shapes_cond4_best.pt
```

### Quick engineering smoke test (NOT a trained benchmark)

```bash
python scripts/02_train_model.py --experiment two_shapes --epochs 1 \
  --batch-size 4 --n-query 40 --n-obs 10 --max-train-batches 2 --max-val-batches 1 \
  --checkpoint checkpoints/two_shapes_SMOKE_ONLY.pt
python scripts/03_evaluate_model.py \
  --checkpoint checkpoints/two_shapes_SMOKE_ONLY.pt \
  --n-obs 10 --n-query 40 --max-batches 1 \
  --output results/SMOKE_ONLY_metrics.json
```

### Solver note

The official [torchdiffeq](https://github.com/rtqichen/torchdiffeq) library is the intended dependency. If it cannot be imported, the code runs a transparent fixed-step **differentiable RK4** fallback so that data, model, and gradient tests can run without a network connection. The fallback **does not** implement `dopri5` adaptive error control or adjoint differentiation. Never claim the fallback reproduces original NeurIPS paper solvers. With `torchdiffeq` installed, `--solver dopri5` uses the official adaptive implementation; `--adjoint` selects `odeint_adjoint`.

## Generated datasets

Data files are `data/processed/{two_shapes,three_shapes}/{train,val,test}.npz`. Every trajectory has **200 reference timestamps over [0,10]**, and metadata include full clean coordinates, one independently generated noisy observation array, an irregular default observation mask (30 points from `t<=7`), the family ID, a unique trajectory ID, and the exact mathematical generation parameters.

| Experiment | Families | Train | Validation | Test | Total |
|---|---|---:|---:|---:|---:|
| Phase 1 | Circle, spiral | 1400 | 300 | 300 | 2000 |
| Phase 2 | Circle, spiral, ellipse | 2100 | 450 | 450 | 3000 |

Each family supplies 700/150/150 trajectories. The **same circle and spiral trajectories remain in the same split** in the two- and three-family datasets. No trajectory ID appears in multiple splits. Test normalization uses training-set statistics only. A clean dense trajectory stored in the dataset is **not an encoder input**. The noisy prefix is an encoder input; clean points are evaluation targets.

### Family equations

Let `angle = omega*t + phi`, with random positive/negative `omega`, random phase and center `(cx,cy)`.

- **Circle:** `x=cx+r*cos(angle)`, `y=cy+r*sin(angle)`.
- **Logarithmic spiral:** `R(t)=r0*exp(k*t)`, `x=cx+R(t)*cos(angle)`, `y=cy+R(t)*sin(angle)`; `k` is positive or negative.
- **Rotated ellipse:** from `u=a*cos(angle), v=b*sin(angle)`, rotate `(u,v)` by `alpha` and translate by `(cx,cy)`.

Parameter ranges are in `src/glo/data.py`; the generated `.npz` files contain exact sampled values for every example. These trajectories are **not the identical original paper spiral dataset**, so do not compare absolute RMSE with Table 2 as if the data were the same.

### Loading a dataset in Python

```python
import numpy as np
with np.load('data/processed/two_shapes/train.npz', allow_pickle=False) as z:
    print(z['xy'].shape)               # (1400, 200, 2)
    print(z['default_obs_mask'].shape) # (1400, 200)
    print(z['family'].shape)           # (1400,)
    print(z['time'].shape)             # (200,)
```

Shape mappings: `circle=0`, `spiral=1`, `ellipse=2` (ellipse appears only in Phase 2).

## Training and evaluation protocol

- **Time range:** `[0,10]`; encoder sees only samples from the prefix `t<=7`; future target points are at `t>7`.
- **Batching:** evaluate all trajectories of a batch on a common query time grid, but use different randomly selected irregular context times for each trajectory.
- **Observation count:** `--n-obs`; must fit within the query grid's `t<=7` prefix. The very first timestamp is always observed so the inferred latent state corresponds to a well-defined initial time.
- **Default training objective:** `--supervision observed`, i.e. Gaussian reconstruction of *only the noisy observed points* plus KL. For an explicitly more-supervised synthetic experiment, `--supervision full_clean` uses all clean trajectory points (including future) for training targets; do not call that comparable to sparse-observation unsupervised training.
- **No future leakage during inference:** time-ordered future points are not passed to the encoder. Future ground truth is used only for scoring (or for training targets when the optional fully supervised objective is chosen).
- **Evaluation:** deterministic posterior mean, test split only; save full test predictions and calculate sample-wise metrics in original coordinate units.
- **Randomness:** fixed seeds for trajectory generation, split assignment, observation masks, and initialization; rerun with multiple seeds and report mean/std for final publication.

## Evaluation metrics

Evaluation generates overall and per-family metrics in JSON:

- `recon_rmse`: coordinate RMSE on **unobserved prefix** points (`t<=7`, excluding context points).
- `forecast_rmse`: coordinate RMSE on **future** points (`t>7`).
- `forecast_mae`: coordinate-wise mean absolute error on future points.
- `forecast_ade`: mean Euclidean distance on future positions.
- `forecast_fde`: Euclidean distance at the final future timestamp.
- `chamfer_full`: squared symmetric Chamfer distance on all predicted/reference XY coordinates. Chamfer is geometric and intentionally ignores time ordering.

Metrics are computed **for each trajectory**, then the JSON reports mean and standard deviation across trajectories. Do not compare with an original Table 2 RMSE number unless trajectory generation, normalization, and precise evaluation interval match.

## Folder guide

```text
Generalized_Latent_ODE/
├── configs/                     # Phase 1 and Phase 2 experiment choices
├── data/processed/
│   ├── two_shapes/              # 2000 samples (2 families), split files and metadata
│   └── three_shapes/            # 3000 samples (3 families), split files and metadata
├── src/glo/
│   ├── data.py                  # Math, synthetic generation, split validation
│   ├── solver.py                # torchdiffeq adapter + offline fallback
│   ├── model.py                 # GRU, variational Latent ODE, optional conditioning
│   ├── pipeline.py              # Losses, sparse inputs, metrics
│   └── visualization.py         # Family galleries and prediction plots
├── scripts/
│   ├── 01_generate_dataset.py
│   ├── 02_train_model.py
│   ├── 03_evaluate_model.py
│   └── 04_validate_dataset.py
├── tests/                       # pytest tests, including no-leak and backprop checks
├── checkpoints/                 # Model files created by training
├── results/                     # Dataset galleries, evaluation JSON, prediction plots
└── references/SOURCES_AND_METHODS.md
```

## Scientific validity / next experiments

1. **Phase 1:** train and score on circles + spirals. Report per-shape performance and visualization, not only a pooled metric.
2. **Phase 2:** train a *new* shared checkpoint on circles + spirals + ellipses using same architecture, compare under matched training conditions.
3. **Model capacity:** compare 4, 8, 16 latent dimensions and parameter-matched conditional baselines.
4. **Sparsity and noise:** vary context counts and testing noise (the `--noise-std` evaluation option supports controlled tests).
5. **Held-out family:** train on two families and evaluate on ellipse via a *separately constructed leave-one-family-out dataset*. Do not use Phase 2's ellipse test set to claim unseen-family generalization, because its training set contains ellipses.
6. **Broader families:** figure-eight, sinusoidal, and damped oscillator trajectories; real timestamped trajectory datasets.

**Novelty boundary:** one model learning circles, ellipses and spirals is an experimental extension, not by itself proof of a novel architecture. A paper needs demonstrable improvements over strong baselines and a relevant prior-work search.

## References

- Chen, Rubanova, Bettencourt and Duvenaud, *Neural Ordinary Differential Equations*, NeurIPS 2018, Section 5 (uploaded paper).
- [rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq), particularly [`examples/latent_ode.py`](https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py).
- [YuliaRubanova/latent_ode](https://github.com/YuliaRubanova/latent_ode) (Rubanova et al., 2019, separate follow-up work).

See `references/SOURCES_AND_METHODS.md` for specific source-to-implementation mapping and differences.

## Sampling new trajectories after training

The **unconditioned** Latent ODE defines a normal prior on the initial state, so it can sample entirely new trajectories without receiving an observation sequence:

```bash
python scripts/05_sample_from_prior.py \
  --checkpoint checkpoints/two_shapes_cond0_best.pt \
  --n-samples 12 --output-prefix results/prior_generated
```

This generates a `.npz` file containing `generated_xy` of shape `(12,200,2)` and a PNG gallery. The generated family is **not controlled** by a class label; prior sampling does not guarantee circles or spirals unless the model has learned a sufficiently good data distribution. The optional conditional extension currently learns `c` from observations and does not define a generative prior over `c`, so unconditional sampling from it is deliberately blocked.
