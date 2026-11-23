# sections 03 (architecture), 04 (dataset), 05 (methodology); executed from build_mtp_package.py
pcA, pcB, pcC, pcD, pcL, pcO = PC['A'], PC['B'], PC['C'], PC['D'], PC['large'], PC['original-size']
ptab = MD(['Model', 'Latent L', 'GRU width H', 'GRU encoder', 'Posterior layer', 'ODE vector field', 'Decoder', 'Total'],
          [[nm, l, h, f"{c['GRU encoder']:,}", f"{c['posterior layer']:,}", f"{c['ODE vector field']:,}", f"{c['decoder']:,}", f"{c['total']:,}"] for nm, l, h, c in
           (('Original-size configuration (hidden 25/20/20, one hidden layer)', 4, 25, pcO), ('A (baseline Model D)', 8, 64, pcA), ('B', 16, 64, pcB), ('C', 8, 128, pcC), ('D', 16, 128, pcD), ('Large (capacity study)', 32, 128, pcL))])
CSV('06_EXPERIMENTAL_RESULTS/RESULTS_TABLES/model_parameter_counts.csv', ['model', 'latent', 'gru_width', 'gru_encoder', 'posterior_layer', 'ode_vector_field', 'decoder', 'total'],
    [[nm, l, h, c['GRU encoder'], c['posterior layer'], c['ODE vector field'], c['decoder'], c['total']] for nm, l, h, c in (('original-size', 4, 25, pcO), ('A', 8, 64, pcA), ('B', 16, 64, pcB), ('C', 8, 128, pcC), ('D', 16, 128, pcD), ('large', 32, 128, pcL))])

W('03_MODEL_ARCHITECTURE/COMPLETE_ARCHITECTURE.md', f'''
# Complete architecture (verified from `src/glo/model.py`, `solver.py`, `pipeline.py`)

**Pipeline:** Observed trajectory -> GRU encoder -> initial latent state -> Neural ODE -> RK4 solver -> decoder -> predicted trajectory.

![architecture](../07_FIGURES/ARCHITECTURE/fig_architecture_pipeline.png)

(Figure file: `07_FIGURES/ARCHITECTURE/fig_architecture_pipeline.png`; drawn from the actual layer sizes.)

## Component summary (baseline Model D: latent L = 8, GRU width H = 64)

| # | Component | Input | Output | Formulation | Implementation | Parameters |
|---|---|---|---|---|---|---:|
| 1 | GRU encoder | 30 observations, newest first: [x, y, t/T, dt/T], shape [batch, 30, 4] | final hidden state, 64 numbers | h_k = GRU(h_(k-1), input_k) | `nn.GRU(input_size=4, hidden_size=H, batch_first=True)` | {pcA['GRU encoder']:,} |
| 2 | Posterior layer | hidden state (64) | mean mu (8) and log-variance (8, clamped to [-8, 5]) | (mu, log sigma^2) = W h + b | `nn.Linear(H, 2L)` | {pcA['posterior layer']:,} |
| 3 | Initial latent state | mu, log-variance | z0 (8) | z0 = mu + sigma*eps, eps ~ N(0, I) in training; z0 = mu in evaluation | reparameterisation in `LatentODE.forward` | 0 |
| 4 | Neural ODE vector field | z (8) | dz/dt (8) | dz/dt = f(z), time-invariant | MLP 8 -> 64 -> 64 -> 8, tanh | {pcA['ODE vector field']:,} |
| 5 | RK4 solver | z0, query times | z(t) at the query times | numerical integral of f | torchdiffeq `odeint`, method `rk4`, step 0.1 | 0 |
| 6 | Decoder | z(t) (8) | (x, y) at each time | (x_hat, y_hat) = g(z(t)) | MLP 8 -> 64 -> 64 -> 2, tanh | {pcA['decoder']:,} |
| | **Total** | | | | | **{pcA['total']:,}** |

Coordinates are normalised with the training-set mean and scale before entering the model and de-normalised before any error is computed. All results in the presentation are in the original coordinate units.

## How the components connect

1. The encoder reads only the 30 noisy observations (never anything after t = 7) and summarises them in 64 numbers.
2. The posterior layer turns those numbers into a Gaussian distribution over the 8-dimensional initial latent state.
3. A sample (or the mean, at test time) is the starting point of the latent ODE.
4. The ODE solver integrates the learned rate-of-change network forward through t in [0, 10], giving a latent state at every requested time.
5. The decoder maps each latent state to an (x, y) point. Times t <= 7 give the reconstruction, times 7 < t <= 10 give the forecast.

## Training versus evaluation

| | Training | Evaluation |
|---|---|---|
| Initial state | sampled: z0 = mu + sigma*eps | posterior mean z0 = mu (deterministic) |
| Query grid | 100 times in [0, 10] | 200 times in [0, 10] |
| Observations | new random 30 times every batch | fixed draw (observation seed 123), identical for all models |
| Loss | negative ELBO on observed points | RMSE against the clean true path |

## Parameter counts for every configuration used

{ptab}

(Counts were obtained by instantiating the model classes with each configuration, without training; they match the counts stored in the checkpoints: 24,602 / 27,186 / 63,642 / 67,250 / 74,466.) Original-size row: counted the same way; **not** a result from a trained run.

## Conditioned variant (an experimental extension; not the main model)

The dynamics-conditioned model M3 adds a 4-dimensional code c = tanh(W_c h) from the encoder and uses dz/dt = f(z, c), dc/dt = 0 (25,118 parameters); see `06_EXPERIMENTAL_RESULTS/CONDITIONING_EXPERIMENTS.md`.
''')

W('03_MODEL_ARCHITECTURE/GRU_ENCODER.md', f'''
# GRU encoder (verified from `src/glo/model.py::RecognitionGRU`)

**What it does.** Reads the sparse, noisy, irregularly timed observations and summarises them into one vector that describes the whole path.
**Why it is needed.** The Neural ODE needs a starting latent state; the encoder is what turns raw observations into that starting state. A GRU (gated recurrent unit) is a recurrent network, so it can read a sequence of any length.

| Item | Verified value |
|---|---|
| Input per observation | 4 features: x, y, t/T, dt/T (T = 10, the last query time) |
| Sequence length | 30 observations |
| Processing direction | **backwards in time**: the sequence is flipped, so the newest observation is read first and the earliest (t = 0) last |
| Network | one-layer unidirectional `nn.GRU(4, H)`, H = 64 (baseline) or 128 |
| Output used | the final hidden state (after reading the earliest observation), H numbers |
| Parameters | 3H(4 + H + 2) = {pcA['GRU encoder']:,} for H = 64; {pcC['GRU encoder']:,} for H = 128 |

**How irregular timestamps are represented.** Each observation carries its own time: t/T (where the point lies on the timeline) and dt/T (the gap to the neighbouring observation in the reading order, 0 for the first one read). So the network sees the actual spacing, not just the order. Observation times are not required to be equally spaced.

**How the posterior over z0 is obtained.** The final hidden state goes through a linear layer (H -> 2L) that outputs the mean and the log-variance of a Gaussian over z0 (see `LATENT_SPACE.md`).

**Mathematical form.** For observations (x_1, t_1) ... (x_N, t_N), reading from the newest: h_k = GRU(h_(k-1), [x_k, t_k/T, dt_k/T]); the code then uses the last h. The GRU's gates (update and reset) are the standard GRU equations.

**Connection to the next component.** The 64-number summary feeds the posterior layer.

**Comparison with the paper.** The paper's recognition network is "an RNN, which consumes the data sequentially backwards in time" with 25 hidden units (cell type not stated). Adding the time features is our design choice.
''')

W('03_MODEL_ARCHITECTURE/LATENT_SPACE.md', f'''
# Latent space and the initial latent state (verified from `model.py` and `pipeline.py`)

| Item | Verified value |
|---|---|
| Latent dimension L | 8 (baseline); 16 and 32 in the size studies; 4 in the original-size configuration |
| Distribution | diagonal Gaussian q(z0 | observations) = N(mu, diag(sigma^2)) |
| Mean and log-variance | produced by `nn.Linear(H, 2L)` from the GRU's final hidden state; log-variance clamped to [-8, 5] |
| Prior | standard normal N(0, I) |
| Reparameterisation (training) | z0 = mu + exp(0.5 * log-variance) * eps, eps ~ N(0, I) |
| Evaluation | z0 = mu (the posterior mean; deterministic) |
| KL term | KL(q || N(0, I)) = 0.5 * sum( mu^2 + exp(logvar) - 1 - logvar ), averaged over the batch; weight 1.0, no annealing in the main runs |

**What the latent state means.** It is a short list of numbers that the model invents to describe "which path is this and where is it now". It is not given a physical meaning by us, and we do **not** claim it is disentangled or interpretable.

**Why the reparameterisation trick.** Sampling directly is not differentiable; writing the sample as mu + sigma*eps moves the randomness to eps so gradients reach mu and sigma.

**Why a distribution rather than one point.** The model is a variational auto-encoder: the KL term keeps the latent codes close to a simple prior, which is part of the training objective (negative ELBO).

**Connection.** z0 is the starting point of the Neural ODE.
''')

W('03_MODEL_ARCHITECTURE/NEURAL_ODE_DYNAMICS.md', f'''
# Neural ODE dynamics (verified from `src/glo/model.py::LatentDynamics`)

| Item | Verified value |
|---|---|
| Input / output | latent state z (L numbers) -> rate of change dz/dt (L numbers) |
| Network | MLP: Linear(L, 64) - tanh - Linear(64, 64) - tanh - Linear(64, L) (two hidden layers of 64 units) |
| Activation | tanh |
| Time dependence | **time-invariant (autonomous)**: the function ignores t, as in the paper |
| Parameters | {pcA['ODE vector field']:,} (L = 8); {pcB['ODE vector field']:,} (L = 16); {pcL['ODE vector field']:,} (L = 32) |
| Hidden width held fixed | 64 in every size study (so most added capacity is in the encoder) |

**Formulation.** dz(t)/dt = f_theta(z(t)); z(t) = z0 + integral from 0 to t of f_theta(z(s)) ds.

**How are the latent dynamics learned?** Only through the training loss: the decoded latent path must match the 30 observed points. Gradients flow back through the solver steps into the vector field. Nothing in the loss involves t > 7, so the future part of the path is an extrapolation of the learned vector field, not something it was directly trained to match.

**Why time-invariant?** A time-invariant field means the latent state alone determines the future, which is the paper's formulation and keeps the model simple. (A limitation: the model must express any dependence on time through the state itself.)

**Connection.** The solver repeatedly evaluates this network to move z forward in time.
''')

W('03_MODEL_ARCHITECTURE/RK4_SOLVER.md', '''
# RK4 solver (verified from `src/glo/solver.py`, the installed torchdiffeq 0.2.5 source, and `audit/verify_2_solver.py`)

| Item | Verified value |
|---|---|
| Library / call | `torchdiffeq.odeint` with `method = 'rk4'`, `options = {'step_size': 0.1}`; `adjoint = False` (direct backpropagation through the steps) |
| Method actually used | torchdiffeq's `rk4` is the **Runge-Kutta 3/8 rule**, a fourth-order method (not the "classical" 1/6-weights RK4) |
| One step (step size h) | k1 = f(y0); k2 = f(y0 + h*k1/3); k3 = f(y0 + h*(k2 - k1/3)); k4 = f(y0 + h*(k1 - k2 + k3)); y1 = y0 + h*(k1 + 3*(k2 + k3) + k4)/8 |
| Step size | 0.1 over t in [0, 10] -> 100 steps, 400 function evaluations per trajectory batch (log shows `nfe/b = 400`) |
| Time grid | uniform solver grid of spacing 0.1; training queries 100 times, evaluation queries 200 times in [0, 10] |
| Off-grid query times | torchdiffeq evaluates the solution on its own 0.1 grid and **linearly interpolates** to query times that fall between steps |
| Offline fallback | a classical RK4 is included for machines without torchdiffeq; it was **not** used for the reported runs (checkpoints record `torchdiffeq`) |

**Why a solver?** The latent state at time t is the integral of a neural network; there is no formula, so it is computed step by step.

**Measured accuracy (checked in this project, `audit/verify_2_solver.py`).** On a test oscillator the error falls by a factor of 16.1 each time the step is halved (fourth order) at times that lie on the solver grid. At off-grid query times the linear interpolation dominates and the error falls only about 4x per halving (second order; 2.7e-3 at step 0.1 on that test problem). For the trained models the effect is small: per-trajectory RMSE changes by about 1.4e-4 to 2.6e-4 on average (maximum 1.2e-3) between rk4 at step 0.1, rk4 at step 0.025 and a tight dopri5 solver, which is negligible against the RMSE differences reported (0.03 to 0.5).

**Wording to use on slides:** "fixed-step fourth-order Runge-Kutta (3/8 rule), step 0.1, with linear interpolation to query times"; avoid "classical RK4" and avoid implying the paper's adaptive solver and adjoint were used.

**Connection.** The solver output z(t) at the query times is passed to the decoder.
''')

W('03_MODEL_ARCHITECTURE/DECODER.md', f'''
# Decoder (verified from `src/glo/model.py`)

| Item | Verified value |
|---|---|
| Input | latent state z(t), L numbers (8 in the baseline) |
| Network | MLP: Linear(L, 64) - tanh - Linear(64, 64) - tanh - Linear(64, 2) |
| Output | 2 numbers per time: the normalised x and y coordinates, de-normalised with the training-set mean and scale |
| Parameters | {pcA['decoder']:,} (L = 8); {pcB['decoder']:,} (L = 16); {pcL['decoder']:,} (L = 32) |

**What it does.** Turns the latent state at each requested time into a point in the plane; the sequence of points over time is the predicted trajectory.

**Why it is needed.** The latent space is abstract; the decoder connects it to the observable (x, y) data and defines the likelihood used for training.

**Likelihood.** The decoder output is the mean of a Gaussian with fixed standard deviation 0.05 (data units); no variance is predicted.

**Connection.** Decoded points at t <= 7 are scored as reconstruction; points at 7 < t <= 10 as forecast.
''')

W('03_MODEL_ARCHITECTURE/TRAINING_LOSS.md', '''
# Training objective (verified from `src/glo/pipeline.py::batch_objective` and `scripts/02_train_model.py`)

**Loss = negative ELBO = reconstruction negative log-likelihood + KL divergence.**

```
NLL  = sum over the 30 OBSERVED points and the 2 coordinates of  [ 0.5 * ((prediction - observation)/s)^2 + log s + 0.5 * log(2 pi) ],   s = 0.05 / data_scale
KL   = 0.5 * sum over latent dims of ( mu^2 + exp(logvar) - 1 - logvar )                               (posterior vs standard normal prior)
loss = mean over the batch of ( NLL + 1.0 * KL )
```

| Question | Answer (verified) |
|---|---|
| Reconstruction likelihood | Gaussian, fixed sigma = 0.05 in data units (divided by the data scale because inputs are normalised) |
| Which points are supervised? | **Only the 30 observed noisy points** (`supervision = observed`) |
| Are future points directly supervised? | **No.** No term involves t > 7, and the unobserved context points are also not supervised. Forecasting is an extrapolation of the learned dynamics. A separate option `full_clean` (clean dense targets) exists in the code but was **never run**. |
| KL weight | 1.0; no annealing in the reported runs |
| Observation noise | the stored noisy values are used; the same noise realisation per trajectory every epoch (only the subset of observed times changes) |
| Validation | posterior-mean z0, the same fixed observation draws every epoch (generator seeded with `seed + 999`), full negative ELBO |
| Checkpoint selection | the epoch with the **lowest validation negative ELBO**; validation forecast RMSE is logged each epoch but never used for selection; test results are never used |
| Optimiser | Adam; learning rate 3e-3 with cosine annealing to 1e-5; gradient-norm clipping 1.0 |

**Why this matters for interpretation.** The model is rewarded for matching the observed points. How well it forecasts the future is a by-product. This is one plausible reason why forecasting can be hard to improve, but it is a **hypothesis that has not been tested** (the `full_clean` supervision experiment was not run).
''')

W('03_MODEL_ARCHITECTURE/MODEL_CONFIGURATION.md', f'''
# Model configurations (verified from `configs/*.yaml` and checkpoint metadata)

## Baseline "Model D" (`configs/model_d.yaml`, used unchanged for all validation experiments)

```
latent_dim 8, encoder_hidden (GRU) 64, dynamics_hidden 64 (2 layers), decoder_hidden 64 (2 layers), tanh
solver rk4, step_size 0.1, adjoint false
epochs 300, batch_size 32, n_query 100 (training grid), n_obs 30
optimizer Adam, lr 0.003, cosine schedule to 1e-05, grad_clip 1.0, patience 60 (early stopping did not trigger in the checked runs)
supervision observed, sigma 0.05, kl_weight 1.0, kl_anneal_epochs 0
seed 42, device cpu, threads 4
```

## All configurations used

{ptab}

| Study | Models | What changes |
|---|---|---|
| Original-size run (E1) | latent 4, GRU 25, MLP 20 x 1 hidden layer | smaller architecture, lr 1e-3 in the first comparison |
| Shared vs specialist | Model D | trained on both families (shared) or one family (specialist) |
| Three-family / five-family | Model D | training data only |
| Conditioning | M1 24,602; M2 25,049 (dynamics width 67); M3 25,118 (code c in R^4); M2aug 25,894 (latent 12) | extra code / extra latent dimensions |
| Capacity scaling | baseline 8/64, medium 16/128, large 32/128 | latent and GRU width together; ODE and decoder widths unchanged |
| Controlled 2x2 | A 8/64, B 16/64, C 8/128, D 16/128 | latent dimension and GRU width separately; same data stream (`data_seed` 42) |

In every size study the ODE vector field and decoder keep 64 hidden units, so the added parameters are mostly in the GRU encoder.
''')

# ============================================================================ 04 DATASET
M5 = json.load(open('data/processed/five_shapes/metadata.json')); M3m = json.load(open('data/processed/three_shapes/metadata.json')); M2m = json.load(open('data/processed/two_shapes/metadata.json'))
fe, si = M5['new_family_parameter_ranges']['figure_eight'], M5['new_family_parameter_ranges']['sinusoid']
W('04_DATASET/DATASET_OVERVIEW.md', f'''
# Dataset overview (verified from `data/processed/*/metadata.json`, `src/glo/data.py`, `src/glo/five_shapes.py`, and `audit/verify_1_datasets.py`)

All data are **synthetic**, generated from known equations, so the true path is known everywhere (including the future) and can be used for evaluation. Real GPS data (GeoLife) are used only in the transfer experiments (not distributed; see `GEOLIFE_TRANSFER.md`).

## Five trajectory families

| # | Family | What the motion looks like |
|---|---|---|
| 0 | circle | constant-radius loop |
| 1 | spiral | loop whose radius grows or shrinks exponentially |
| 2 | ellipse | stretched loop at a random orientation |
| 3 | figure-eight | a path crossing itself in an "8" |
| 4 | sinusoid | a sine wave drifting along x |

## Verified statistics

| Item | Value |
|---|---|
| Total trajectories (five-family set) | {M5['n_per_family'] * 5:,} (1,000 per family) |
| Time points per trajectory | {M5['n_points']} equally spaced times on [0, {M5['t_end']:g}] (spacing 10/199 = 0.0503) |
| Noise | Gaussian, SD {M5['base_noise_std']} added to x and y (clean paths kept for evaluation) |
| Observed interval | t <= {M5['forecast_start']:g} (30 observations) |
| Forecast interval | {M5['forecast_start']:g} < t <= {M5['t_end']:g} |
| Split | per family 700 / 150 / 150 -> train {M5['split_counts']['train']:,} / validation {M5['split_counts']['val']} / test {M5['split_counts']['test']} (70% / 15% / 15%) |
| Generation seed | {M5['seed']} (per-family generator seed = 42 + 1009*(family id + 1)) |
| Related datasets | two-family: circle + spiral ({M2m['split_counts']['train']:,} / {M2m['split_counts']['val']} / {M2m['split_counts']['test']}); three-family: + ellipse ({M3m['split_counts']['train']:,} / {M3m['split_counts']['val']} / {M3m['split_counts']['test']}) |
| Relationship | the five-family set contains the three-family trajectories **unchanged**, with the same train/validation/test membership (array-for-array identical) |
| Leakage checks (re-verified) | no trajectory id and no identical path appears in more than one split; observations are only at t <= 7 |

**Figures:** `07_FIGURES/DATASETS/` (all five families, parameter variations (illustration), regular vs irregular sampling, observed vs forecast interval, split organisation).
''')

W('04_DATASET/TRAJECTORY_GENERATION.md', f'''
# Trajectory generation: equations, parameters and ranges (verified from the generation code and metadata)

Common random draws for circle, spiral and ellipse (`src/glo/data.py::sample_trajectory`): radius R ~ U(0.55, 1.5); angular speed omega = s * U(0.75, 1.45) with random sign s = +/-1 (rotation direction); phase phi ~ U(-pi, pi); centre (cx, cy) ~ U(-0.35, 0.35)^2. The angle is theta(t) = omega*t + phi.

| Family | Equation | Parameters and ranges |
|---|---|---|
| circle | x = cx + R cos(theta), y = cy + R sin(theta) | R in [0.55, 1.5]; omega in +/-[0.75, 1.45]; phi in [-pi, pi]; centre in [-0.35, 0.35]^2 |
| spiral (logarithmic) | r(t) = R exp(g t); x = cx + r(t) cos(theta), y = cy + r(t) sin(theta) | as circle, plus growth rate g = +/- U(0.055, 0.12) (positive = expanding, negative = contracting) |
| ellipse | x = cx + cos(alpha) a cos(theta) - sin(alpha) b sin(theta); y = cy + sin(alpha) a cos(theta) + cos(alpha) b sin(theta) | semi-major a in [0.85, 1.65]; semi-minor b = a * U(0.40, 0.78); orientation alpha in [0, pi]; omega, phi, centre as above |
| figure-eight | x = cx + A sin(theta), y = cy + B sin(2 theta) | A in {fe['A_horizontal']}, B in {fe['B_vertical']}, abs(omega) in {fe['abs_omega']} with random sign, phi in [-pi, pi], centre in {fe['centre']} |
| sinusoid | x = x0 + v t, y = cy + A sin(omega t + phi), x0 = cx - 5v | A in {si['amplitude']}, omega in {si['omega']}, abs(v) in {si['abs_velocity']} with random sign (drift direction), phi in [-pi, pi], centre in {si['centre']}; x0 = cx - 5v centres the path on cx over t in [0, 10] |

**Meaning of the parameters (simple English).** R / A / B = size of the loop or wave; omega = how fast it turns (sign = clockwise or counter-clockwise); phase = where on the loop it starts; centre = where on the plane it sits; growth g = how fast a spiral expands or contracts; a, b, alpha = the two radii and the tilt of an ellipse; v = the sideways drift speed of the sinusoid.

**How generation works.** For each family, a generator seeded as 42 + 1009*(id + 1) draws the parameters of 1,000 trajectories in turn, evaluates the equation on the 200-point time grid, adds Gaussian noise (SD 0.05) to make the noisy copy, and finally shuffles the 1,000 trajectories into 700 / 150 / 150 for train / validation / test. The circle, spiral and ellipse arrays in the five-family set are copied unchanged from the three-family set (checked array by array).

**Shape checks (saved in `results/five_shapes/dataset_checks.json`, no failures).** Figure-eight: all 1,000 cover a full eight with alternating centre crossings (largest vertical offset at a centre crossing 3.7e-05 of B; median y-to-x zero-crossing ratio 2.0). Sinusoid: all 1,000 have monotone x and the y half-range equals the amplitude (ratio 0.9994 to 1.0000).

**Note on illustrations.** `fig2_parameter_variations_illustration.png` is drawn from these equations with hand-chosen parameter values; it illustrates the meaning of parameters and is **not** experimental data.
''')

W('04_DATASET/DATA_PREPROCESSING.md', f'''
# Data preprocessing (verified from `data.py`, `pipeline.py`, `metadata.json`)

1. **Storage.** Each split is saved as `train.npz`, `val.npz`, `test.npz` with arrays: time (200), clean xy, noisy xy, family id, trajectory id, parameters, and a stored default observation mask (not used by the training or evaluation code; used only by the leakage validator).
2. **Normalisation.** Coordinates are normalised as (xy - mean) / scale, with mean and scale computed **from the training split only** (scale = root-mean-square deviation from the mean over all coordinates). Values: five-family mean [{M5['normalization_mean'][0]:.5f}, {M5['normalization_mean'][1]:.5f}], scale {M5['normalization_scale']:.5f}; three-family scale {M3m['normalization_scale']:.5f}; two-family scale {M2m['normalization_scale']:.5f}. Each model is evaluated with the constants of the dataset it was trained on.
3. **Time scaling.** Time enters the encoder as t/T with T = 10.
4. **Observation selection.** See `IRREGULAR_OBSERVATIONS.md`.
5. **Targets.** Clean (noise-free) paths on the full grid are the evaluation targets; they are never shown to the encoder.
6. **Noise.** The noisy copy is generated once and stored, so each trajectory's noise realisation is fixed; only *which* times are observed changes between batches.
''')

W('04_DATASET/TRAIN_VALIDATION_TEST_SPLIT.md', f'''
# Train / validation / test split (verified)

| Dataset | Train | Validation | Test | Per family (train / val / test) |
|---|---:|---:|---:|---|
| two-family | {M2m['split_counts']['train']:,} | {M2m['split_counts']['val']} | {M2m['split_counts']['test']} | 700 / 150 / 150 |
| three-family | {M3m['split_counts']['train']:,} | {M3m['split_counts']['val']} | {M3m['split_counts']['test']} | 700 / 150 / 150 |
| five-family | {M5['split_counts']['train']:,} | {M5['split_counts']['val']} | {M5['split_counts']['test']} | 700 / 150 / 150 |

- **Ratio:** 70% / 15% / 15%, stratified by family (the same proportions in each family).
- **Purpose:** *train* = fit the weights and the normalisation; *validation* = choose the checkpoint (and, earlier, the configuration); *test* = final scores only.
- **No leakage (independently re-verified in this project):** zero overlap of trajectory ids between splits, zero identical paths across splits, nearest-neighbour path distances from test to train as large as between training trajectories, and observation times only at t <= 7.
- **Nesting:** the two-family, three-family and five-family sets share the same circle / spiral / ellipse trajectories and split membership (array-for-array identical where they overlap).
- **GeoLife (real data):** splits are by **user** (30% development users; held-out test users), 1,000 test segments from 97 users; development users were further split into training and validation users for the GeoLife-trained models. Test users were never used for training or selection.

Figure: `07_FIGURES/DATASETS/fig5_dataset_split_organisation.png`.
''')

W('04_DATASET/IRREGULAR_OBSERVATIONS.md', '''
# Irregular observation sampling (verified from `src/glo/pipeline.py::prepare_batch` and the saved evaluation observations)

**Procedure (every batch, and at evaluation).**

1. The model works on a query grid of times (100 times during training, 200 during evaluation; equally spaced indices of the 200-point trajectory).
2. Only grid times with t <= 7 are allowed as observation times (about 70 of 100 in training; 140 of 200 in evaluation).
3. The first observation is **always at t = 0** (index 0). The other 29 are drawn **at random without replacement** from the remaining allowed times (random ranks, top-29), then sorted in time.
4. The observed values are the stored **noisy** coordinates at those times; the encoder receives them together with their (normalised) times and gaps.
5. **Training:** a fresh random subset is drawn for every trajectory in every batch. **Evaluation:** one fixed draw (observation seed 123), identical for every model compared.

**What the saved evaluation shows** (750 test trajectories, 30 observations each): 750 distinct sets of observation times; gaps between consecutive observed times from 0.05 to 2.16 (mean 0.23); all times are multiples of the 0.05 grid; the latest observed time is 6.985 (< 7). So the sampling is irregular in spacing but discretised to the dataset's grid, and it always includes t = 0.

**What is held out.** The 110 other context times (t <= 7, not observed) are used to score reconstruction; all 60 times with t > 7 are used to score the forecast.

**Related finding.** Models trained with exactly 30 observations work only near that count (validation reconstruction RMSE 0.061 at 30 observations but 0.494 at 20, 0.363 at 40 and 0.595 at 50; `results/validation/nobs_ablation`).

Figures: `07_FIGURES/DATASETS/fig3_regular_vs_irregular_sampling.png`, `fig4_observed_vs_forecast_interval.png`.
''')

# ============================================================================ 05 METHODOLOGY
W('05_EXPERIMENTAL_METHODOLOGY/COMPLETE_METHODOLOGY.md', f'''
# Complete methodology (the research workflow)

![workflow](../07_FIGURES/ARCHITECTURE/fig_methodology_flowchart.png)

| Step | What is done | Where verified |
|---|---|---|
| 1 Dataset generation | five families x 1,000 trajectories, 200 time points on [0, 10], noise SD 0.05, seed 42 | `data.py`, `five_shapes.py`, `scripts/01, 30` |
| 2 Training / validation / test split | 700 / 150 / 150 per family, stratified; no id shared | `validate_dataset`, audit |
| 3 Normalisation | mean and scale from the training split only | `metadata.json` |
| 4 Irregular observation sampling | 30 random times with t <= 7, first at t = 0 | `prepare_batch` |
| 5 Input preparation | features [x, y, t/T, dt/T], newest first | `RecognitionGRU` |
| 6 GRU encoding | final hidden state -> mean, log-variance | `model.py` |
| 7 Initial latent state | z0 = mu + sigma*eps (train), mu (test) | `model.py` |
| 8 Continuous-time evolution | RK4 integration of dz/dt = f(z) | `solver.py` |
| 9 Coordinate decoding | MLP maps z(t) to (x, y) | `model.py` |
| 10 Training objective | negative ELBO on observed points + KL | `batch_objective` |
| 11 Model training | Adam, cosine learning rate, 300 epochs | `scripts/02_train_model.py` |
| 12 Checkpoint selection | lowest validation negative ELBO | training script |
| 13 Reconstruction evaluation | RMSE on unobserved context points (t <= 7) | `evaluation.py` |
| 14 Forecast evaluation | RMSE on 7 < t <= 10 | `evaluation.py` |
| 15 Cross-family evaluation | test on families absent from training (zero-shot) | `scripts/10, 31` |
| 16 Statistical comparison | paired bootstrap, Wilcoxon, Holm; Welch across seeds | `scripts/09, 32, 33` |

**Reconstruction vs forecasting evaluation.** *Reconstruction* asks: inside the period where the model has seen some points, how well does it fill the gaps (110 unobserved times)? It is an interpolation problem and is what the training loss directly encourages. *Forecasting* asks: how well does it predict a period it has never seen (60 times with t > 7)? It is an extrapolation problem and is not directly trained. They are always reported **separately**, and never mixed into one number.

**Fairness rules used.** Identical observations (times and noise) for every model in a comparison (asserted in the evaluation scripts); each model normalised with its own training constants; test split never used for selection; checkpoints unchanged during evaluation (SHA-256 checked).
''')

W('05_EXPERIMENTAL_METHODOLOGY/TRAINING_PROCEDURE.md', f'''
# Training procedure (verified from `scripts/02_train_model.py`, configs, training histories and logs)

| Setting | Verified value |
|---|---|
| Optimiser | Adam |
| Learning rate | 3e-3, cosine annealing to 1e-5 over the 300 epochs (the first comparison of the original-size model used 1e-3) |
| Batch size | 32 |
| Epochs | 300 (early stopping with patience 60 never triggered in the runs checked) |
| Optimiser steps | 110 per epoch for five-family data (3,500 / 32, last batch partial) = **33,000**; 66 per epoch for the three-family model = 19,800; two-family 44 per epoch |
| Gradient clipping | norm 1.0 |
| Loss | negative ELBO, sigma 0.05, KL weight 1.0 |
| Training query grid | 100 times; 30 observations per trajectory, redrawn every batch |
| Random seeds | 42 for most runs; 123, 2024, 2025, 2026 added for the two-family multi-seed studies; the 2x2 ablation additionally fixes `data_seed` 42 (separate generators for batch order and observation draws) |
| Solver | torchdiffeq `rk4`, step 0.1, no adjoint |
| Hardware | CPU only: AMD EPYC 7313 (16 cores, 32 threads), 125 GB RAM; 4 threads per run; PyTorch 2.5.1, torchdiffeq 0.2.5, Python 3.12.4 |
| Run time | about 16-17 s per epoch; {cap('models.baseline.training_minutes', '.0f')} min (earlier 8/64 run), {cap('models.medium.training_minutes', '.0f')} min (16/128), {cap('models.large.training_minutes', '.0f')} min (32/128); 80.6 to 81.5 min for the four ablation cells (several runs executed in parallel) |
| Checkpoint | saved whenever the validation negative ELBO improves; the last saved one is the selected checkpoint (epochs 294-299 of 300 in the ablation) |
| Per-epoch log | training and validation loss (NLL and KL), validation context and forecast RMSE, gradient norms, function evaluations, seconds |

**Not converged.** In the 2x2 ablation the selected epochs are 294-299 of 300 and the validation loss was still falling slowly, so the models are not fully converged.
''')

W('05_EXPERIMENTAL_METHODOLOGY/EVALUATION_METRICS.md', '''
# Evaluation metrics (verified from `src/glo/pipeline.py::infer_test_metrics`, `evaluation.py`)

All metrics are computed in **original coordinate units** after de-normalising predictions, **per trajectory**, and then averaged over trajectories.

| Metric | Definition | Interval |
|---|---|---|
| **Reconstruction RMSE** | sqrt( sum of squared (x, y) errors over the context times that were **not** observed / (2 * number of such times) ) | t <= 7, 110 unobserved times of 140 |
| **Forecast RMSE** | sqrt( mean over the future times and both coordinates of the squared error ) | 7 < t <= 10 (60 times) |
| Also available (not used for headline results) | forecast MAE, ADE (average displacement), FDE (final displacement), Chamfer distance | |

**Aggregation.** Tables report the **mean (and median, SD) of per-trajectory RMSE**. A different quantity, the *pooled* RMSE (square root of the mean squared error over all points of all trajectories), is stored separately and is never mixed with the per-trajectory mean.

**Baselines.** *Linear interpolation* of the 30 observed points (reconstruction); *hold-last* = stay at the last observed point; *constant velocity* = extrapolate the velocity of the last two observations (forecast). Both baselines see exactly the same observations as the models.

**Statistics.** Paired differences on identical trajectories and observations: bootstrap 95% intervals (resampling test trajectories), Wilcoxon signed-rank test, Holm correction for many comparisons. Across training seeds: mean +/- SD with a Student-t 95% interval, Welch t-test for shared vs specialist. **Important:** trajectory-level intervals and p-values describe only the variation across test trajectories for the specific trained models; they do not describe variation across training runs.

**GeoLife.** Median per-segment RMSE in metres at the recorded timestamps against the recorded GPS positions; user-cluster bootstrap intervals.
''')

W('05_EXPERIMENTAL_METHODOLOGY/EXPERIMENTAL_CONFIGURATIONS.md', f'''
# Experimental configurations (what was run, under which conditions)

| ID | Experiment | Data | Models | Seeds | Training budget | Test protocol | Evidence |
|---|---|---|---|---|---|---|---|
| E1 | Failure diagnosis; original-size vs Model D | two-family | original-size (latent 4, GRU 25), candidates B-D | 42 | 300 epochs | validation split for selection; test once | single run |
| E2 | Reproducibility of Model D | two-family | Model D | 42, 123, 2024, 2025, 2026 | 300 epochs | 300 test trajectories | multi-seed |
| E3 | Shared vs specialist | two-family | shared Model D; circle-only and spiral-only specialists | shared 5; specialists 3 (42, 123, 2024) | shared 1,400 x 300 epochs; specialists 700 x 600 (same steps) | circle/spiral test (150 each) | multi-seed |
| E4 | Three families; two-family models on unseen ellipses | three-family | three-family model; five two-family models | 3-family 42; two-family 5 | 300 epochs | 450 test (150 ellipses) | multi-seed (ellipse test) |
| E5 | Conditioning | two-family | M1, M2, M3 (5 seeds), M2aug (3 seeds) | 42, 123, 2024, 2025, 2026 | 300 epochs; `data_seed` = seed | 300 test | multi-seed |
| E6 | Observation count | two-family | Model D | 42 | - | validation split, 5 to 100 observations | single run |
| E7-8 | GeoLife | real GPS | frozen three-family model; GeoLife-trained (scratch, fine-tuned) | 42 | 300 epochs | 1,000 held-out segments, 97 users | single run |
| E9 | Five-family experiment | five-family | frozen three-family model; five-family Model D | 42 | 33,000 steps (five) vs 19,800 (three) | 750 test | single run |
| E10 | Capacity scaling | five-family | 8/64, 16/128, 32/128 | 42; `data_seed` unset | 33,000 steps | 750 test | single run |
| E11 | Controlled 2x2 | five-family | A, B, C, D | 42 with `data_seed` 42 | 33,000 steps | 750 test, identical observations | single run |

**Do not assume the experiments were run under identical conditions** (see `02_EXISTING_RESEARCH/ORIGINAL_VS_OUR_IMPLEMENTATION.md`).
''')

W('05_EXPERIMENTAL_METHODOLOGY/EXPERIMENT_WORKFLOW.md', '''
# Experiment workflow (order of the research, and the scripts used)

1. **Original-size model and first diagnosis** -> the first checkpoint was from an unfinished run; Model D chosen on validation (`reports/reconstruction_failure_analysis.md`, `reports/improved_two_shapes_results.md`).
2. **Validation study** (5 seeds; specialists; forecast analysis; observation-count and noise sensitivity; three families and unseen ellipses) -> `scripts/06-12`, `reports/validation_study_report.md`.
3. **Dynamics-conditioning study** -> `scripts/12`, `reports/conditioning_study_report.md`.
4. **Packaging of frozen models and inference** -> `scripts/package_models.py`, `predict_trajectory.py`.
5. **GeoLife transfer** (frozen model) -> `scripts/20-21`, `reports/geolife_*`; then **GeoLife-trained models** (`scripts/22-23`).
6. **Five-family dataset and zero-shot test** -> `scripts/30-31`, `reports/five_shapes_report.md`.
7. **Capacity scaling** -> `scripts/32`; **checkpoint-selection audit and controlled 2x2 ablation** -> `scripts/33-34`.
8. **Independent audit and correction of reports** -> `PROJECT_COMPLETION_AUDIT.md`, `audit/verify_*.py`.
9. **Results analysis and presentation material** -> `presentation_results/`, this package.

**Rules followed:** checkpoints are never overwritten; test results are never used for selection; every new result directory is separate from earlier ones.
''')
