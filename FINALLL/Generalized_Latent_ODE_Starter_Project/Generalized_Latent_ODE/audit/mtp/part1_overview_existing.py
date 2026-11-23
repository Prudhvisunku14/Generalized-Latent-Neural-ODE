# sections 01 (overview) and 02 (existing research); executed from build_mtp_package.py
orig_rc = imp('latent_ode_original.overall.recon_rmse.mean', '.3f'); orig_fc = imp('latent_ode_original.overall.forecast_rmse.mean', '.3f')
impr_rc = imp('latent_ode_improved.overall.recon_rmse.mean', '.3f'); impr_fc = imp('latent_ode_improved.overall.forecast_rmse.mean', '.3f')

W('01_PROJECT_OVERVIEW/PROJECT_INTRODUCTION.md', f'''
# Project introduction (simple English)

**Title:** Cross-Family Generalization and Forecasting in Latent Neural Ordinary Differential Equations

**One-sentence summary.** We build and study a *Latent Neural ODE* (a neural network that models motion as a continuously evolving hidden state) that looks at a few noisy points of a 2-D path and must (a) fill in the gaps and (b) predict the future, and we ask whether one shared model can handle several different kinds of paths, including kinds it has never seen.

**What this project is, honestly.** It is an **adapted re-implementation and experimental extension** of the Latent ODE of Chen et al. (2018). It is **not an exact replication** of the paper (different data, solver settings and objective details; the paper's own numbers were not reproduced), and it has **not produced an improved forecasting model**.

---
## A. Neural Ordinary Differential Equations

**Ordinary neural network (for contrast).** A deep network such as a residual network changes its hidden state in a fixed number of discrete steps: h(k+1) = h(k) + f(h(k), theta_k). Each layer is one step.

**Neural ODE.** Imagine making the steps infinitely small. The hidden state then changes *continuously*, and a neural network f describes its **rate of change**:

```
dz(t)/dt = f( z(t), t, theta )            (Chen et al. 2018, Eq. 2-3; theta = the network's weights)
z(t1)  =  z(t0) + integral from t0 to t1 of f( z(t), t, theta ) dt
```

- *What does "continuous-time modelling" mean?* The state exists at every time, not just at the layers of a network, so we can ask for it at **any** time, including irregular times.
- *What is the role of the ODE solver?* The integral cannot be solved by hand because f is a neural network, so a numerical **ODE solver** (such as Runge-Kutta) takes many small steps to compute z(t1) from z(t0). The solver is like the "forward pass".
- *How does it differ from a traditional network?* There is no fixed list of layers; the number of solver steps is a numerical choice, the weights are shared across time, and the model naturally lives in continuous time.
- *Backpropagation:* the paper proposes the adjoint method (a second ODE solved backwards) to save memory. **In this project we do not use it**: we differentiate directly through the solver steps (`adjoint = False`).

---
## B. Latent Neural Ordinary Differential Equations

A **Latent Neural ODE** adds an *encoder* and a *decoder* around the Neural ODE:

```
observations  --encoder-->  initial latent state z0  --Neural ODE-->  z(t) at any time  --decoder-->  predicted observation x(t)
```

- **What is a latent state?** A short hidden vector (here 8 numbers in the baseline model) that summarises "what this particular path is doing". We never observe it; the model learns it.
- **Why is an encoder required?** The ODE needs a starting point. The encoder reads the sparse, noisy, irregularly timed points and produces the starting latent state z0 (as a probability distribution, so the model is a variational auto-encoder).
- **How does the Neural ODE evolve the latent state?** From z0, the learned rate-of-change network f is integrated forward in time, giving the latent state at every time we ask for.
- **How does the decoder generate predictions?** A small network turns the latent state at time t into the (x, y) position at time t.

---
## C. Trajectory modelling vocabulary used throughout

- **Trajectory:** the path of a moving point: a list of (x, y) positions over time.
- **Observations:** here, 30 noisy (x, y) points taken at random times between t = 0 and t = 7. The model never sees anything after t = 7.
- **Reconstruction:** estimating the path *between* the observed points (at times t <= 7 that were not given to the model).
- **Forecasting:** predicting the path in the future, 7 < t <= 10.
- **Irregularly sampled observations:** the observation times are not equally spaced (gaps from 0.05 to 2.16 time units in the saved evaluation).
- **Trajectory family:** a type of path: circle, spiral, ellipse, figure-eight, sinusoid.
- **Shared model vs specialist model:** one model for several families vs one model per family.
- **Cross-family generalisation (zero-shot):** testing a model on a family that was **not** in its training data. (Testing on a family it was trained on is *not* zero-shot.)
- **Baselines:** simple comparisons: linear interpolation (join the dots; reconstruction), hold-last (stay at the last observed point; forecast), constant velocity (keep moving at the last speed; forecast).
- **RMSE:** root-mean-square error: the typical distance between prediction and truth (smaller is better).
''')

W('01_PROJECT_OVERVIEW/RESEARCH_OBJECTIVES.md', '''
# Research objectives

Taken from `README.md` (objectives 1-3) and the later five-family work (objectives 4-5). Status as verified in `PROJECT_COMPLETION_AUDIT.md`.

| # | Objective | Status |
|---|---|---|
| 1 | Establish a working, validated Latent ODE baseline for sparse, noisy 2-D trajectories (circle, spiral), including diagnosing why the first checkpoint failed and showing reproducibility across seeds | Completed (5-seed study on two families) |
| 2 | Test unification: can one model cover several families with no shape label; how does it compare with specialist models; does conditioning the dynamics on an inferred code help? | Completed (shared vs specialist 5 vs 3 runs; conditioning 5 runs) |
| 3 | Test synthetic-to-real transfer on real GPS (GeoLife) | Completed (frozen model and GeoLife-trained models; single run each) |
| 4 | Extend to five families and test zero-shot generalisation to unseen families | Completed (single seed; ellipse test multi-seed) |
| 5 | Probe model size (latent dimension, GRU width) on the five-family data | Completed, single seed, exploratory (controlled 2x2 ablation) |

Two formulations used when judging completeness (from the audit):

- **Objective A:** "systematically investigate reconstruction, forecasting and cross-family generalisation in Latent Neural ODEs" - **partially complete** (multi-seed confirmation of the five-family work, leave-one-family-out tests and stronger baselines are missing).
- **Objective B:** "develop an improved Latent ODE with better forecasting across families" - **not complete**; no improved method was developed or tested.
''')

W('01_PROJECT_OVERVIEW/RESEARCH_GAP.md', '''
# Research gap (stated carefully)

**What is verified.**

- The experiment in the original 2018 paper on irregular time series (Section 5.1) uses **two directions of one type of spiral** ("bi-directional spiral dataset": half clockwise, half counter-clockwise). It does not evaluate several different trajectory families, and it does not test a model on a family it was not trained on. *(Verified from the arXiv HTML text of the paper.)*
- In this project, we asked what happens when one shared Latent ODE is trained on several different path types and then tested on unseen ones.

**What is NOT verified.**

- Whether "cross-family forecasting with a Latent ODE" is an **open research gap relative to the literature after 2018** has **not** been checked here. That requires a literature review. The project README cites later related work (Latent ODEs for irregularly sampled time series, Rubanova et al. 2019; Neural ODE Processes; GOKU-nets; CoDA; LEADS) but this package does not review them. **NOT VERIFIED.**
- Therefore the slides should say "the original experiment covers only two spiral directions; we extend it to five families", not "no one has studied this".

**How to phrase the motivation on a slide.** "The original latent-ODE experiment uses one family of paths. Real movement is varied. We therefore ask how far one shared latent-ODE model goes across several kinds of paths, and what it costs." (No novelty claim; the README explicitly makes none.)
''')

W('01_PROJECT_OVERVIEW/PROJECT_CONTRIBUTIONS.md', f'''
# Project contributions (what can legitimately be claimed)

**A. Implementation (verified).** A Latent Neural ODE (backward GRU encoder, Gaussian posterior over the initial state, time-invariant neural vector field, torchdiffeq RK4, MLP decoder, negative-ELBO training); generators and validated datasets for five trajectory families with leakage-free splits; a paired per-trajectory evaluation pipeline with bootstrap / Wilcoxon / Holm statistics and simple baselines; frozen packaged models with checksums; a GeoLife evaluation pipeline; 106 passing tests. Independently verified in `PROJECT_COMPLETION_AUDIT.md`.

**B. Experimental contribution beyond the paper's spiral experiment.** (i) shared vs specialist comparison (5 vs 3 training runs); (ii) a dynamics-conditioning study (5 runs; hypothesis not supported); (iii) zero-shot tests on unseen families (negative for figure-eight and sinusoid; partial for ellipses); (iv) a five-family dataset and model; (v) single-seed capacity scaling and a controlled 2x2 ablation; (vi) synthetic-to-real transfer on GeoLife (negative), with scratch and fine-tuned real-data training.

**C. Methodological improvement: NONE.** No method that improves forecasting was developed or validated. "Model D" is a larger configuration of the same architecture (latent 8 and wider networks) that improved on a properly trained original-size model (latent 4, GRU 25, MLP 20) on our two-family data (validation context RMSE 0.335 -> 0.059, forecast RMSE 0.815 -> 0.262; single run each; source `reports/improved_two_shapes_results.md`); this is a hyper-parameter change, not a new method. (The much worse numbers of the very first checkpoint, test reconstruction {orig_rc} and forecast {orig_fc}, came from a run stopped after about 3 epochs and are not evidence about the architecture.) The conditioning variants did not forecast better.

**Phrases to avoid on slides:** "exact replication"; "improved / better forecasting model"; "state of the art"; "proved that capacity hurts forecasting"; "generalises to unseen families".
''')

# ============================================================================ 02 EXISTING RESEARCH
W('02_EXISTING_RESEARCH/ORIGINAL_PAPER_SUMMARY.md', '''
# Original paper summary: Chen, Rubanova, Bettencourt, Duvenaud (2018), "Neural Ordinary Differential Equations" (NeurIPS 2018, arXiv:1806.07366)

**Provenance of this summary.** Read by helper agents from the arXiv HTML version (v6) during this session. The HTML text was only partly available verbatim; items marked **UNVERIFIED** come from a model-generated summary of the remaining text and **must be checked against the PDF before being quoted on a slide**. The PDF itself could not be read as text here, and the paper's figures are **not** included in this package (copyright; SOURCE REQUIRED).

**Central idea (abstract, quoted):** continuous-depth models "have constant memory cost, adapt their evaluation strategy to each input, and can explicitly trade numerical precision for speed". The paper also builds continuous normalizing flows and "show[s] how to scalably backpropagate through any ODE solver".

**Core equation (Section 1, Eq. 2):** `dh(t)/dt = f(h(t), t, theta)`; Section 2 writes the state as z(t) (Eq. 3). The hidden state is the solution of an initial-value problem solved by a black-box ODE solver.

**Stated advantages (Section 1 sub-headings, verified):** memory efficiency; adaptive computation; scalable and invertible normalizing flows; continuous time-series models.

**Adjoint method (Section 2, Figure 2, verified):** gradients are obtained by solving a second, augmented ODE backwards in time, so memory does not grow with the number of solver steps. (This project does **not** use the adjoint method.)

**The three experimental applications used in this package** (details in the three experiment files):
1. Supervised learning: ODE-Net classification on MNIST (Section 3, Table 1, Figures 1 and 3).
2. Continuous normalizing flows (Section 4, Figures 4 and 6, Eq. 8-10).
3. Generative latent-ODE time-series model: bi-directional spirals (Section 5 and 5.1, Table 2).

**Figure list (verified where marked):** Fig 1 residual network vs ODE network (verified); Fig 2 adjoint method as an augmented backward ODE (verified); Fig 3 statistics of a trained ODE-Net (verified); Fig 4 normalizing flow vs CNF density matching, loss vs K/M (verified); Fig 5 not in the readable text; Fig 6 noise-to-data transformations for two circles and two moons (verified); Figs 7-11 (latent-ODE computation graph, intensity, spiral extrapolation, latent-dimension variations, reconstructions) **UNVERIFIED**.

**Other UNVERIFIED claims from the summary:** Section 6 states the default solver tolerance 1.5e-8 was used for sequence modelling and 1e-3 / 1e-5 for classification and density estimation.
''')

W('02_EXISTING_RESEARCH/EXPERIMENT_1_CLASSIFICATION.md', '''
# Experiment 1 of the paper: ODE-Net for supervised classification (MNIST)

## What the paper did (verified from the arXiv HTML text, Section 3 and Table 1)

- **Objective:** show that a residual network can be replaced by a continuous-depth ODE module with comparable accuracy but fewer parameters and constant memory.
- **Dataset:** MNIST handwritten digits.
- **Architecture:** "a small residual network which downsamples the input twice then applies 6 standard residual blocks"; in the ODE-Net the residual blocks are replaced by an ODE-solve module. The exact form of the ODE function and the MNIST tolerance are **not stated** in the readable text.
- **Methodology:** the ODE-Net is trained by backpropagating with the adjoint method (implemented in autograd; solver = "the implicit Adams method implemented in LSODE and VODE" via scipy). An RK-Net (direct backpropagation through a Runge-Kutta integrator) is also compared.
- **Metric:** test error, number of parameters, memory and time complexity.

**Original results (Table 1, MNIST):**

| Model | Test error | Parameters | Memory | Time |
|---|---:|---:|---|---|
| 1-layer MLP (LeCun et al. 1998) | 1.60% | 0.24 M | - | - |
| ResNet | 0.41% | 0.60 M | O(L) | O(L) |
| RK-Net | 0.47% | 0.22 M | O(L~) | O(L~) |
| ODE-Net | 0.42% | 0.22 M | O(1) | O(L~) |

(L = number of layers; L~ = number of function evaluations per forward pass.) Figure 3 shows statistics of a trained ODE-Net: number of function evaluations (NFE) vs error, time, backward/forward NFE ratio (median about 0.8 for tolerances between 1e-5 and 1e-1) and NFE during training.

## Our implementation and results

**NOT PRESENT in this repository.** A search of the repository and its parent folder for classification, MNIST, ODE-Net and CIFAR code or results found nothing. **Status: SOURCE REQUIRED FROM EARLIER PRESENTATION.** Do not invent a reproduced error rate. If the earlier presentation contains one, it must be pasted in with its source and clearly labelled as a separate piece of work from this repository.

## Differences between the original and our work

Cannot be stated for this experiment because no implementation is available here. **NOT VERIFIED.**

## Suggested use on a slide

Present the paper's Table 1 as background ("same accuracy as a ResNet with about a third of the parameters and constant memory"), cite it as the paper's result, and add the placeholder "our reproduction: see earlier presentation".
''')

W('02_EXISTING_RESEARCH/EXPERIMENT_2_CNF.md', '''
# Experiment 2 of the paper: Continuous Normalizing Flows (CNF)

## What the paper did (verified from the arXiv HTML text, Section 4)

- **Objective:** use continuous-time dynamics to build normalizing flows (invertible density models) whose log-density can be computed cheaply.
- **Key equations (verified):** discrete change of variables, Eq. 6: `log p(z1) = log p(z0) - log|det df/dz0|`. **Instantaneous change of variables** (Theorem 1, Eq. 8): `d log p(z(t))/dt = - trace( df/dz(t) )`, assuming f is uniformly Lipschitz continuous in z and continuous in t. For f a sum of M functions (Eq. 10) the cost is only **linear** in the number of hidden units M, instead of cubic for standard normalizing flows.
- **Time dependence:** the parameters may depend on t ("a kind of hypernetwork"), with gates sigma_n(t) in (0, 1).
- **Datasets:** 2-D toy distributions: density matching (target densities; Figure 4) and maximum-likelihood fitting of **two circles** and **two moons** (Figure 6).
- **Models compared:** CNF vs standard normalizing flow (NF). Density matching used K = M = 2, 8, 32; the CNF was trained "for 10,000 iterations using Adam" and the NF "for 500,000 iterations using RMSprop"; loss = KL(q || p). Maximum-likelihood models: CNF with 64 hidden units; NF with 64 stacked one-hidden-unit layers.
- **Original results (qualitative):** "Figure 4 shows that CNF generally achieves lower loss"; for two circles the CNF "rotates the planar flows" to fit; the NF "has difficulty fitting the two moons dataset" (Figure 6). Exact loss values, the number of CNF layers, and the optimiser and training length for the maximum-likelihood runs are **not stated** in the readable text.

## Our implementation and results

**NOT PRESENT in this repository.** A search for normalizing-flow / CNF code or results found nothing. **Status: SOURCE REQUIRED FROM EARLIER PRESENTATION.** Do not invent numbers or figures.

## Differences between the original and our work

**NOT VERIFIED** (no implementation available here).

## Suggested use on a slide

Present the instantaneous change-of-variables equation and the "two circles / two moons" qualitative result as the paper's findings; add the placeholder "our reproduction: see earlier presentation".
''')

W('02_EXISTING_RESEARCH/EXPERIMENT_3_LATENT_ODE.md', f'''
# Experiment 3 of the paper: Latent ODE for irregularly sampled time series, and how our work relates to it

## A. What the paper did (Section 5 and 5.1)

**Verified from the readable HTML text:**

- **Objective:** a generative model of time series in which the latent state follows an ODE, so that irregularly sampled data are handled naturally.
- **Generative model (Eq. 11-12):** `z_t0 ~ p(z_t0)`; `z_t1 ... z_tN = ODESolve(z_t0, f, theta_f, t0 ... tN)`; each `x_ti ~ p(x | z_ti, theta_x)`. The dynamics f is **time-invariant** (so a latent state at any time uniquely determines the whole trajectory).
- **Recognition network (verified):** "an RNN, which consumes the data sequentially backwards in time, and outputs q(z0 | x1..xN)"; "25 hidden units" (Section 5.1). RNN cell type not stated.
- **Dataset ("Bi-directional spiral dataset", Section 5.1):** 1000 two-dimensional spirals, each starting at a different point, sampled at 100 equally spaced time steps; half clockwise, half counter-clockwise; Gaussian noise added (level **not stated**); observation times subsampled **without replacement**, n = 30, 50, 100. The train/test split procedure is not stated.
- **Architecture (verified):** latent dimension 4; the dynamics function and the decoder each have one hidden layer of 20 units; RNN encoder 25 units. The optimiser, learning rate and ODE solver for this experiment are **not stated**.
- **Evaluation:** predictive RMSE on "100 time points extending beyond those that were used for training" (extrapolation), Table 2.
- **Original results (Table 2, predictive RMSE on the test set) - reported by two independent reads of the arXiv HTML; verify against the PDF before quoting:**

| Observations | RNN baseline | Latent ODE |
|---|---:|---:|
| 30 / 100 | 0.3937 | 0.1642 |
| 50 / 100 | 0.3202 | 0.1502 |
| 100 / 100 | 0.1813 | 0.1346 |

- **UNVERIFIED details (summary only):** the exact ELBO form and prior N(0, I) (Appendix E); a Poisson-process likelihood variant; Figures 7-11.
- The paper evaluates **only two spiral directions**; no other trajectory families, and no test on unseen families.

## B. What we did (our re-implementation and extension)

1. **Re-implementation of the architecture (original size):** latent 4, GRU 25, one-hidden-layer MLPs of 20 units ({PC['original-size']['total']:,} parameters, counted by instantiating the model class) trained on **our own** two-family data (circle + logarithmic spiral). This is **not** the paper's spiral dataset, so the paper's Table 2 numbers are **not comparable** with ours and were **not reproduced**. Properly trained, this original-size model reached validation context RMSE 0.335 and forecast RMSE 0.815 on our data (single run, validation split; README table, run "A"), against 0.059 / 0.262 for Model D (latent 8, hidden 64, two layers). *Source: `reports/improved_two_shapes_results.md`.*
2. **Extension beyond the paper:** shared vs specialist models, dynamics conditioning, three and five trajectory families, zero-shot tests, model-size studies and real GPS (all in `06_EXPERIMENTAL_RESULTS/`). Model D (24,602 parameters) is the baseline used from then on.

## C. Original spiral experiment vs our five-family extension

| Aspect | Paper (Section 5.1) | Our work |
|---|---|---|
| Families | 2 directions of one spiral type | circle, spiral, ellipse (+ figure-eight, sinusoid in the five-family data) |
| Observations | 30 / 50 / 100 of 100 time points | 30 random times from t <= 7 on a 200-point grid; first always t = 0 |
| Prediction target | extrapolation beyond the training time range | reconstruction (t <= 7) and forecast (7 < t <= 10) |
| Metrics | predictive RMSE | reconstruction RMSE, forecast RMSE, per family, paired statistics |
| Models | RNN vs latent ODE | latent ODE only (plus simple baselines; no RNN baseline) |
| Seeds | not stated | single seed for most; 5 seeds for the two-family studies |
''')

W('02_EXISTING_RESEARCH/ORIGINAL_VS_OUR_IMPLEMENTATION.md', '''
# Original paper vs our implementation: differences and what counts as what

**Classification of our work: an adapted re-implementation of the Latent ODE architecture, plus an experimental extension. NOT an exact replication** (the paper's data, noise level, protocol and numbers were not reproduced) **and NOT an improved model.** Exact replication would require the original experimental setup and numerical results to be reproduced and verified, which has not been done. (Source: `README.md`, `references/SOURCES_AND_METHODS.md`, `PROJECT_COMPLETION_AUDIT.md`.)

## Component-by-component differences (Latent ODE experiment)

| Component | Chen et al. 2018 (Section 5.1) | This project | Evidence |
|---|---|---|---|
| Data | 1000 bi-directional spirals, 100 time points | own synthetic families (circle, logarithmic spiral, ellipse, figure-eight, sinusoid), 200 points on [0, 10], noise SD 0.05 | `src/glo/data.py` |
| Observations | n = 30 / 50 / 100 sampled without replacement | 30 random times from t <= 7 (140 candidate times on the 200-point evaluation grid), first always t = 0 | `pipeline.py::prepare_batch` |
| Encoder | RNN, backwards in time, 25 units (cell type not stated) | GRU, backwards, inputs [x, y, t/T, dt/T], 64 or 128 units | `model.py` |
| Latent size | 4 | 8 (Model D), up to 32 in size studies | configs |
| Dynamics / decoder | one hidden layer of 20 units | two hidden layers of 64 units, tanh | configs |
| ODE function | time-invariant | time-invariant (autonomous) | `model.py::LatentDynamics` |
| Solver | adjoint method described; tolerance details UNVERIFIED/not stated for this experiment | torchdiffeq fixed-step `rk4` (3/8 rule), step 0.1, direct backpropagation, no adjoint | `solver.py` |
| Objective | VAE ELBO (details in Appendix E, UNVERIFIED) | Gaussian NLL, fixed sigma 0.05, on observed points + KL to N(0, I), weight 1 | `pipeline.py::batch_objective` |
| Evaluation | RMSE of extrapolation beyond the training range | reconstruction (withheld context points) and forecast (7 < t <= 10) | `evaluation.py` |
| Baseline | RNN | linear interpolation, hold-last, constant velocity | `evaluation.py::BaselineModel` |

## Which experiments belong to which part

| Part | Experiments |
|---|---|
| Re-implementation of the original architecture (original-size configuration on our data) | E1: failure diagnosis and original-size run (`reports/reconstruction_failure_analysis.md`, `reports/improved_two_shapes_results.md`) |
| Extension (our research questions) | shared vs specialist; three families; dynamics conditioning; observation-count sensitivity; GeoLife; five families; capacity scaling; 2x2 ablation |
| Not in this repository | ODE-Net classification and CNF experiments (SOURCE REQUIRED FROM EARLIER PRESENTATION) |

## The experiments were NOT all run under identical conditions

| Experiment | Dataset | Training budget / seeds | Observation draws / notes |
|---|---|---|---|
| Model D, shared vs specialist, conditioning | two_shapes (1,400 / 300 / 300) | shared: 300 epochs; specialists: 700 trajectories x 600 epochs (same optimiser steps); 5 shared vs 3 specialist runs | observation seed 123 on the 300-trajectory test split |
| Three-family model | three_shapes (2,100 / 450 / 450) | 300 epochs, 66 steps/epoch = 19,800 steps; seed 42 | its own test split |
| Five-family models | five_shapes (3,500 / 750 / 750) | 300 epochs, 110 steps/epoch = 33,000 steps | observation seed 123 on the 750-trajectory test split; **observation subsets differ from the three-shape test split**, so a given model's numbers are not interchangeable across the two splits |
| Capacity scaling | five_shapes | seed 42; `data_seed` unset, so batch order / training draws differ between architectures | single run each |
| 2x2 ablation | five_shapes | seed 42 and `data_seed` 42 for all four cells, identical batch order and training observation draws; initial weights differ (different shapes) | single run each |
| GeoLife | real GPS, development vs held-out users | models trained on development users only, 30 observations | single run each |
''')
