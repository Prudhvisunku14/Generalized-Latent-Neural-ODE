# Research development log

## Stage 0: Source review

- Read user-provided NeurIPS 2018 Neural ODE paper, especially pp. 6-8.
- Inspected official torchdiffeq README and `examples/latent_ode.py` through GitHub integration.
- Inspected follow-up Latent ODE repository README and confirmed that its ODE-RNN is an additional model compared with the original 2018 backward-RNN recognition network.
- Decision: implement a clean, independently written prototype, with clear source attribution.

## Stage 1: Datasets

- Generate 1000 circle and 1000 log-spiral instances; reserve 700/150/150 per family.
- Optionally include 1000 ellipses in a distinct second dataset, using identical circle/spiral train/val/test memberships.
- Save clean, noisy, time grid, trajectory family, parameters, unique IDs and precomputed default sparse masks.
- Validate analytical geometric properties, deterministic reproduction and split leakage.

## Stage 2: Modeling

- Baseline: 4D latent state; 25-unit reverse-time GRU encoder; 20-unit dynamics MLP; 20-unit decoder.
- ODE solver from torchdiffeq when installed; offline local test-only RK4 fallback otherwise.
- Conditioning ablation: constant latent code supplied in augmented ODE state.
- Gaussian reconstruction + KL objective. Encoder only sees prefix observations.

## Stage 3: Benchmark protocol

- Test on held-out complete trajectories from known families.
- Evaluate RMSE, MAE, ADE, FDE, geometric Chamfer separately for each family.
- Then compare against discrete RNN baselines and add new shape families. New methods must be compared under matched budgets and seeds.

## Outstanding before publication

- Full multi-seed training; model convergence; proper hyperparameter tuning.
- Reproduction of the original paper's *exact* spiral generator/evaluation procedure if claiming replication of its Table 2.
- Strong RNN and time-aware baseline with matched capacity.
- Dedicated held-out-family dataset and controls; confirm prior novelty with literature search.
