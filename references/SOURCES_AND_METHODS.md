# Research sources and implementation provenance

## Uploaded paper
Chen, Rubanova, Bettencourt & Duvenaud (2018), *Neural Ordinary Differential Equations*, NeurIPS 2018. The attached PDF's Section 5 defines the generative latent dynamics `z0 ~ p(z0); z(t) = ODESolve(z0,f,t); x(t) ~ p(x|z(t))`. Section 5.1 uses 1000 clockwise/counterclockwise spiral trajectories, a 4-dimensional latent state, RNN encoder with 25 hidden units, dynamics network with 20 hidden units, decoder with 20 hidden units, and reports RMSE with 30/50/100 observations.

## Source repositories inspected, not copied wholesale

1. https://github.com/rtqichen/torchdiffeq
   - `README.md`: `odeint(func, y0, t)` / `odeint_adjoint` interface, PyTorch GPU support, `dopri5` and `rk4` solvers, nn.Module requirement for adjoints.
   - `examples/latent_ode.py`: backward RNN inference, Gaussian initial-state posterior, KL objective, decoder, spiral generation. Its implementation details and dataset generation are not identical to the concise architecture description in the paper; e.g. the example dynamics MLP has two hidden layers, and it uses an Archimedean-style spiral generator.
   - `examples/ode_demo.py`: ODE fitting demo (see repository).
   - License: MIT, copyright Ricky Tian Qi Chen (2018).
2. https://github.com/YuliaRubanova/latent_ode
   - `README.md`: later Latent ODE experiments and ODE-RNN encoder, code examples for irregular temporal data and optional Poisson event likelihood.
   - This is a separate 2019 research work; its ODE-RNN encoder is not the same as the 2018 plain RNN recognition network.

## Project-specific decisions

- `src/glo/` is newly written implementation inspired by the published methods; the upstream source files are *not* vendored or represented as original contributions.
- This project generates **logarithmic** spirals and circles/ellipses, not the exact Chen et al. benchmark spirals. Accordingly, performance from this dataset must not be compared numerically to their Table 2 as if the dataset were identical.
- The first comparison is one unconditional shared Latent ODE on circle+spiral and then circle+spiral+ellipse. Conditioning (`--cond-dim 4`) is an explicitly labeled experimental extension.
- Training objective `observed` uses a Gaussian likelihood on sparse, noisy prefix observations + KL, resembling the variational approach in the paper. `full_clean` is a separately flagged synthetic-supervision ablation; it uses clean dense future targets during training, never during test-time inference.
- For a prediction evaluation, *only* noisy observations from `t <= 7` are encoded; ground-truth future points are used only for testing and optional supervised training.
- Dataset noise/sampling/parameter ranges are project choices, not claims about the original paper.
- A local RK4 integrator is provided for offline smoke tests where `torchdiffeq` cannot be installed. For full research runs, install the official library and use its `dopri5` solver with selected tolerances. The offline fallback differentiates through its RK4 steps and does not implement the adjoint method.
