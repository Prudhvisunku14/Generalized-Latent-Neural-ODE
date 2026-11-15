# Reproducibility notes

## Exact numerical identity is not expected

This project reproduces the three experimental families and paper-facing outputs, but several implementation details are not fully specified in Chen et al. (2018).

1. **MNIST solver stack** — the paper used implicit Adams through LSODE/VODE with an Autograd/TensorFlow/Fortran stack.  The modern reproduction uses PyTorch + `torchdiffeq`; the authors' public `odenet_mnist.py` is itself described as a reproduction rather than the original stack.
2. **Figure 4** — the paper states the main optimizer families and iteration budgets but not every learning rate, batch size, initialization, seed, integration interval and plotting detail.
3. **Figure 4 targets** — the three plotted targets are reconstructed as the Rezende & Mohamed U1, U3 and U4 potentials after visual comparison with the paper figure.
4. **Figure 5** — the paper states 64 CNF units and 64 NF layers, but not the complete iteration budget or exact Two Circles/Two Moons noise parameters.  This project reuses the Figure 4 iteration budgets and documents the dataset assumptions in code.
5. **Latent ODE** — the paper gives the core architecture and the 30/50/100-observation evaluation design, but not every split/seed/optimizer/generator detail.  The current pipeline uses a fixed shuffled 80/20 split, independently irregular observation masks per trajectory and noise-free future ground truth for RMSE.
6. **RNN baseline** — Section 5.1 mentions both an RNN baseline and a second version augmented with time differences.  This project uses the time-aware version by default for irregular sampling; pass `--rnn-no-dt` for the plain baseline.

## What counts as a successful reproduction

- **Exp 1** — ResNet/RK-Net/ODE-Net should reach similar MNIST performance while RK/ODE models use substantially fewer parameters than the six-block ResNet; ODE-Net should show tolerance/NFE/computation trade-offs and adaptive NFE growth.
- **Exp 2 Figure 4** — increasing NF depth/CNF width should improve density matching; CNF should qualitatively fit all three targets and generally compare favorably at matched capacity.
- **Exp 2 Figure 5** — CNF should smoothly deform the base Gaussian toward Two Circles/Two Moons; the discrete NF transformation is expected to be less smooth/interpretable.
- **Exp 3** — Latent ODE should reconstruct/extrapolate the spiral family and should be compared with the RNN on the *same* irregular observation masks.

Never alter generated results to match the paper.  Report differences together with the settings used.
