# Reproducibility notes — read before comparing numbers

## Exact numerical identity cannot be guaranteed

This folder is designed to reproduce the **three experimental families and paper-facing outputs**, but it would be misleading to promise bit-for-bit or number-for-number identity with the 2018 paper.

The reasons are concrete:

1. The paper's original MNIST experiment used implicit Adams through LSODE/VODE with an Autograd/TensorFlow/Fortran stack. The authors' own public `torchdiffeq` README describes `odenet_mnist.py` as a *reproduction* whose ODE solver library/method differs from the original experiment, although the results are similar.
2. Figure 4 specifies the major training budgets and optimizers, but not every learning-rate, batch-size, initialization, random seed, integration-interval, and plotting detail.
3. The three Figure-4 toy targets are shown graphically but are not written out in the Neural ODE paper. This folder uses the Rezende & Mohamed (2015) potentials **U1, U3, U4** (rows 1-3). This was checked by rendering the attached PDF at 300 dpi: row 2 is the sine band with a bump branch that rejoins (U3), row 3 is the band that permanently splits (U4); U2 does not appear. An earlier version of this folder used U2 for row 3, which was wrong. The paper's images are also vertically flipped relative to the mathematical orientation (crest of the sine at z1=-1); `figure4` reproduces that orientation. This is a documented reconstruction, not a claim that the paper printed the formulas.
4. Figure 5 states 64 CNF units and 64 NF layers, but does not state its training-iteration counts or exact Two-Circles/Two-Moons noise parameters. The script reuses the Figure-4 training budgets by default and uses the authors' later public CNF demo's Two-Circles noise convention as a guide.
5. The latent-ODE section states the key architecture and RMSE evaluation settings but not the complete train/test split, optimizer schedule, seed, or full trajectory-generation implementation. The data generator here is based on the authors' historical public spiral example and uses an explicit 80/20 train/test split.

## What counts as a successful replication

- MNIST: ODE-Net and RK-Net should use far fewer parameters than the 6-block ResNet and reach similar test error. ODE-Net should show adaptive NFE behavior.
- Figure 4: increasing NF depth/CNF width should improve density fitting; CNF should qualitatively match the three target densities and usually achieve lower KL* than NF at comparable K/M.
- Figure 5: CNF should smoothly deform Gaussian noise into Two Circles/Two Moons; the ordinary planar NF is expected to be less intuitive and can struggle with Two Moons.
- Latent ODE: it should reconstruct/extrapolate the spiral family and generally outperform the simple RNN baseline, with improved prediction as more observations are supplied.

Do not alter generated numbers to make them match the paper. If a result differs, report the difference and the implementation details above.
