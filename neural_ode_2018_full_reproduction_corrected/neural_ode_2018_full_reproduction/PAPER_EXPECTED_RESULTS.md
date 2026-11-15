# Paper targets to compare against

These are the published results/outputs from Chen et al. (NeurIPS 2018), **not pre-filled reproduction results**.

## Section 3 — MNIST (Table 1)

| Model | Test error | Parameters | Memory | Time |
|---|---:|---:|---|---|
| 1-Layer MLP (LeCun et al. reference) | 1.60% | 0.24 M | — | — |
| ResNet | 0.41% | 0.60 M | O(L) | O(L) |
| RK-Net | 0.47% | 0.22 M | O(L~) | O(L~) |
| ODE-Net | 0.42% | 0.22 M | O(1) | O(L~) |

The MNIST script also produces the four-panel Figure-3-style ODE statistics plot: numerical error, forward NFE/time, forward-vs-backward NFE, and NFE over training.

## Section 4 — Continuous Normalizing Flows

### Figure 4
- NF depths: K = 2, 8, 32
- CNF widths: M = 2, 8, 32
- CNF: 10,000 iterations, Adam
- NF: 500,000 iterations, RMSprop
- Objective: KL(q(x) || p(x))
- Published qualitative result: CNF generally obtains lower loss and improves as width M increases.

### Figure 5
- Datasets: Two Circles and Two Moons
- CNF: 64 hidden units
- NF: 64 stacked one-hidden-unit layers
- Shown transformation stages: 5%, 20%, 40%, 60%, 80%, 100%
- Rows: Density, Samples, NF, with the target shown at the right.

## Section 5 — Latent ODE (Table 2)

| Observations | RNN RMSE | Latent ODE RMSE |
|---|---:|---:|
| 30/100 | 0.3937 | 0.1642 |
| 50/100 | 0.3202 | 0.1502 |
| 100/100 | 0.1813 | 0.1346 |

The paper also shows spiral reconstruction/extrapolation (Figure 8) and latent interpolation (Figure 9).
