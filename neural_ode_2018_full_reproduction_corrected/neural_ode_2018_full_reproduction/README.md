# Neural ODE (NeurIPS 2018) — reproduction of Sections 3, 4 and 5

Reproduces, from actual model computation only (nothing from the paper is hard-coded or drawn):

| Section | Paper item | Script | Output |
|---|---|---|---|
| 3 MNIST | Table 1, Figure 3 | `run_mnist.py` | `results/mnist/table1.csv`, `figure3.png` |
| 4 CNF | Figure 4 (NF vs CNF density matching) | `run_cnf_fig4.py` | `results/cnf/figure4.png` |
| 4 CNF | Figure 5 (max-likelihood CNF vs NF) | `run_cnf_fig5.py` | `results/cnf/figure5.png` |
| 5 Latent ODE | Table 2, Figure 8 | `run_latent_ode.py` | `results/latent_ode/table2.csv`, `latent_ode_figure.png` |

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python run_all_paper.py --mode smoke      # minutes; writes to results_smoke/ (functional check only)
python run_all_paper.py --mode paper      # full paper-scale settings; writes to results/
python run_mnist.py --mode paper          # individual experiments (same for the others)
python run_cnf_fig4.py --mode paper
python run_cnf_fig5.py --mode paper
python run_latent_ode.py --mode paper
python tests/validate_cnf_math.py         # CNF/NF maths checks (trace, signs, reversibility, normalisation)
```
All runs use `SEED = 0` (`common.py`, `--seed` to change). Every run **checkpoints and resumes**
(model, optimiser, iteration, RNG state): re-run the same command after an interruption.
**Runtime warning:** PAPER mode is very long (Fig. 4 = 15 NF runs x 500,000 RMSprop steps + 15 CNF runs x 10,000
Adam steps; MNIST = 3 x 160 epochs). A GPU is strongly recommended.
In SMOKE mode MNIST falls back to random data if it cannot be downloaded (flagged in the output folder).

## Paper values to compare against (published, not produced here)
MNIST (error / params): ResNet 0.41% / 0.60 M, RK-Net 0.47% / 0.22 M, ODE-Net 0.42% / 0.22 M.
Latent ODE RMSE at 30/50/100 observations: RNN 0.3937 / 0.3202 / 0.1813, Latent ODE 0.1642 / 0.1502 / 0.1346.

## Settings taken directly from the paper
ResNet with 6 residual blocks after two downsamplings, ODE-Net/RK-Net replace them (tol 1e-3);
Fig. 4: K, M in {2, 8, 32}, CNF 10,000 Adam iterations, NF 500,000 RMSprop iterations, loss KL(q||p), tol 1e-5;
Fig. 5: CNF 64 hidden units, NF 64 one-unit planar layers, maximum likelihood, stages 5/20/40/60/80/100%;
Latent ODE: 1000 spirals, 100 steps, half cw / half ccw, Gaussian noise, recognition RNN 25 units (backwards in time),
4-D latent, dynamics and decoder 1 hidden layer x 20 units, RNN baseline 25 units, tolerance 1.5e-8, n in {30, 50, 100}, RMSE on 100 later points.
CNF dynamics follow eq. 9-10: `dz/dt = sum_n sigma_n(t) u_n tanh(w_n^T z + b_n)` with hypernetwork parameters,
`d log p/dt = -tr(df/dz)` (closed form, checked against autograd).

## Assumptions (marked `# ASSUMPTION` in code)
Fig. 4/5 batch size 512, Adam lr 1e-3, RMSprop lr 1e-5 / momentum 0.9, integration time T = 10, base Gaussian
(Fig. 5 variance 0.1; Fig. 4 learned diagonal), hypernetwork width 32 and 1/M output initialisation, Fig. 5 iteration
budgets (reuse Fig. 4's), Two-Moons noise/scale, plotting window; Loss-vs-K/M uses K=M in {2,4,8,16,32} (read from the paper's panel);
MNIST 160 epochs / SGD lr 0.1 with decay at 60/100/140 / random-crop augmentation (authors' public example), RK-Net = RK4 step 0.1;
Latent ODE 2000 iterations, Adam lr 1e-2, batch 128, 80/20 split, RMSE against the noise-free spiral, time gaps given to the encoder.
Fig. 4 targets are Rezende & Mohamed U1, U3, U4 (verified against the PDF, see `02_cnf/targets.py`).

See `REPRODUCIBILITY_NOTES.md` for what cannot be reproduced exactly (the original MNIST experiment used a different solver).
