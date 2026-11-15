# Neural ODE (NeurIPS 2018) — clean 3-experiment reproduction

This project is a cleaned and corrected rebuild of the supplied ZIP.  The three experimental families can be run independently on three computers, or sequentially on one GPU.

## What is reproduced

| Experiment | Paper result family | Main outputs |
|---|---|---|
| **Exp 1 — MNIST** | Table 1 + Figure 3 | MLP/ResNet/RK-Net/ODE-Net comparison, NFE/error/time plots |
| **Exp 2 — Normalizing flows** | Figures 4 and 5 | NF vs CNF density matching, loss vs K/M, Two Circles/Two Moons PNG/PDF/GIF |
| **Exp 3 — Latent ODE** | Table 2 + spiral figures | RNN vs Latent ODE RMSE, 30/50/100 trajectory plots, latent-space PNG/PDF/GIF |

Published paper values are kept only as **comparison targets**; reproduced values are always computed by the code.

---

## 1. Install

Use Python 3.10+ and a CUDA-enabled PyTorch build that matches your NVIDIA driver.

```bash
python -m pip install -r requirements.txt
```

Before a long run:

```bash
python check_gpu.py
```

For another installed GPU, e.g. GPU 1:

```bash
python check_gpu.py --gpu 1
```

The script prints the detected GPU, VRAM and a CUDA tensor test.

> Neural ODE/CNF/Latent ODE experiments use FP32 by default.  AMP/FP16 is intentionally not enabled because solver tolerances and NFE are part of the reproduced results.

---

## 2. Run on three different computers

Copy this same folder to all three computers.

### Computer 1 — Experiment 1

```bash
python run_exp1.py --mode paper --require-cuda
```

### Computer 2 — Experiment 2

```bash
python run_exp2.py --mode paper --require-cuda
```

### Computer 3 — Experiment 3

```bash
python run_exp3.py --mode paper --require-cuda
```

Each experiment writes to a separate results/checkpoint folder, so the outputs can later be copied onto one PC without mixing the experiments.

---

## 3. Run everything on one PC

With one GPU, run sequentially:

```bash
python run_all.py --mode paper --require-cuda
```

Do **not** run all three simultaneously on a single GPU unless you deliberately want them to compete for VRAM/compute.  Sequential execution is safer and makes timing measurements meaningful.

---

## 4. Quick smoke test first

Run a short functional test before paper-scale training:

```bash
python run_all.py --mode smoke
```

Smoke mode is only a pipeline check.  Its numerical results must not be compared with the paper.

---

## 5. Windows/Linux shortcuts

Linux/macOS/Git Bash:

```bash
bash scripts/run_exp1.sh
bash scripts/run_exp2.sh
bash scripts/run_exp3.sh
bash scripts/run_all.sh
```

Windows Command Prompt / PowerShell:

```text
scripts\run_exp1.bat
scripts\run_exp2.bat
scripts\run_exp3.bat
scripts\run_all.bat
```

The shortcut scripts forward additional flags. For example, `bash scripts/run_exp1.sh --fresh` forces a clean restart.

---

## 6. Resume and restart behaviour

Checkpointing/resume is automatic.  If power is lost, run the same command again and training continues from the matching checkpoint.

To intentionally restart an experiment/model from zero:

```bash
python run_exp1.py --mode paper --require-cuda --fresh
```

The same `--fresh` flag works for Exp 2 and Exp 3.

---

## 7. Terminal timing

### Experiment 1

Every real MNIST epoch prints:

```text
[odenet] Epoch 042/160 | loss ... | test ... | NFE-F ... | NFE-B ... |
epoch 00:01:27 | total 00:59:43 | ETA 02:47:10 | peak GPU 6420 MB
```

### Experiments 2 and 3

The paper defines these runs in **optimizer iterations**, not dataset epochs.  To keep the terminal useful, iterations are grouped into *logging epochs* without changing any optimizer update.

For example, Figure 4 paper mode uses:

- NF: 500,000 steps = 10 logging epochs of 50,000 steps
- CNF: 10,000 steps = 10 logging epochs of 1,000 steps

Experiment 3 uses 20 optimizer steps per logging epoch by default.  Every logging epoch prints elapsed time, total time, ETA, NFE (where applicable) and peak GPU memory.

---

## 8. Result folders

### Experiment 1

```text
results/exp1_mnist/
├── table1_comparison.csv
├── figure3.png
├── figure3.pdf
├── figure3_stats.csv
├── system_info.json
└── training/
    ├── mlp_epochs.csv
    ├── resnet_epochs.csv
    ├── rknet_epochs.csv
    └── odenet_epochs.csv
```

`table1_comparison.csv` includes paper error, reproduced error, parameter counts, theoretical paper memory complexity, measured peak GPU memory and training time.

### Experiment 2

```text
results/exp2_cnf/
├── figure4_nf_vs_cnf.png
├── figure4_nf_vs_cnf.pdf
├── figure4_losses.csv
├── figure5_transformations.png
├── figure5_transformations.pdf
├── figure5_nll.csv
├── circles_nf_vs_cnf.gif
├── moons_nf_vs_cnf.gif
├── circles_stage_samples.npz
├── moons_stage_samples.npz
└── training/
```

### Experiment 3

```text
results/exp3_latent_ode/
├── table2_comparison.csv
├── rnn_vs_latent_ode_obs30.png/.pdf
├── rnn_vs_latent_ode_obs50.png/.pdf
├── rnn_vs_latent_ode_obs100.png/.pdf
├── latent_space.png/.pdf
├── rnn_vs_latent_ode_obs30.gif
├── latent_space_evolution.gif
├── obs30_predictions.npz
├── obs50_predictions.npz
├── obs100_predictions.npz
└── training/
```

Static PNG figures are saved at 600 dpi.  GIFs are intended for presentations; the `.npz` files retain raw predictions/samples for later replotting.

---

## 9. Useful advanced commands

Only Figure 4:

```bash
python run_exp2.py --part figure4 --mode paper --require-cuda
```

Only Figure 5:

```bash
python run_exp2.py --part figure5 --mode paper --require-cuda
```

Skip GIF generation:

```bash
python run_exp2.py --mode paper --require-cuda --no-gif
python run_exp3.py --mode paper --require-cuda --no-gif
```

Select a different GPU:

```bash
python run_exp1.py --mode paper --device cuda:1 --require-cuda
```

Direct advanced options remain available in the experiment files:

```bash
python 01_mnist/experiment.py --help
python 02_cnf/figure4_density_matching.py --help
python 02_cnf/figure5_maximum_likelihood.py --help
python 03_latent_ode/experiment.py --help
```

---

## 10. Important reproduction note

This is a faithful modern reproduction pipeline, **not a promise of bit-for-bit identity** with the 2018 experiments.  Several low-level settings are not specified in the paper, and the original MNIST experiment used an older LSODE/VODE/Autograd/TensorFlow/Fortran stack.  See `REPRODUCIBILITY_NOTES.md` and `AUDIT_AND_CHANGES.md` before interpreting numerical differences.
