# Experiment 7: dynamics conditioning

## 1. Objective
Does conditioning the ODE on an inferred dynamics code close the unified-versus-specialist gap?

## 2. Dataset
Circles and spirals (`data/processed/two_shapes/`).

## 3. Model configuration
M1 unified baseline; M2 parameter-matched baseline; M2aug latent 12; M3 conditioned: a 64->4 tanh head gives a constant code c and dz/dt = f(z, c) (25,118 parameters). Configs in `configs/conditioning/`.

## 4. Training procedure
`bash scripts/run_conditioning_study.sh`; 5 seeds per model (3 for M2aug).

## 5. Evaluation methodology
`python scripts/12_conditioning_study.py` (test split; forecast RMSE; probes of the code).

## 6. Saved results
`results/conditioning/` (table, figures, `conditioning_study.json`) and `reports/conditioning_study_report.md`. Conclusion: the hypothesis is **not supported** at this scale.

## 7. How to reproduce the evaluation
`python scripts/12_conditioning_study.py --help`; packaged experimental model: `--model dynamics_conditioned`.

## 8. Limitations
Small scale, two families, 5 seeds.

Back to the [experiment index](../README.md). Full numbers and sources: [`docs/RESULTS.md`](../../docs/RESULTS.md).
