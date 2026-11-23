# Experiment 6: controlled 2x2 ablation (latent dimension x GRU width)

## 1. Objective
Separate the effect of latent dimension from encoder width with identical data order and seeds.

## 2. Dataset
`data/processed/five_shapes/`; fixed `--seed 42 --data-seed 42` for all four cells.

## 3. Model configuration
| Model | Latent | GRU | Parameters |
|---|---|---|---|
| A | 8 | 64 | 24,602 |
| B | 16 | 64 | 27,186 |
| C | 8 | 128 | 63,642 |
| D | 16 | 128 | 67,250 |

## 4. Training procedure
`bash scripts/run_ablation_2x2.sh` (four 4-thread jobs). **Single training seed per cell.**

## 5. Evaluation methodology
`python scripts/33_evaluate_ablation_2x2.py` (paired bootstrap and Wilcoxon with Holm correction; factorial effects); `python scripts/34_audit_checkpoint_selection.py`.

## 6. Saved results
Reconstruction A/B/C/D 0.0882 / 0.0462 / 0.0619 / 0.0412 (linear interpolation 0.0672); forecast 0.3998 / 0.5199 / 0.4248 / 0.5936 (hold-last 1.0238). Forecast change vs A: B +30.1%, C +6.3% (Holm p 0.18, not significant), D +48.5%. Larger models are better for 7 < t <= 8 and cross above A at t = 8.09 (B), 8.64 (C), 7.94 (D). Corrected report: `results/ablation_2x2/ABLATION_2X2_REPORT.md` (original kept as `ABLATION_2X2_REPORT.original_uncorrected.md`).

## 7. How to reproduce the evaluation
`python scripts/33_evaluate_ablation_2x2.py`; packaged cells: `--model ablation_2x2_A` ... `ablation_2x2_D`.

## 8. Limitations
**Single training seed.** The same architecture retrained with a different data stream differs by +22.4% and +20.7% in forecast RMSE, and the five-seed two-shape study has a forecast CV of about 23%: the forecast differences above are of that order. No causal mechanism is established.

Back to the [experiment index](../README.md). Full numbers and sources: [`docs/RESULTS.md`](../../docs/RESULTS.md).
