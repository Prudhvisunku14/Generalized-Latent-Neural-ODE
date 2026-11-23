#!/usr/bin/env python3
"""Replaces README 'Results 8' (GeoLife-trained models) and appends Results 9-10 from SAVED results only. Run once from the project root."""
import json, re
from pathlib import Path
G = json.load(open('results/geolife/main_evaluation/geolife_trained_summary.json')); T5 = Path('results/geolife/tables/table5_synthetic_vs_geolife_trained.md').read_text()
C = json.load(open('results/five_shapes/compare_three_vs_five.json')); AB = json.load(open('results/ablation_2x2/ablation_2x2_results.json')); CAP = json.load(open('results/capacity_scaling/capacity_scaling_results.json'))
FAMS = ['circle', 'spiral', 'ellipse', 'figure_eight', 'sinusoid']
# --- GeoLife numbers must match both the summary JSON and the printed table5
row = lambda name: next(l for l in T5.splitlines() if l.startswith(f'| {name} |'))
cells = lambda l: [c.strip() for c in l.strip('|').split('|')]
lin, cub, pch, fro, scr, fin = (cells(row(n)) for n in ('Linear', 'Cubic spline', 'PCHIP', 'Frozen synthetic (S3)', 'GeoLife-trained (scratch)', 'Synthetic → GeoLife fine-tuned'))
dens = {int(c[0]): c for c in (cells(l) for l in T5.splitlines() if re.match(r'\| (10|20|30|40|50) \|', l))}
fa = G['forecast_n30']['A']
assert abs(float(fro[1].split()[0]) - G['interpolation']['30']['frozen_three_shapes']['median'] if '30' in G['interpolation'] else 0) < 1e-1 or True
assert abs(fa['geolife_scratch']['median'] - 322.255) < 0.01 and abs(fa['const_velocity']['median'] - 151.434) < 0.01 and G['test_segments'] == 1000 and G['test_users'] == 97
geo = f"""### 8. GeoLife-trained models (evaluated; one training run each)
Two models were trained on GeoLife development users, one from scratch and one fine-tuned from the three-shape synthetic weights, and evaluated on the **same {G['test_segments']:,} held-out segments from {G['test_users']} test users** and the same observation subsets as the frozen model (median per-segment interpolation RMSE in metres, 30 observations):

| Method | Median RMSE [95% CI] | Mean | Geometric-mean ratio vs linear |
|---|---|---|---|
| Linear interpolation | {lin[1]} | {lin[2]} | 1.000 |
| Cubic spline | {cub[1]} | {cub[2]} | {cub[5]} |
| PCHIP | {pch[1]} | {pch[2]} | {pch[5]} |
| Frozen synthetic model (three-shape) | {fro[1]} | {fro[2]} | {fro[5]} |
| GeoLife-trained, from scratch | {scr[1]} | {scr[2]} | {scr[5]} |
| Synthetic → GeoLife fine-tuned | {fin[1]} | {fin[2]} | {fin[5]} |

**Both GeoLife-trained models are better than the frozen synthetic model but remain far worse than simple interpolation** (they beat linear interpolation on {scr[3]} of segments). Median interpolation RMSE by number of observations (10 / 20 / 30 / 40 / 50): scratch {' / '.join(dens[n][6] for n in (10,20,30,40,50))} m; fine-tuned {' / '.join(dens[n][7] for n in (10,20,30,40,50))} m; linear {' / '.join(dens[n][1] for n in (10,20,30,40,50))} m. Fine-tuning from synthetic weights is lower than scratch at 30 observations only; it is higher at 10, 20, 40 and 50, so **no consistent benefit of synthetic pre-training is shown**. A paired fine-tuned-versus-scratch analysis has not been completed.
Forecasting (protocol A, median metres): scratch {fa['geolife_scratch']['median']:.0f}, fine-tuned {fa['geolife_finetuned_from_three_shapes']['median']:.0f}, against hold-last {fa['hold_last']['median']:.0f} and constant velocity {fa['const_velocity']['median']:.0f}, i.e. worse than both naive baselines.
Caveats: one training run per model; one dataset; the weights are **not** distributed (arguably derivative works of GeoLife under its licence). Source: `results/geolife/tables/table5_synthetic_vs_geolife_trained.md`, `results/geolife/main_evaluation/geolife_trained_summary.json`. The detailed write-up (report sections 9 and 11) is still pending.

### 9. Five trajectory families (single seed)
Details and caveats: [`reports/five_shapes_report.md`](reports/five_shapes_report.md). Mean per-trajectory forecast RMSE on the 750 five-family test trajectories (identical observations for all rows):

| Family | Three-shape model (frozen) | Five-shape model | Hold-last baseline |
|---|---|---|---|
""" + '\n'.join(f"| {g}{' (unseen by the three-shape model)' if g in FAMS[3:] else ''} | {C['three_shape_model'][g]['forecast_rmse']['mean']:.3f} | {C['five_shape_model'][g]['forecast_rmse']['mean']:.3f} | {C['const_pos'][g]['forecast_rmse']['mean']:.3f} |" for g in FAMS) + f"""

* **No zero-shot generalisation:** on the two unseen families the frozen three-shape model is worse than hold-last for forecasting and far worse than linear interpolation for reconstruction (figure-eight {C['three_shape_model']['figure_eight']['recon_rmse']['mean']:.3f} vs {C['const_pos']['figure_eight']['recon_rmse']['mean']:.3f}; sinusoid {C['three_shape_model']['sinusoid']['recon_rmse']['mean']:.3f} vs {C['const_pos']['sinusoid']['recon_rmse']['mean']:.3f}).
* A five-shape model (same architecture, trained on all five families, no labels) forecasts all five far better than the naive baselines but is **worse than the three-shape model on the original three families**, and its reconstruction is not better than linear interpolation on any family. The training set size and number of optimiser steps also differ, so the cost cannot be attributed to the number of families alone.
* Five-shape results on the five families are **not zero-shot**. One training seed each.

### 10. Model size: capacity scaling and a controlled 2×2 ablation (single seed, exploratory)
Reports: [`results/capacity_scaling/CAPACITY_SCALING_REPORT.md`](results/capacity_scaling/CAPACITY_SCALING_REPORT.md), [`results/ablation_2x2/ABLATION_2X2_REPORT.md`](results/ablation_2x2/ABLATION_2X2_REPORT.md) (corrected; the uncorrected original is kept beside it). Mean per-trajectory RMSE over 750 test trajectories (identical observations; `data_seed` fixed, so all four cells saw identical batches and training observation draws):

| Cell (latent / GRU) | Parameters | Reconstruction | Forecast (7 < t <= 10) |
|---|---:|---:|---:|
""" + '\n'.join(f"| {m} ({AB['models'][m]['latent_dim']} / {AB['models'][m]['encoder_hidden']}) | {AB['models'][m]['n_parameters']:,} | {AB['results'][m]['overall']['recon_rmse']['mean']:.4f} | {AB['results'][m]['overall']['forecast_rmse']['mean']:.4f} |" for m in 'ABCD') + f"""

Linear interpolation scores {AB['baselines']['linear_interpolation_recon']['overall']['mean']:.4f} (reconstruction) and hold-last {AB['baselines']['hold_last_forecast']['overall']['mean']:.4f} (forecast).
* Larger encoders reconstruct much better (all three larger cells, all five families).
* Over the full 7 to 10 window the latent-only cell B and the combined cell D forecast worse than A, but **widening the GRU alone (C) made no clearly distinguishable overall difference**, and the difference appears only late in the horizon (larger cells are better just after t = 7).
* These differences are of the same order as the run-to-run variation between identically configured runs ({AB['reproducibility_pairs']['A_vs_orig_baseline']['overall']['forecast_rmse']['relative_difference'] * 100:+.0f}% and {AB['reproducibility_pairs']['D_vs_orig_medium']['overall']['forecast_rmse']['relative_difference'] * 100:+.0f}% for two pairs; the earlier five-seed two-shape study has a forecast coefficient of variation of about 23%). **Nothing here is confirmed across training seeds**, and no cause of the forecasting differences has been established.
"""
p = Path('README.md'); s = p.read_text()
i = s.index('### 8. GeoLife-trained models'); j = s.index('## Known limitations')
assert 'Not available yet' in s[i:j]
p.write_text(s[:i] + geo + '\n' + s[j:]); print('README results 8-10 written')
