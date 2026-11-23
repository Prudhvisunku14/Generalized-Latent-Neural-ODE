#!/usr/bin/env python3
"""Checkpoint-selection audit from the saved per-epoch histories (validation split only; no test data, no model loading).

  python scripts/34_audit_checkpoint_selection.py

Training (scripts/02_train_model.py) saves a checkpoint whenever the validation -ELBO (beta = kl_weight, NLL of the 30 observed noisy
points with fixed sigma + KL, posterior-mean z0, fixed validation observation draws) improves. Validation forecast and context RMSE
(pooled over the validation split, clean targets) are logged every epoch but never used for selection. Intermediate checkpoints are not
saved, so only validation-split quantities can be audited. min-over-epochs values are post-hoc diagnostics, not a selection rule.
"""
import json
from pathlib import Path
import numpy as np, yaml
root = Path(__file__).resolve().parents[1]
RUNS = {'earlier baseline (8/64)': 'checkpoints/five_shapes/five_shapes_seed42_history.csv',
        'earlier medium (16/128)': 'checkpoints/capacity_scaling/five_shapes_medium_seed42_history.csv',
        'earlier large (32/128)': 'checkpoints/capacity_scaling/five_shapes_large_seed42_history.csv'}
abl = yaml.safe_load((root / 'configs/ablation_2x2.yaml').read_text())
for c, v in abl['variants'].items():
    RUNS[f"cell {c} ({v['latent_dim']}/{v['encoder_hidden']})"] = v['checkpoint'].replace('.pt', '_history.csv')
rows = {}
for name, f in RUNS.items():
    p = root / f
    if not p.exists(): continue
    h = np.genfromtxt(p, delimiter=',', names=True); n = len(h); s = int(np.argmin(h['val_elbo_loss'])); b = int(np.argmin(h['val_forecast_rmse'])); late = slice(49, None)
    corr = lambda a, c: float(np.corrcoef(h[a][late], h[c][late])[0, 1]) if n > 60 else None
    rows[name] = {'epochs_logged': n, 'selected_epoch': int(h['epoch'][s]), 'selected_val_neg_elbo': float(h['val_elbo_loss'][s]),
                  'val_forecast_rmse_at_selected': float(h['val_forecast_rmse'][s]), 'val_context_rmse_at_selected': float(h['val_context_rmse'][s]),
                  'min_val_forecast_rmse': float(h['val_forecast_rmse'][b]), 'min_val_forecast_epoch': int(h['epoch'][b]),
                  'forecast_gap_selected_vs_min': float(h['val_forecast_rmse'][s] / h['val_forecast_rmse'][b] - 1),
                  'train_nll_at_selected': float(h['train_nll'][s]), 'val_nll_at_selected': float(h['val_nll'][s]), 'val_kl_at_selected': float(h['val_kl'][s]),
                  'corr_val_elbo_vs_val_forecast_epochs_ge_50': corr('val_elbo_loss', 'val_forecast_rmse'),
                  'corr_val_context_vs_val_forecast_epochs_ge_50': corr('val_context_rmse', 'val_forecast_rmse'),
                  'val_forecast_sd_last_50_epochs': float(np.std(h['val_forecast_rmse'][-50:], ddof=1)) if n >= 50 else None,
                  'selected_epoch_is_within_last_10_epochs': bool(h['epoch'][s] > n - 10)}
out = Path(root / 'results/ablation_2x2'); out.mkdir(parents=True, exist_ok=True)
(out / 'checkpoint_selection_audit.json').write_text(json.dumps({'note': __doc__.strip(), 'runs': rows}, indent=2))
T = ['| Run | selected epoch | val -ELBO | val forecast RMSE at selected | min val forecast RMSE (epoch) | gap | train NLL / val NLL | val KL | corr(ctx, forecast) epochs>=50 |', '|---|---:|---:|---:|---|---:|---|---:|---:|']
for k, r in rows.items():
    c = r['corr_val_context_vs_val_forecast_epochs_ge_50']
    T.append(f"| {k} | {r['selected_epoch']} | {r['selected_val_neg_elbo']:.2f} | {r['val_forecast_rmse_at_selected']:.4f} | {r['min_val_forecast_rmse']:.4f} ({r['min_val_forecast_epoch']}) | "
             f"{r['forecast_gap_selected_vs_min'] * 100:+.1f}% | {r['train_nll_at_selected']:.1f} / {r['val_nll_at_selected']:.1f} | {r['val_kl_at_selected']:.1f} | {'n/a' if c is None else f'{c:.2f}'} |")
(out / 'checkpoint_selection_audit.md').write_text('\n'.join(T) + '\n'); print('\n'.join(T))
