#!/usr/bin/env python3
"""Phase 1: Model D reproducibility across independent training seeds (two-shape test split).

Every run is evaluated with the same protocol and the same observation subsets (seed 123).
Across-run statistics: mean, sample std (ddof=1) and Student-t 95% CI with n_runs - 1 dof.
"""
import argparse, json, sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from glo.evaluation import load_model, collect, by_family, t_ci95, geometry, baseline

p = argparse.ArgumentParser()
p.add_argument('--checkpoints', nargs='+', required=True, help='label=path pairs')
p.add_argument('--out', default='results/validation/seed_study')
a = p.parse_args()
torch.set_num_threads(4)
root = Path(__file__).resolve().parents[1]; data_root = root / 'data/processed'
out = root / a.out; out.mkdir(parents=True, exist_ok=True)
KEYS = ['recon_rmse', 'forecast_rmse', 'forecast_ade', 'forecast_fde', 'chamfer_full']
FAMS = ['overall', 'circle', 'spiral']
runs = {}
for item in a.checkpoints:
    label, path = item.split('=', 1)
    m, ck = load_model(root / path, 'cpu')
    r = collect(m, ck, data_root, 'two_shapes')
    g = geometry(r)
    hist = list(csv.DictReader(open((root / path).with_name(Path(path).stem + '_history.csv'))))
    runs[label] = {'checkpoint': path, 'seed': ck['config']['training']['seed'], 'best_epoch': ck['epoch'],
                   'epochs_run': len(hist), 'best_val_neg_elbo': ck['val_loss'],
                   'metrics': by_family(r),
                   'rotation_direction_accuracy': float((np.sign(g['omega_hat']) == np.sign(g['omega'])).mean()),
                   'spiral_growth_sign_accuracy': float((np.sign(g['growth_hat']) == np.sign(g['growth']))[r['family'] == 1].mean())}
    print(label, 'epoch', ck['epoch'], {k: round(runs[label]['metrics']['overall'][k]['mean'], 4) for k in KEYS}, flush=True)

agg = {f: {k: t_ci95([runs[l]['metrics'][f][k]['mean'] for l in runs]) for k in KEYS} for f in FAMS}
base = {}
for kind in ('const_pos', 'const_vel'):
    m, ck = baseline(kind, 'two_shapes')
    base[kind] = by_family(collect(m, ck, data_root, 'two_shapes'))
report = {'protocol': 'two_shapes test split (300), n_obs=30 stored noisy observations, n_query=200, '
                      'observation subsets seed 123, posterior mean z0',
          'ci_method': 'Student-t, two-sided 95%, dof = n_runs - 1, over per-run mean metrics',
          'runs': runs, 'across_seeds': agg, 'baselines_single_run': base}
(out / 'seed_study.json').write_text(json.dumps(report, indent=2))

lines = ['| Family | Metric | ' + ' | '.join(f'{l} (seed {runs[l]["seed"]})' for l in runs) + ' | Mean ± SD | 95% CI |',
         '|---|---|' + '---|' * len(runs) + '---|---|']
for f in FAMS:
    for k in KEYS:
        s = agg[f][k]
        lines.append(f'| {f} | {k} | ' + ' | '.join(f'{v:.4f}' for v in s['values'])
                     + f' | {s["mean"]:.4f} ± {s["std"]:.4f} | [{s["ci95_low"]:.4f}, {s["ci95_high"]:.4f}] |')
lines += ['', '| Run | Seed | Best epoch | Epochs run | Best val −ELBO | Rotation dir. acc. | Spiral growth-sign acc. |',
          '|---|---|---|---|---|---|---|']
for l, r in runs.items():
    lines.append(f'| {l} | {r["seed"]} | {r["best_epoch"]} | {r["epochs_run"]} | {r["best_val_neg_elbo"]:.2f} | '
                 f'{r["rotation_direction_accuracy"]:.3f} | {r["spiral_growth_sign_accuracy"]:.3f} |')
(out / 'seed_table.md').write_text('\n'.join(lines) + '\n')

fig, ax = plt.subplots(1, len(KEYS), figsize=(4 * len(KEYS), 4))
for i, k in enumerate(KEYS):
    for j, (f, col) in enumerate((('circle', '#2a9469'), ('spiral', '#e0533a'))):
        v = agg[f][k]['values']
        ax[i].scatter(np.full(len(v), j) + np.linspace(-.12, .12, len(v)), v, color=col, s=26, zorder=3)
        ax[i].errorbar(j + .3, agg[f][k]['mean'], yerr=[[agg[f][k]['mean'] - agg[f][k]['ci95_low']],
                                                       [agg[f][k]['ci95_high'] - agg[f][k]['mean']]],
                       fmt='s', color='k', capsize=4, ms=5)
    ax[i].set_xticks([0, 1]); ax[i].set_xticklabels(['circle', 'spiral']); ax[i].set_title(k); ax[i].grid(alpha=.2)
fig.suptitle(f'Model D across {len(runs)} independent seeds (dots = runs; black = mean with 95% t-CI)')
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(out / 'seed_variability.png', dpi=140); plt.close(fig)
print((out / 'seed_table.md').read_text())
