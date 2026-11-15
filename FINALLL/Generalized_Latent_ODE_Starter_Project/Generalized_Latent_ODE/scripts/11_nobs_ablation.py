#!/usr/bin/env python3
"""Ablation: Model D (fixed 30 training observations) vs the same configuration trained with a
random 10-50 observations per batch. Evaluated across observation counts on the validation split
(which motivated the ablation) and the test split. Identical observation subsets for both models.
This is NOT part of Model D and does not replace any requested-phase result.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from glo.evaluation import load_model, collect

p = argparse.ArgumentParser()
p.add_argument('--fixed', default='checkpoints/improved_two_shapes/D_candidate_lr3e-3.pt')
p.add_argument('--variable', default='checkpoints/validation/ablation_variable_nobs_seed42.pt')
p.add_argument('--levels', type=int, nargs='+', default=[5, 10, 15, 20, 25, 28, 30, 32, 35, 40, 50, 70, 100])
p.add_argument('--out', default='results/validation/nobs_ablation')
a = p.parse_args()
torch.set_num_threads(4)
root = Path(__file__).resolve().parents[1]; data_root = root / 'data/processed'
out = root / a.out; out.mkdir(parents=True, exist_ok=True)
models = {'fixed_n30 (Model D)': load_model(root / a.fixed, 'cpu'), 'variable_n10-50 (ablation)': load_model(root / a.variable, 'cpu')}
res = {}
for split in ('val', 'test'):
    res[split] = {}
    for name, (m, ck) in models.items():
        res[split][name] = {}
        for n in a.levels:
            r = collect(m, ck, data_root, 'two_shapes', split=split, n_obs=n, seed=31)
            res[split][name][str(n)] = {f: {k: float(r[k][r['family'] == i].mean()) for k in ('recon_rmse', 'forecast_rmse', 'forecast_fde')}
                                        for f, i in (('circle', 0), ('spiral', 1))}
            res[split][name][str(n)]['overall'] = {k: float(r[k].mean()) for k in ('recon_rmse', 'forecast_rmse', 'forecast_fde')}
report = {'note': 'observation subsets seed 31 (same for both models); n_obs >= 2; stored noise 0.05; '
                  'levels outside the training range 10-50 test extrapolation in observation count',
          'checkpoints': {k: str(v[1]['epoch']) for k, v in models.items()}, 'results': res}
(out / 'nobs_ablation.json').write_text(json.dumps(report, indent=2))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for i, k in enumerate(('recon_rmse', 'forecast_rmse')):
    for name, col in zip(models, ('#e0533a', '#2578bc')):
        for split, ls in (('val', ':'), ('test', '-')):
            ax[i].plot(a.levels, [res[split][name][str(n)]['overall'][k] for n in a.levels], ls, marker='o', ms=3,
                       color=col, label=f'{name} [{split}]')
    ax[i].axvline(30, color='gray', ls=':', lw=.8); ax[i].axvspan(10, 50, color='#2578bc', alpha=.06)
    ax[i].set_xscale('log'); ax[i].set_xticks(a.levels); ax[i].set_xticklabels(a.levels, fontsize=7)
    ax[i].set_yscale('log'); ax[i].set_xlabel('number of observations (t ≤ 7)'); ax[i].set_title(f'two-shape {k}')
    ax[i].grid(alpha=.2); ax[i].legend(frameon=False, fontsize=7)
fig.suptitle('Observation-count robustness: fixed 30 vs random 10–50 training observations (shaded = ablation training range)')
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out / 'nobs_ablation.png', dpi=140); plt.close(fig)
for split in ('val', 'test'):
    for name in models:
        print(split, name, {n: (round(res[split][name][str(n)]['overall']['recon_rmse'], 3), round(res[split][name][str(n)]['overall']['forecast_rmse'], 3)) for n in a.levels})
