#!/usr/bin/env python3
"""Phase 3: does one unified circle+spiral model lose accuracy vs specialised models?

Each family's test subset (150 trajectories) is evaluated with family-filtered data so the
unified and the specialised model see IDENTICAL observation subsets (seed 123).
Paired per-trajectory tests use the seed-42 models; seed-level comparisons use all seeds given.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, torch
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from glo.evaluation import load_model, collect, summarize, t_ci95

p = argparse.ArgumentParser()
p.add_argument('--unified', nargs='+', required=True, help='paths; FIRST is the paired-test model (seed 42)')
p.add_argument('--circle', nargs='+', required=True)
p.add_argument('--spiral', nargs='+', required=True)
p.add_argument('--out', default='results/validation/shared_vs_separate')
a = p.parse_args()
torch.set_num_threads(4)
root = Path(__file__).resolve().parents[1]; data_root = root / 'data/processed'
out = root / a.out; out.mkdir(parents=True, exist_ok=True)
KEYS = ['recon_rmse', 'forecast_rmse', 'forecast_ade', 'forecast_fde', 'chamfer_full']
rng = np.random.default_rng(0)


def run(paths, fam):
    res = []
    for pth in paths:
        m, ck = load_model(root / pth, 'cpu')
        res.append((pth, ck, collect(m, ck, data_root, 'two_shapes', families=[fam])))
    return res


report = {'protocol': 'two_shapes test split filtered by family (150 each); n_obs=30; observation seed 123; posterior mean',
          'training_budget_note': 'specialists: 700 train trajectories x 600 epochs; unified: 1400 x 300 epochs '
                                  '(same number of optimiser steps, same architecture/config)'}
fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
for r_, fam in enumerate(('circle', 'spiral')):
    U = run(a.unified, fam); S = run(getattr(a, fam), fam)
    assert all(np.array_equal(U[0][2]['obs'], x[2]['obs']) for x in U + S), 'observation subsets differ'
    u0, s0 = U[0][2], S[0][2]
    paired = {}
    for k in KEYS:
        d = u0[k] - s0[k]                                      # >0 => unified worse
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
        paired[k] = {'unified_mean': float(u0[k].mean()), 'specialist_mean': float(s0[k].mean()),
                     'mean_diff_unified_minus_specialist': float(d.mean()),
                     'bootstrap95_mean_diff': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                     'relative_diff_pct': float(100 * d.mean() / s0[k].mean()),
                     'wilcoxon_p': float(stats.wilcoxon(u0[k], s0[k]).pvalue),
                     'fraction_traj_unified_better': float((d < 0).mean())}
    seed_level = {}
    for k in KEYS:
        uv = [float(x[2][k].mean()) for x in U]; sv = [float(x[2][k].mean()) for x in S]
        seed_level[k] = {'unified': t_ci95(uv) if len(uv) > 1 else {'values': uv},
                         'specialist': t_ci95(sv) if len(sv) > 1 else {'values': sv}}
        if len(uv) > 1 and len(sv) > 1:
            seed_level[k]['welch_t_p'] = float(stats.ttest_ind(uv, sv, equal_var=False).pvalue)
    report[fam] = {'unified_checkpoints': [x[0] for x in U], 'specialist_checkpoints': [x[0] for x in S],
                   'paired_seed42': paired, 'seed_level': seed_level,
                   'specialist_best_epoch': [int(x[1]['epoch']) for x in S]}
    # figures: seed-level dots, paired scatter, same-trajectory overlay
    A = ax[r_, 0]
    for j, (lab, R_) in enumerate((('unified', U), ('specialist', S))):
        for i, k in enumerate(('recon_rmse', 'forecast_rmse')):
            v = [x[2][k].mean() for x in R_]
            A.scatter(np.full(len(v), 2 * i + j) + np.linspace(-.1, .1, len(v)), v, s=28,
                      color='#2578bc' if lab == 'unified' else '#999', zorder=3, label=lab if i == 0 else None)
    A.set_xticks([0, 1, 2, 3]); A.set_xticklabels(['recon\nunified', 'recon\nspecial.', 'forecast\nunified', 'forecast\nspecial.'])
    A.set_title(f'{fam}: per-seed test RMSE'); A.grid(alpha=.2); A.legend(frameon=False, fontsize=8)
    A = ax[r_, 1]
    A.scatter(s0['forecast_rmse'], u0['forecast_rmse'], s=10, alpha=.7)
    mx = max(s0['forecast_rmse'].max(), u0['forecast_rmse'].max()) * 1.05
    A.plot([0, mx], [0, mx], 'k--', lw=.7); A.set_xlabel('specialist forecast RMSE'); A.set_ylabel('unified forecast RMSE')
    A.set_title(f'{fam}: paired trajectories (seed 42)'); A.grid(alpha=.2)
    A = ax[r_, 2]; k = int(np.argsort(s0['forecast_rmse'])[len(s0['forecast_rmse']) // 2])
    qt = u0['query_t']
    A.plot(u0['truth'][k][:, 0], u0['truth'][k][:, 1], color='#2a9469', lw=2.6, label='truth')
    A.plot(u0['pred'][k][:, 0], u0['pred'][k][:, 1], color='#2578bc', lw=1.6, label='unified')
    A.plot(s0['pred'][k][:, 0], s0['pred'][k][:, 1], '--', color='#e0533a', lw=1.6, label='specialist')
    A.scatter(u0['obs'][k][:, 0], u0['obs'][k][:, 1], s=10, color='k', zorder=3, label='observations')
    A.set_aspect('equal', adjustable='datalim'); A.grid(alpha=.15); A.legend(frameon=False, fontsize=7)
    A.set_title(f'{fam}: median-difficulty trajectory (specialist-ranked)')
fig.suptitle('Unified circle+spiral model vs specialised single-family models (same architecture, same optimiser steps)')
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out / 'shared_vs_separate.png', dpi=130); plt.close(fig)
(out / 'shared_vs_separate.json').write_text(json.dumps(report, indent=2))
for fam in ('circle', 'spiral'):
    for k in ('recon_rmse', 'forecast_rmse', 'forecast_fde'):
        q = report[fam]['paired_seed42'][k]
        print(fam, k, 'unified', round(q['unified_mean'], 4), 'specialist', round(q['specialist_mean'], 4),
              'diff', round(q['mean_diff_unified_minus_specialist'], 4), 'CI', [round(x, 4) for x in q['bootstrap95_mean_diff']],
              'p', f"{q['wilcoxon_p']:.2g}")
