#!/usr/bin/env python3
"""Capacity-scaling study: paired test-set evaluation, statistics and figures (evaluation only; checksums asserted).

  python scripts/32_evaluate_capacity_scaling.py

Models: frozen three-shape model (saved_models/three_shapes_unified), five-shape baseline / medium / large
(configs/capacity_scaling.yaml). Data: five_shapes test split, 750 trajectories (150 per family).
Protocol (identical to scripts/31): 30 stored noisy observations at t <= 7, drawn with observation seed 123 in dataset
order, so every model receives the SAME observation indices, timestamps and noise realisations (asserted);
200-point evaluation grid; posterior mean z0; each model normalised with its own training-set constants.

Metrics, per trajectory, original data units:
  recon_rmse    RMSE over the context timestamps (t <= 7) that were not observed  (interpolation)
  forecast_rmse RMSE over all timestamps with 7 < t <= 10                         (extrapolation)
Aggregation: tables report the MEAN (and median, SD) of per-trajectory RMSE values. The pooled RMSE (square root of the
mean squared error over all points of all trajectories) is stored separately under 'pooled_rmse' and never mixed in.
Paired statistics: per family, paired differences of per-trajectory RMSE; 95% CI of the mean difference from 10,000
trajectory-level bootstrap resamples (seed 0); Wilcoxon signed-rank test; Holm correction over all tests in the table.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, torch, yaml
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from glo.evaluation import load_model, collect, baseline
from glo.inference import sha256_file

torch.set_num_threads(4)
root = Path(__file__).resolve().parents[1]; data_root = root / 'data/processed'
OUT = root / 'results/capacity_scaling'; TAB = OUT / 'tables'; FIG = OUT / 'figures'
for d in (TAB, FIG): d.mkdir(parents=True, exist_ok=True)
CAP = yaml.safe_load((root / 'configs/capacity_scaling.yaml').read_text())
FAMS = ['circle', 'spiral', 'ellipse', 'figure_eight', 'sinusoid']; ORIG = FAMS[:3]; NEW = FAMS[3:]
KEYS = ['recon_rmse', 'forecast_rmse']
CK = {'three_shape': root / 'saved_models/three_shapes_unified/best_model.pt',
      **{v: root / CAP['variants'][v]['checkpoint'] for v in ('baseline', 'medium', 'large')}}
LAB = {'three_shape': 'Three-shape (frozen)', 'baseline': 'Baseline five-shape', 'medium': 'Medium five-shape', 'large': 'Large five-shape'}
COL = {'three_shape': '#d1495b', 'baseline': '#2578bc', 'medium': '#3a9d5d', 'large': '#8a5cc2'}
FCOL = dict(zip(FAMS, ['#2578bc', '#e0a030', '#3a9d5d', '#d1495b', '#8a5cc2']))


def same_obs(a, b):
    return (np.array_equal(a['obs_t'], b['obs_t']) and np.array_equal(a['obs_mask'], b['obs_mask'])
            and np.allclose(a['obs'], b['obs'], atol=1e-5) and np.array_equal(a['ids'], b['ids']))


def pooled(res, sel):
    d2 = ((res['pred'] - res['truth']) ** 2).sum(-1)[sel]                # [n, Q]
    fut = res['query_t'] > 7
    miss = (~res['obs_mask'][sel]) & ~fut[None]
    return {'recon_rmse': float(np.sqrt(d2[miss].sum() / (2 * miss.sum()))), 'forecast_rmse': float(np.sqrt(d2[:, fut].mean() / 2))}


def summary(res, sel):
    return {k: {'mean': float(res[k][sel].mean()), 'median': float(np.median(res[k][sel])), 'sd': float(res[k][sel].std(ddof=1)),
                'n': int(sel.sum())} for k in KEYS}


def boot_ci(d, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.
    for r, i in enumerate(order):
        run = max(run, min(1., (len(ps) - r) * ps[i])); adj[i] = run
    return adj


# ------------------------------------------------------------------------------------------------ evaluate
before = {k: sha256_file(p) for k, p in CK.items()}
R, info = {}, {}
for name, path in CK.items():
    m, ck = load_model(path, 'cpu')
    R[name] = collect(m, ck, data_root, 'five_shapes')
    hist = path.with_name(path.stem + '_history.csv')
    h = np.genfromtxt(hist, delimiter=',', names=True) if hist.exists() else None
    info[name] = {'checkpoint': str(path.relative_to(root)), 'trained_on': ck['experiment'], 'latent_dim': ck['model']['latent_dim'],
                  'encoder_hidden': ck['model']['encoder_hidden'], 'dynamics_hidden': ck['model']['dynamics_hidden'],
                  'decoder_hidden': ck['model']['decoder_hidden'], 'n_parameters': int(sum(p.numel() for p in m.parameters() if p.requires_grad)),
                  'best_epoch': int(ck['epoch']), 'best_val_neg_elbo': float(ck['val_loss']), 'seed': ck['config']['training']['seed'],
                  'epochs_run': int(h['epoch'].max()) if h is not None else None,
                  'training_minutes': float(h['seconds'].sum() / 60) if h is not None else None, 'sha256': before[name]}
    if name == 'three_shape':
        info[name]['history_note'] = 'packaged copy of checkpoints/validation/three_shapes_seed42.pt'
        h3 = np.genfromtxt(root / 'checkpoints/validation/three_shapes_seed42_history.csv', delimiter=',', names=True)
        info[name].update(epochs_run=int(h3['epoch'].max()), training_minutes=float(h3['seconds'].sum() / 60))
for name in R:
    assert same_obs(R['baseline'], R[name]), f'{name} saw different observations'
    assert np.isfinite(R[name]['pred']).all(), f'{name} non-finite predictions'
BL = {k: collect(*baseline(k, 'five_shapes'), data_root, 'five_shapes') for k in ('const_pos', 'const_vel')}
assert all(same_obs(R['baseline'], b) for b in BL.values())
fam = R['baseline']['family']; names = {v: k for k, v in R['baseline']['meta']['family_to_id'].items()}
sel = {f: fam == i for i, f in names.items()}; sel['original_three'] = np.isin(fam, [0, 1, 2]); sel['new_two'] = np.isin(fam, [3, 4])
sel['overall'] = np.ones_like(fam, bool)
GROUPS = FAMS + ['original_three', 'new_two', 'overall']

out = {'protocol': __doc__.split('Metrics')[0].strip(), 'aggregation': 'mean / median / SD of per-trajectory RMSE; pooled RMSE reported separately',
       'models': info, 'observations_identical_across_models': True, 'predictions_finite': True,
       'results': {n: {g: summary(R[n], sel[g]) for g in GROUPS} for n in R},
       'pooled_rmse': {n: {g: pooled(R[n], sel[g]) for g in GROUPS} for n in R},
       'baselines': {'linear_interpolation_recon': {g: summary(BL['const_pos'], sel[g])['recon_rmse'] for g in GROUPS},
                     'hold_last_forecast': {g: summary(BL['const_pos'], sel[g])['forecast_rmse'] for g in GROUPS},
                     'const_velocity_forecast': {g: summary(BL['const_vel'], sel[g])['forecast_rmse'] for g in GROUPS}},
       'relative_change_vs_baseline_of_mean': {n: {g: {k: out_k for k, out_k in
                                                       ((k, float(R[n][k][sel[g]].mean() / R['baseline'][k][sel[g]].mean() - 1)) for k in KEYS)}
                                                   for g in GROUPS} for n in ('three_shape', 'medium', 'large')}}
PAIRS = [('baseline', 'medium'), ('baseline', 'large'), ('medium', 'large'), ('three_shape', 'medium'), ('three_shape', 'large')]
tests = []
out['paired'] = {}
for a, b in PAIRS:
    for g in FAMS:
        for k in KEYS:
            x, y = R[a][k][sel[g]], R[b][k][sel[g]]; d = y - x
            p = float(stats.wilcoxon(x, y).pvalue)
            key = f'{b}_minus_{a}/{g}/{k}'
            out['paired'][key] = {'n': int(len(d)), 'mean_difference': float(d.mean()), 'median_difference': float(np.median(d)),
                                  'bootstrap_ci95_mean_difference': boot_ci(d), 'fraction_improved_by_' + b: float((y < x).mean()),
                                  'relative_change_of_mean': float(y.mean() / x.mean() - 1), 'wilcoxon_p': p}
            tests.append((key, p, (a, b) in PAIRS[:3]))
# Holm within the primary family of tests (the three capacity pairs), and separately within the secondary (vs three-shape)
for primary in (True, False):
    ks = [t for t in tests if t[2] == primary]
    for (key, _, _), adj in zip(ks, holm(np.array([p for _, p, _ in ks]))):
        out['paired'][key]['wilcoxon_p_holm'] = float(adj)
        out['paired'][key]['holm_family'] = 'capacity pairs (30 tests)' if primary else 'vs three-shape model (20 tests)'
after = {k: sha256_file(p) for k, p in CK.items()}
assert before == after, 'a checkpoint changed during evaluation'
out['checkpoints_unchanged'] = True
(OUT / 'capacity_scaling_results.json').write_text(json.dumps(out, indent=2))
np.savez_compressed(OUT / 'predictions_test.npz', ids=R['baseline']['ids'], family=fam, query_t=R['baseline']['query_t'],
                    truth=R['baseline']['truth'], obs=R['baseline']['obs'], obs_t=R['baseline']['obs_t'],
                    **{f'pred_{n}': R[n]['pred'] for n in R})

# ------------------------------------------------------------------------------------------------ tables
M = list(R)
def fmt(n, g, k):
    s = out['results'][n][g][k]; return f"{s['mean']:.4f} ({s['median']:.4f}; {s['sd']:.4f})"
for k, title, bl in (('recon_rmse', 'Reconstruction RMSE of withheld context points (t <= 7)', [('linear interp.', 'linear_interpolation_recon')]),
                     ('forecast_rmse', 'Forecast RMSE (7 < t <= 10)', [('hold last', 'hold_last_forecast'), ('const. velocity', 'const_velocity_forecast')])):
    T = [f'**{title}.** Mean (median; SD) of per-trajectory RMSE over the 150 test trajectories per family, data units. '
         'Bold family names were never seen by the three-shape model.', '',
         '| Family | ' + ' | '.join(LAB[n] for n in M) + ' | ' + ' | '.join(b for b, _ in bl) + ' |', '|---|' + '---:|' * (len(M) + len(bl))]
    for g in GROUPS:
        gl = f'**{g}**' if g in NEW else g
        T.append(f'| {gl} | ' + ' | '.join(fmt(n, g, k) for n in M) + ' | ' + ' | '.join(f"{out['baselines'][b][g]['mean']:.4f}" for _, b in bl) + ' |')
    T += ['', '| Family | change of mean vs baseline: medium | large | three-shape |', '|---|---:|---:|---:|']
    for g in GROUPS:
        T.append(f'| {g} | ' + ' | '.join(f"{out['relative_change_vs_baseline_of_mean'][n][g][k] * 100:+.1f}%" for n in ('medium', 'large', 'three_shape')) + ' |')
    (TAB / f'table_{k}.md').write_text('\n'.join(T) + '\n')
T = ['| Model | latent | GRU | ODE / decoder hidden | trainable parameters | trained on | best epoch | best val. −ELBO | epochs run | training time (min) |',
     '|---|---:|---:|---|---:|---|---:|---:|---:|---:|']
for n in M:
    i = info[n]
    T.append(f"| {LAB[n]} | {i['latent_dim']} | {i['encoder_hidden']} | {i['dynamics_hidden']} / {i['decoder_hidden']} | {i['n_parameters']:,} | {i['trained_on']} | "
             f"{i['best_epoch']} | {i['best_val_neg_elbo']:.2f} | {i['epochs_run']} | {i['training_minutes']:.1f} |")
T += ['', 'Validation −ELBO is comparable only between the three five-shape models (same validation data); the three-shape model was validated on three-shape data.']
(TAB / 'table_models.md').write_text('\n'.join(T) + '\n')
T = ['| Comparison | Family | Metric | mean diff. [95% bootstrap CI] | median diff. | % improved | rel. change of mean | Wilcoxon p (Holm) |', '|---|---|---|---|---:|---:|---:|---:|']
for key, v in out['paired'].items():
    comp, g, k = key.split('/')
    imp = next(x for kk, x in v.items() if kk.startswith('fraction_improved_by_'))
    T.append(f"| {comp.replace('_minus_', ' − ')} | {g} | {k.split('_')[0]} | {v['mean_difference']:+.4f} [{v['bootstrap_ci95_mean_difference'][0]:+.4f}, "
             f"{v['bootstrap_ci95_mean_difference'][1]:+.4f}] | {v['median_difference']:+.4f} | {imp * 100:.0f}% | {v['relative_change_of_mean'] * 100:+.1f}% | {v['wilcoxon_p_holm']:.2g} |")
T += ['', 'Difference = second model minus first; negative = the second model has lower error. "% improved" = share of the 150 paired '
      'trajectories on which the second model has lower RMSE. Holm correction within the 30 capacity-pair tests and, separately, the 20 tests against the three-shape model.']
(TAB / 'table_paired_statistics.md').write_text('\n'.join(T) + '\n')

# ------------------------------------------------------------------------------------------------ figures
FIVE = ['baseline', 'medium', 'large']
P = [info[n]['n_parameters'] for n in FIVE]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, k in zip(axes, ('forecast_rmse', 'recon_rmse')):
    for f in FAMS:
        y = [out['results'][n][f][k]['mean'] for n in FIVE]
        se = [out['results'][n][f][k]['sd'] / np.sqrt(150) for n in FIVE]
        ax.errorbar(P, y, yerr=se, marker='o', capsize=3, color=FCOL[f], label=f)
        if f in ORIG:
            ax.axhline(out['results']['three_shape'][f][k]['mean'], color=FCOL[f], ls=':', lw=1)
    ax.set_xscale('log'); ax.set_xticks(P); ax.set_xticklabels([f'{p:,}\n({n})' for p, n in zip(P, FIVE)])
    ax.set_xlabel('trainable parameters'); ax.set_ylabel(f"mean per-trajectory {'forecast' if k[0] == 'f' else 'reconstruction'} RMSE"); ax.grid(alpha=.2)
axes[0].set_title('Forecast (7 < t <= 10)'); axes[1].set_title('Reconstruction of withheld context points (t <= 7)')
axes[0].legend(frameon=False, fontsize=8, title='family (dotted: frozen three-shape model)', title_fontsize=8)
fig.suptitle('Figure 1. Model capacity vs test RMSE (one training seed; error bars = SE over 150 trajectories)', fontsize=11)
fig.tight_layout(); fig.savefig(FIG / 'fig1_capacity_vs_rmse.png', dpi=150); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8)); x = np.arange(len(FAMS)); w = .2
for ax, k in zip(axes, ('forecast_rmse', 'recon_rmse')):
    for j, n in enumerate(M):
        y = [out['results'][n][f][k]['mean'] for f in FAMS]; e = [out['results'][n][f][k]['sd'] / np.sqrt(150) for f in FAMS]
        ax.bar(x + (j - 1.5) * w, y, w, yerr=e, capsize=2, color=COL[n], label=LAB[n], hatch='//' if n == 'three_shape' else None, alpha=.9)
    ax.set_xticks(x); ax.set_xticklabels([f.replace('_', '-') + ('\n(unseen by\nthree-shape)' if f in NEW else '') for f in FAMS], fontsize=9)
    ax.grid(axis='y', alpha=.2); ax.set_ylabel('mean per-trajectory RMSE')
axes[0].set_title('Forecast (7 < t <= 10)'); axes[1].set_title('Reconstruction of withheld context points (t <= 7)')
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle('Figure 2. Per-family comparison on the same 750 test trajectories and observations (error bars = SE)', fontsize=11)
fig.tight_layout(); fig.savefig(FIG / 'fig2_family_comparison.png', dpi=150); plt.close(fig)

z = np.load(OUT / 'predictions_test.npz'); qt = z['query_t']; fut = qt > 7
fig, axes = plt.subplots(5, 2, figsize=(11, 25))
for r, f in enumerate(FAMS):
    for c, k in enumerate(np.flatnonzero(z['family'] == r)[:2]):
        ax = axes[r, c]
        ax.plot(*z['truth'][k][~fut].T, color='#888', lw=4, alpha=.45, label='ground truth, t <= 7')
        ax.plot(*z['truth'][k][fut].T, color='#888', lw=4, alpha=.45, ls=(0, (2, 1)), label='ground truth, t > 7')
        for n in FIVE:
            ax.plot(*z[f'pred_{n}'][k][~fut].T, color=COL[n], lw=1.3, label=f'{LAB[n]}: reconstruction')
            ax.plot(*z[f'pred_{n}'][k][fut].T, color=COL[n], lw=1.6, ls='--', label=f'{LAB[n]}: forecast')
        ax.scatter(*z['obs'][k].T, s=11, color='k', zorder=5, label='30 observations')
        ax.plot(*z['truth'][k][fut][0], 'o', mfc='none', mec='k', ms=10, label='forecast start (t = 7)')
        ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.15)
        ax.set_title(f"{f.replace('_', '-')}, test trajectory {c + 1} (first two in test order)", fontsize=10)
        if r == 0 and c == 0: ax.legend(fontsize=7, frameon=False, loc='lower left')
fig.suptitle('Figure 3. Example test predictions (solid = reconstruction t <= 7, dashed = forecast t > 7)', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, .985)); fig.savefig(FIG / 'fig3_examples.png', dpi=110); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
for n in M:
    a, b = out['results'][n]['original_three']['forecast_rmse'], out['results'][n]['new_two']['forecast_rmse']
    ax.errorbar(a['mean'], b['mean'], xerr=a['sd'] / np.sqrt(a['n']), yerr=b['sd'] / np.sqrt(b['n']), fmt='o', ms=9, color=COL[n], capsize=3, label=LAB[n])
ax.set_xlabel('mean forecast RMSE, circle + spiral + ellipse (450 trajectories)'); ax.set_ylabel('mean forecast RMSE, figure-eight + sinusoid (300)')
ax.set_yscale('log'); ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8); ax.set_title('Original vs new families (lower-left is better)')
ax = axes[1]
for j, n in enumerate(FIVE):
    d = [R[n]['forecast_rmse'][sel[f]].mean() / R['three_shape']['forecast_rmse'][sel[f]].mean() - 1 for f in ORIG]
    ax.bar(np.arange(3) + (j - 1) * .25, np.array(d) * 100, .25, color=COL[n], label=LAB[n])
ax.axhline(0, color='k', lw=.8); ax.set_xticks(range(3)); ax.set_xticklabels(ORIG)
ax.set_ylabel('change of mean forecast RMSE vs frozen three-shape model (%)'); ax.grid(axis='y', alpha=.2); ax.legend(frameon=False, fontsize=8)
ax.set_title('Cost on the original families of adding two families')
fig.suptitle('Figure 4. Performance trade-off (measured; one training seed per model)', fontsize=11)
fig.tight_layout(); fig.savefig(FIG / 'fig4_tradeoff.png', dpi=150); plt.close(fig)

for f in ('table_models.md', 'table_forecast_rmse.md', 'table_recon_rmse.md'):
    print((TAB / f).read_text())
print('checkpoints unchanged; figures in', FIG)
