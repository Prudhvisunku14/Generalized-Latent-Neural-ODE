#!/usr/bin/env python3
"""Controlled 2 x 2 ablation (latent dimension x GRU width): paired test-set evaluation, factorial effects, figures.

  python scripts/33_evaluate_ablation_2x2.py            # writes results/ablation_2x2/   (ABL_OUT overrides, e.g. for dry runs)

Cells (configs/ablation_2x2.yaml): A latent 8/GRU 64, B 16/64, C 8/128, D 16/128, all trained with the same protocol and --data-seed 42.
Earlier single runs (data_seed unset) are evaluated only as references: orig_baseline ~ A, orig_medium ~ D, orig_large (latent 32 / GRU 128).
Test protocol is identical to scripts/31 and 32: five_shapes test split (750 trajectories), 30 stored noisy observations at t <= 7 drawn
with observation seed 123, identical across models (asserted), posterior-mean z0, each model normalised with its own training constants.
Metrics per trajectory (data units): recon_rmse (unobserved context points, t <= 7) and forecast_rmse (all points with t > 7).
Tables report mean / median / SD of per-trajectory RMSE; pooled RMSE is stored separately and never mixed in.
Statistics: all intervals/tests resample or pair TEST TRAJECTORIES. They quantify evaluation-sample variability only. Every cell is a single
training run, so no interval here reflects training-seed variability, and no causal claim should rest on a single seed.
No test information is used for any selection; checkpoints were selected during training by validation -ELBO only.
"""
import json, math, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, torch, yaml
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from glo.evaluation import load_model, collect, baseline
from glo.inference import sha256_file

torch.set_num_threads(4)
root = Path(__file__).resolve().parents[1]; data_root = root / 'data/processed'
OUT = Path(os.environ.get('ABL_OUT', root / 'results/ablation_2x2')); TAB = OUT / 'tables'; FIG = OUT / 'figures'
for d in (TAB, FIG): d.mkdir(parents=True, exist_ok=True)
ABL = yaml.safe_load((root / 'configs/ablation_2x2.yaml').read_text())
FAMS = ['circle', 'spiral', 'ellipse', 'figure_eight', 'sinusoid']; KEYS = ['recon_rmse', 'forecast_rmse']
CELLS = ['A', 'B', 'C', 'D']; REFS = ['orig_baseline', 'orig_medium', 'orig_large']
CK = {**{c: root / ABL['variants'][c]['checkpoint'] for c in CELLS}, **{r: root / ABL['references'][r] for r in REFS}}
LAB = {'A': 'A  latent 8 / GRU 64', 'B': 'B  latent 16 / GRU 64', 'C': 'C  latent 8 / GRU 128', 'D': 'D  latent 16 / GRU 128',
       'orig_baseline': 'earlier baseline (8/64)', 'orig_medium': 'earlier medium (16/128)', 'orig_large': 'earlier large (32/128)'}
COL = {'A': '#2578bc', 'B': '#e0a030', 'C': '#3a9d5d', 'D': '#8a5cc2', 'orig_baseline': '#2578bc', 'orig_medium': '#8a5cc2', 'orig_large': '#777'}
FCOL = dict(zip(FAMS, ['#2578bc', '#e0a030', '#3a9d5d', '#d1495b', '#8a5cc2']))


def same_obs(a, b):
    return (np.array_equal(a['obs_t'], b['obs_t']) and np.array_equal(a['obs_mask'], b['obs_mask'])
            and np.allclose(a['obs'], b['obs'], atol=1e-5) and np.array_equal(a['ids'], b['ids']))


def pooled(res, sel):
    d2 = ((res['pred'] - res['truth']) ** 2).sum(-1)[sel]; fut = res['query_t'] > 7; miss = (~res['obs_mask'][sel]) & ~fut[None]
    return {'recon_rmse': float(np.sqrt(d2[miss].sum() / (2 * miss.sum()))), 'forecast_rmse': float(np.sqrt(d2[:, fut].mean() / 2))}


def summary(res, sel):
    return {k: {'mean': float(res[k][sel].mean()), 'median': float(np.median(res[k][sel])), 'sd': float(res[k][sel].std(ddof=1)),
                'n': int(sel.sum())} for k in KEYS}


def boot_ci(d, B=10000, seed=0):
    m = d[np.random.default_rng(seed).integers(0, len(d), (B, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.
    for r, i in enumerate(order):
        run = max(run, min(1., (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def wilcoxon_p(d):
    return float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0


# ------------------------------------------------------------------------------------------------ evaluate
before = {k: sha256_file(p) for k, p in CK.items()}
R, info, hist = {}, {}, {}
n_train = int(np.load(data_root / 'five_shapes/train.npz')['family'].shape[0]); steps_per_epoch = math.ceil(n_train / 32)
for name, path in CK.items():
    m, ck = load_model(path, 'cpu'); R[name] = collect(m, ck, data_root, 'five_shapes')
    h = np.genfromtxt(path.with_name(path.stem + '_history.csv'), delimiter=',', names=True); hist[name] = h
    tr = ck['config']['training']; i_sel = int(np.argmin(h['val_elbo_loss'])); i_fc = int(np.argmin(h['val_forecast_rmse']))
    assert int(h['epoch'][i_sel]) == int(ck['epoch']), 'history and checkpoint disagree on the selected epoch'
    info[name] = {'checkpoint': str(path.relative_to(root)), 'latent_dim': ck['model']['latent_dim'], 'encoder_hidden': ck['model']['encoder_hidden'],
                  'dynamics_hidden': ck['model']['dynamics_hidden'], 'decoder_hidden': ck['model']['decoder_hidden'],
                  'n_parameters': int(sum(p.numel() for p in m.parameters() if p.requires_grad)), 'seed': tr['seed'], 'data_seed': tr['data_seed'],
                  'batch_size': tr['batch_size'], 'lr': tr['lr'], 'epochs_run': int(h['epoch'].max()),
                  'optimizer_steps': steps_per_epoch * int(h['epoch'].max()), 'training_minutes': float(h['seconds'].sum() / 60),
                  'selection_criterion': 'validation -ELBO (beta = 1, 30 observed noisy points + KL); forecast RMSE is logged, not used',
                  'selected_epoch': int(ck['epoch']), 'selected_val_neg_elbo': float(ck['val_loss']),
                  'selected_val_forecast_rmse_pooled': float(h['val_forecast_rmse'][i_sel]), 'selected_val_context_rmse_pooled': float(h['val_context_rmse'][i_sel]),
                  'selected_train_nll': float(h['train_nll'][i_sel]), 'selected_val_nll': float(h['val_nll'][i_sel]),
                  'selected_val_kl': float(h['val_kl'][i_sel]),
                  'diagnostic_min_val_forecast_epoch': int(h['epoch'][i_fc]), 'diagnostic_min_val_forecast_rmse_pooled': float(h['val_forecast_rmse'][i_fc]),
                  'sha256': before[name]}
for n in R:
    assert same_obs(R['A'], R[n]), f'{n} saw different observations'; assert np.isfinite(R[n]['pred']).all(), f'{n} non-finite predictions'
BL = {k: collect(*baseline(k, 'five_shapes'), data_root, 'five_shapes') for k in ('const_pos', 'const_vel')}
assert all(same_obs(R['A'], b) for b in BL.values())
fam = R['A']['family']; names = {v: k for k, v in R['A']['meta']['family_to_id'].items()}
sel = {f: fam == i for i, f in names.items()}; sel['overall'] = np.ones_like(fam, bool); GROUPS = FAMS + ['overall']
qt = R['A']['query_t']; fut = qt > 7

out = {'protocol': __doc__.strip(), 'models': info, 'observations_identical_across_models': True, 'predictions_finite': True,
       'optimizer_steps_per_epoch': steps_per_epoch,
       'results': {n: {g: summary(R[n], sel[g]) for g in GROUPS} for n in R},
       'pooled_rmse': {n: {g: pooled(R[n], sel[g]) for g in GROUPS} for n in R},
       'baselines': {'linear_interpolation_recon': {g: summary(BL['const_pos'], sel[g])['recon_rmse'] for g in GROUPS},
                     'hold_last_forecast': {g: summary(BL['const_pos'], sel[g])['forecast_rmse'] for g in GROUPS},
                     'const_velocity_forecast': {g: summary(BL['const_vel'], sel[g])['forecast_rmse'] for g in GROUPS}}}
out['relative_change_vs_A_of_mean'] = {n: {g: {k: float(R[n][k][sel[g]].mean() / R['A'][k][sel[g]].mean() - 1) for k in KEYS} for g in GROUPS} for n in R if n != 'A'}
out['fraction_of_trajectories_better_than_A'] = {n: {g: {k: float((R[n][k][sel[g]] < R['A'][k][sel[g]]).mean()) for k in KEYS} for g in GROUPS} for n in R if n != 'A'}

# paired cell contrasts (second minus first); Holm over all of them
PAIRS = [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'D'), ('C', 'D'), ('B', 'C')]
tests, out['paired'] = [], {}
for a, b in PAIRS:
    for g in GROUPS:
        for k in KEYS:
            x, y = R[a][k][sel[g]], R[b][k][sel[g]]; d = y - x; key = f'{b}_minus_{a}/{g}/{k}'
            out['paired'][key] = {'n': int(len(d)), 'mean_difference': float(d.mean()), 'median_difference': float(np.median(d)),
                                  'bootstrap_ci95_mean_difference': boot_ci(d), 'fraction_improved_by_' + b: float((y < x).mean()),
                                  'relative_change_of_mean': float(y.mean() / x.mean() - 1), 'wilcoxon_p': wilcoxon_p(d)}
            tests.append((key, out['paired'][key]['wilcoxon_p']))
for (key, _), adj in zip(tests, holm(np.array([p for _, p in tests]))):
    out['paired'][key]['wilcoxon_p_holm'] = float(adj)
out['paired_holm_family'] = f'all {len(tests)} cell-contrast tests'

# factorial effects on per-trajectory RMSE (positive = higher error)
def contrasts(k, s):
    A, B, C, D = (R[c][k][s] for c in CELLS)
    return {'latent_main_effect (16 vs 8, averaged over GRU)': .5 * ((B - A) + (D - C)),
            'gru_main_effect (128 vs 64, averaged over latent)': .5 * ((C - A) + (D - B)),
            'interaction ((D-C) - (B-A))': (D - C) - (B - A)}
out['factorial'] = {}; ftests = []
for g in GROUPS:
    for k in KEYS:
        for cname, d in contrasts(k, sel[g]).items():
            key = f'{g}/{k}/{cname}'
            out['factorial'][key] = {'n': int(len(d)), 'mean': float(d.mean()), 'bootstrap_ci95': boot_ci(d), 'wilcoxon_p': wilcoxon_p(d)}
            ftests.append((key, out['factorial'][key]['wilcoxon_p']))
for (key, _), adj in zip(ftests, holm(np.array([p for _, p in ftests]))):
    out['factorial'][key]['wilcoxon_p_holm'] = float(adj)
out['factorial_holm_family'] = f'all {len(ftests)} factorial tests'

# reproducibility indicator: same architecture, different training data stream (one pair each, not a variance estimate)
out['reproducibility_pairs'] = {}
for new, old in (('A', 'orig_baseline'), ('D', 'orig_medium')):
    out['reproducibility_pairs'][f'{new}_vs_{old}'] = {g: {k: {'mean_new': float(R[new][k][sel[g]].mean()), 'mean_old': float(R[old][k][sel[g]].mean()),
                                                              'relative_difference': float(R[new][k][sel[g]].mean() / R[old][k][sel[g]].mean() - 1)} for k in KEYS} for g in GROUPS}
# error versus time (RMSE across the 750 trajectories at each query time, per-coordinate convention)
out['error_vs_time'] = {'query_t': qt.tolist(), **{n: {g: np.sqrt((((R[n]['pred'] - R[n]['truth']) ** 2).sum(-1)[sel[g]]).mean(0) / 2).tolist() for g in ('overall',) + tuple(FAMS)} for n in R}}
after = {k: sha256_file(p) for k, p in CK.items()}
assert before == after, 'a checkpoint changed during evaluation'
out['checkpoints_unchanged'] = True
(OUT / 'ablation_2x2_results.json').write_text(json.dumps(out, indent=2))
np.savez_compressed(OUT / 'predictions_test.npz', ids=R['A']['ids'], family=fam, query_t=qt, truth=R['A']['truth'], obs=R['A']['obs'], obs_t=R['A']['obs_t'],
                    **{f'pred_{n}': R[n]['pred'] for n in R})

# ------------------------------------------------------------------------------------------------ tables
def fmt(n, g, k):
    s = out['results'][n][g][k]; return f"{s['mean']:.4f} ({s['median']:.4f}; {s['sd']:.4f})"
M = CELLS + REFS
for k, title, bl in (('recon_rmse', 'Reconstruction RMSE of withheld context points (t <= 7)', [('linear interp.', 'linear_interpolation_recon')]),
                     ('forecast_rmse', 'Forecast RMSE (7 < t <= 10)', [('hold last', 'hold_last_forecast'), ('const. velocity', 'const_velocity_forecast')])):
    T = [f'**{title}.** Mean (median; SD) of per-trajectory RMSE, 150 test trajectories per family, data units. Cells A-D: controlled protocol; '
         'the last three model columns are earlier single runs kept for reference.', '',
         '| Family | ' + ' | '.join(LAB[n] for n in M) + ' | ' + ' | '.join(b for b, _ in bl) + ' |', '|---|' + '---:|' * (len(M) + len(bl))]
    for g in GROUPS:
        T.append(f'| {g} | ' + ' | '.join(fmt(n, g, k) for n in M) + ' | ' + ' | '.join(f"{out['baselines'][b][g]['mean']:.4f}" for _, b in bl) + ' |')
    T += ['', f'Change of the mean relative to cell A, and share of the paired trajectories on which the model beats A ({k}):', '',
          '| Family | ' + ' | '.join(f'{n}: change / better on' for n in 'BCD') + ' |', '|---|---:|---:|---:|']
    for g in GROUPS:
        T.append(f'| {g} | ' + ' | '.join(f"{out['relative_change_vs_A_of_mean'][n][g][k] * 100:+.1f}% / {out['fraction_of_trajectories_better_than_A'][n][g][k] * 100:.0f}%" for n in 'BCD') + ' |')
    (TAB / f'table_{k}.md').write_text('\n'.join(T) + '\n')
T = ['| Model | latent | GRU | parameters | seed / data seed | optimiser steps | selected epoch | val. -ELBO | val. forecast RMSE (pooled) | val. context RMSE (pooled) | train NLL | val NLL | min val. forecast RMSE (epoch) | training min |',
     '|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
for n in M:
    i = info[n]
    T.append(f"| {LAB[n]} | {i['latent_dim']} | {i['encoder_hidden']} | {i['n_parameters']:,} | {i['seed']} / {i['data_seed']} | {i['optimizer_steps']:,} | {i['selected_epoch']} | "
             f"{i['selected_val_neg_elbo']:.2f} | {i['selected_val_forecast_rmse_pooled']:.4f} | {i['selected_val_context_rmse_pooled']:.4f} | {i['selected_train_nll']:.2f} | "
             f"{i['selected_val_nll']:.2f} | {i['diagnostic_min_val_forecast_rmse_pooled']:.4f} ({i['diagnostic_min_val_forecast_epoch']}) | {i['training_minutes']:.1f} |")
T += ['', 'Selection criterion for every row: validation -ELBO only. The "min val. forecast RMSE" column is a post-hoc diagnostic on the validation split; '
      'no checkpoint was selected with it and intermediate checkpoints were not saved, so it cannot be evaluated on the test set. '
      'Train NLL is measured with a sampled z0 and stored noisy targets; val NLL uses the posterior mean, so the two are not strictly like-for-like.']
(TAB / 'table_models.md').write_text('\n'.join(T) + '\n')
T = ['| Effect on per-trajectory RMSE | Group | Metric | mean [95% bootstrap CI] | Wilcoxon p (Holm) |', '|---|---|---|---|---:|']
for key, v in out['factorial'].items():
    g, k, c = key.split('/')
    T.append(f"| {c} | {g} | {k.split('_')[0]} | {v['mean']:+.4f} [{v['bootstrap_ci95'][0]:+.4f}, {v['bootstrap_ci95'][1]:+.4f}] | {v['wilcoxon_p_holm']:.2g} |")
T += ['', 'Positive = higher error. Intervals and tests resample test trajectories only (single training run per cell); they do not capture training-seed variability.']
(TAB / 'table_factorial_effects.md').write_text('\n'.join(T) + '\n')
T = ['| Comparison | Group | Metric | mean diff. [95% bootstrap CI] | median diff. | % improved | rel. change of mean | Wilcoxon p (Holm) |', '|---|---|---|---|---:|---:|---:|---:|']
for key, v in out['paired'].items():
    comp, g, k = key.split('/'); imp = next(x for kk, x in v.items() if kk.startswith('fraction_improved_by_'))
    T.append(f"| {comp.replace('_minus_', ' - ')} | {g} | {k.split('_')[0]} | {v['mean_difference']:+.4f} [{v['bootstrap_ci95_mean_difference'][0]:+.4f}, {v['bootstrap_ci95_mean_difference'][1]:+.4f}] | "
             f"{v['median_difference']:+.4f} | {imp * 100:.0f}% | {v['relative_change_of_mean'] * 100:+.1f}% | {v['wilcoxon_p_holm']:.2g} |")
(TAB / 'table_paired_statistics.md').write_text('\n'.join(T) + '\n')

# ------------------------------------------------------------------------------------------------ figures
def se(n, g, k):
    s = out['results'][n][g][k]; return s['sd'] / np.sqrt(s['n'])
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8)); x = np.arange(len(GROUPS)); w = .2
for ax, k in zip(axes, ('forecast_rmse', 'recon_rmse')):
    for j, n in enumerate(CELLS):
        ax.bar(x + (j - 1.5) * w, [out['results'][n][g][k]['mean'] for g in GROUPS], w, yerr=[se(n, g, k) for g in GROUPS], capsize=2, color=COL[n], label=LAB[n])
    if k == 'forecast_rmse':
        ax.plot(x, [out['baselines']['hold_last_forecast'][g]['mean'] for g in GROUPS], 'k_', ms=22, mew=2, label='hold-last baseline')
    else:
        ax.plot(x, [out['baselines']['linear_interpolation_recon'][g]['mean'] for g in GROUPS], 'k_', ms=22, mew=2, label='linear interpolation')
    ax.set_xticks(x); ax.set_xticklabels([g.replace('_', '-') for g in GROUPS], fontsize=9); ax.grid(axis='y', alpha=.2); ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel('mean per-trajectory RMSE')
axes[0].set_title('Forecast (7 < t <= 10)'); axes[1].set_title('Reconstruction of withheld context points (t <= 7)')
fig.suptitle('Figure 1. The four controlled cells on the same 750 test trajectories (one training run per cell; error bars = SE over test trajectories)', fontsize=11)
fig.tight_layout(); fig.savefig(FIG / 'fig1_cells_by_family.png', dpi=150); plt.close(fig)

fig, axes = plt.subplots(2, 6, figsize=(19, 7))
for r, k in enumerate(('forecast_rmse', 'recon_rmse')):
    for c, g in enumerate(GROUPS):
        ax = axes[r, c]
        for lat, ls, mk in ((8, '-', 'o'), (16, '--', 's')):
            cells = {'A': (64, 8), 'C': (128, 8), 'B': (64, 16), 'D': (128, 16)}
            pts = [(gru, out['results'][n][g][k]['mean'], se(n, g, k)) for n, (gru, l) in cells.items() if l == lat]
            pts.sort(); ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], ls=ls, marker=mk, capsize=3, label=f'latent {lat}', color='#333')
        ax.set_xticks([64, 128]); ax.set_xlabel('GRU width'); ax.grid(alpha=.2); ax.set_title(f"{g.replace('_', '-')}: {'forecast' if r == 0 else 'recon.'}", fontsize=10)
        if c == 0: ax.set_ylabel('mean per-trajectory RMSE'); ax.legend(frameon=False, fontsize=8)
fig.suptitle('Figure 2. 2 x 2 interaction plots (parallel lines = no interaction; one training run per cell)', fontsize=11)
fig.tight_layout(); fig.savefig(FIG / 'fig2_interaction.png', dpi=140); plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
for ax, g in zip(axes.ravel(), ('overall',) + tuple(FAMS)):
    for n in CELLS:
        ax.plot(qt, out['error_vs_time'][n][g], color=COL[n], label=LAB[n], lw=1.6)
    ax.axvline(7, color='k', ls=':', lw=1); ax.set_title(g.replace('_', '-')); ax.grid(alpha=.2); ax.set_xlabel('time t'); ax.set_ylabel('RMSE across test trajectories')
    ax.set_yscale('log')
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle('Figure 3. Error versus time (dotted line: t = 7; before it the curve includes the 30 observed timestamps, after it every point is an extrapolation)', fontsize=11)
fig.tight_layout(); fig.savefig(FIG / 'fig3_error_vs_time.png', dpi=140); plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
for ax, (col, ttl, logy) in zip(axes, (('val_nll', 'validation NLL (posterior mean)', False), ('train_nll', 'training NLL (sampled z0)', False),
                                       ('val_context_rmse', 'validation context RMSE (pooled)', True), ('val_forecast_rmse', 'validation forecast RMSE (pooled)', False))):
    for n in CELLS:
        ax.plot(hist[n]['epoch'], hist[n][col], color=COL[n], label=LAB[n], lw=1.3)
        ax.plot(info[n]['selected_epoch'], hist[n][col][info[n]['selected_epoch'] - 1], 'o', color=COL[n], ms=6)
    ax.set_title(ttl, fontsize=10); ax.set_xlabel('epoch'); ax.grid(alpha=.2)
    if logy: ax.set_yscale('log')
    if col in ('val_nll', 'train_nll'): ax.set_yscale('symlog', linthresh=100)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle('Figure 4. Training histories (dots: epoch selected by validation -ELBO). Validation split only; no test data.', fontsize=11)
fig.tight_layout(); fig.savefig(FIG / 'fig4_training_histories.png', dpi=140); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.6)); x = np.arange(len(GROUPS)); w = .26
for ax, k in zip(axes, ('forecast_rmse', 'recon_rmse')):
    for j, n in enumerate('BCD'):
        ax.bar(x + (j - 1) * w, [out['fraction_of_trajectories_better_than_A'][n][g][k] * 100 for g in GROUPS], w, color=COL[n], label=f'{LAB[n]} better than A')
    ax.axhline(50, color='k', lw=.8, ls=':'); ax.set_xticks(x); ax.set_xticklabels([g.replace('_', '-') for g in GROUPS], fontsize=9); ax.set_ylim(0, 100)
    ax.set_ylabel('% of paired test trajectories'); ax.grid(axis='y', alpha=.2); ax.legend(frameon=False, fontsize=8)
axes[0].set_title('Forecast RMSE lower than cell A'); axes[1].set_title('Reconstruction RMSE lower than cell A')
fig.suptitle('Figure 5. Share of the same test trajectories on which each cell beats cell A', fontsize=11)
fig.tight_layout(); fig.savefig(FIG / 'fig5_percent_improved.png', dpi=150); plt.close(fig)

z = np.load(OUT / 'predictions_test.npz'); fut = z['query_t'] > 7
fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))
for ax, (r, f) in zip(axes, enumerate(FAMS)):
    k = np.flatnonzero(z['family'] == r)[0]
    ax.plot(*z['truth'][k][~fut].T, color='#888', lw=4, alpha=.45, label='truth t <= 7'); ax.plot(*z['truth'][k][fut].T, color='#888', lw=4, alpha=.45, ls=(0, (2, 1)), label='truth t > 7')
    for n in CELLS:
        ax.plot(*z[f'pred_{n}'][k][~fut].T, color=COL[n], lw=1.1); ax.plot(*z[f'pred_{n}'][k][fut].T, color=COL[n], lw=1.6, ls='--', label=LAB[n])
    ax.scatter(*z['obs'][k].T, s=10, color='k', zorder=5, label='30 observations'); ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.15)
    ax.set_title(f"{f.replace('_', '-')} (first test trajectory of the family)", fontsize=9)
axes[0].legend(fontsize=6.5, frameon=False, loc='lower left')
fig.suptitle('Figure 6. Example predictions (solid = reconstruction, t <= 7; dashed = forecast, t > 7). Illustration only, not selected for being typical.', fontsize=11)
fig.tight_layout(); fig.savefig(FIG / 'fig6_examples.png', dpi=120); plt.close(fig)

for f in ('table_models.md', 'table_forecast_rmse.md', 'table_recon_rmse.md', 'table_factorial_effects.md'):
    print((TAB / f).read_text())
print('checkpoints unchanged; outputs in', OUT)
