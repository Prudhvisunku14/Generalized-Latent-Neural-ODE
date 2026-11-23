#!/usr/bin/env python3
"""Presentation figures for presentation_results/figures/, drawn ONLY from saved result files (no model is run, no original figure is touched).
Run from the project root: python audit/build_presentation_figures.py
"""
import sys
sys.path.insert(0, 'audit')
import numpy as np
from pathlib import Path
from results_common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT = Path('presentation_results/figures'); OUT.mkdir(parents=True, exist_ok=True)
AB, C5, ZS, SV, SS, GL, CAP = D['ab'], D['c5'], D['zs'], D['sv'], D['ss'], D['gl'], D['cap']
CELLS = 'ABCD'; CL = {'A': 'A: latent 8, GRU 64', 'B': 'B: latent 16, GRU 64', 'C': 'C: latent 8, GRU 128', 'D': 'D: latent 16, GRU 128'}
res = lambda m, g, k: AB['results'][m][g][k]['mean']
se = lambda m, g, k: AB['results'][m][g][k]['sd'] / np.sqrt(AB['results'][m][g][k]['n'])
G6 = FAMS + ['overall']; lab = lambda g: fname(g) if g != 'overall' else 'all five'


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches='tight'); plt.close(fig); print('saved', OUT / name)


def grouped_bars(ax, k, baseline_key, baseline_label, ymax=None):
    w = .2; x = np.arange(len(G6))
    for j, m in enumerate(CELLS):
        ax.bar(x + (j - 1.5) * w, [res(m, g, k) for g in G6], w * .92, yerr=[se(m, g, k) for g in G6], capsize=2, color=SER[j], label=CL[m], ecolor=INK2, error_kw={'lw': 1})
    ax.plot(x, [AB['baselines'][baseline_key][g]['mean'] for g in G6], ls='none', marker='_', ms=34, mew=2.5, color=BASE, label=baseline_label)
    ax.set_xticks(x); ax.set_xticklabels([lab(g) for g in G6]); style(ax)
    if ymax: ax.set_ylim(0, ymax)


# ---- Fig 1 and 2: reconstruction and forecast by family (controlled cells A-D, single seed)
fig, ax = new_fig(11.5, 5.2)
grouped_bars(ax, 'recon_rmse', 'linear_interpolation_recon', 'linear interpolation (simple baseline)')
ax.set_ylabel('reconstruction error (RMSE, lower is better)'); ax.set_title('Reconstruction of withheld points (t <= 7), by trajectory family')
ax.legend(frameon=False, fontsize=10, ncol=2, loc='upper left'); ax.set_ylim(0, .185)
fig.text(.5, -.02, 'Mean of per-trajectory RMSE over 150 test trajectories per family (750 overall); error bars = standard error. One training run per model (seed 42).', ha='center', fontsize=9.5, color=INK2)
save(fig, 'fig1_reconstruction_by_family.png')

fig, ax = new_fig(11.5, 5.2)
grouped_bars(ax, 'forecast_rmse', 'hold_last_forecast', 'hold-last (simple baseline)')
ax.set_ylabel('forecast error (RMSE, lower is better)'); ax.set_title('Forecast of the future (7 < t <= 10), by trajectory family')
ax.legend(frameon=False, fontsize=10, ncol=2, loc='upper left'); ax.set_ylim(0, 1.55)
fig.text(.5, -.02, 'Mean of per-trajectory RMSE; error bars = standard error. Constant-velocity baseline (2.0 to 2.6) is off the scale. One training run per model (seed 42).', ha='center', fontsize=9.5, color=INK2)
save(fig, 'fig2_forecast_by_family.png')

# ---- Fig 3: shared vs specialist (two-family data; individual training runs shown)
fig, axes = new_fig(11.5, 5.0, ncols=2)
for ax, k, ttl in ((axes[0], 'forecast_rmse', 'Forecast error (7 < t <= 10)'), (axes[1], 'recon_rmse', 'Reconstruction error (t <= 7)')):
    for gi, fam in enumerate(['circle', 'spiral']):
        for si, (who, col) in enumerate((('unified', SER[0]), ('specialist', SER[1]))):
            d = SV[fam]['seed_level'][k][who]; x = gi + (si - .5) * .36; vals = np.array(d['values'])
            ax.bar(x, d['mean'], .32, color=col, yerr=d['std'], capsize=4, ecolor=INK2, label=f"{'Shared model' if who == 'unified' else 'Specialist models'} ({d['n_runs']} training runs)" if gi == 0 else None)
            ax.scatter(x + np.linspace(-.07, .07, len(vals)), vals, s=22, color=INK, zorder=3, label='one dot = one training run' if (gi == 0 and si == 0) else None)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['circle', 'spiral']); ax.set_title(ttl); style(ax)
    ax.set_ylabel('RMSE (lower is better)')
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(.5, -.07))
fig.suptitle('Shared model vs specialist models (circle + spiral data; bars = mean of runs, whiskers = SD across runs)', fontsize=12.5, color=INK, y=1.02)
save(fig, 'fig3_shared_vs_specialist.png')

# ---- Fig 4: cross-family (dot plots, log axis). three-shape model seen/unseen; five-shape model (trained on all five) shown hollow
fig, axes = new_fig(12.5, 5.4, ncols=2, sharey=True)
rows = list(reversed(FAMS)); ypos = {g: i for i, g in enumerate(rows)}
for ax, k, ttl, extra in ((axes[0], 'recon_rmse', 'Reconstruction error (t <= 7)', [('linear interpolation (baseline)', 'const_pos', SER[2], 's')]),
                          (axes[1], 'forecast_rmse', 'Forecast error (7 < t <= 10)', [('hold-last (baseline)', 'const_pos', SER[1], 'D'), ('constant velocity (baseline)', 'const_vel', INK2, '^')])):
    ax.axhspan(-.5, 1.5, color='#f1efe9', zorder=0)
    n_ser = 2 + len(extra); offs = np.linspace(.24, -.24, n_ser)
    ax.scatter([ZS['three_shape_model'][g][k]['mean'] for g in rows], [ypos[g] + offs[0] for g in rows], s=80, color=SER[0], zorder=3, label='three-shape model (frozen)')
    ax.scatter([C5['five_shape_model'][g][k]['mean'] for g in rows], [ypos[g] + offs[1] for g in rows], s=80, facecolors='none', edgecolors=SER[0], lw=2, zorder=3, label='five-shape model (trained on all five)')
    for i, (lbl, key, col, mk_) in enumerate(extra):
        ax.scatter([ZS[key][g][k]['mean'] for g in rows], [ypos[g] + offs[2 + i] for g in rows], s=60, marker=mk_, color=col, zorder=2, label=lbl)
    ax.set_xscale('log'); ax.set_title(ttl); style(ax, 'x'); ax.set_xlabel('RMSE (log scale, lower is better)'); ax.set_ylim(-.6, len(rows) - .4)
axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels([lab(g) + ('  (unseen)' if g in FAMS[3:] else '') for g in rows])
axes[0].text(axes[0].get_xlim()[1] * .97, 1.42, 'unseen by the three-shape model', fontsize=9.5, color=INK2, style='italic', va='center', ha='right')
hs, ls = [], []
for ax in axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in ls: hs.append(h); ls.append(l)
fig.legend(hs, ls, loc='lower center', ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(.5, -.1))
fig.subplots_adjust(wspace=.05)
fig.suptitle('Unseen families (shaded): the three-shape model does not generalise; the five-shape model has seen all five (not zero-shot)', fontsize=12.5, color=INK, y=1.0)
save(fig, 'fig4_cross_family_generalisation.png')

# ---- Fig 5: controlled ablation, overall
fig, axes = new_fig(11.5, 5.0, ncols=2)
for ax, k, ttl in ((axes[0], 'recon_rmse', 'Reconstruction error (t <= 7)'), (axes[1], 'forecast_rmse', 'Forecast error (7 < t <= 10)')):
    for j, m in enumerate(CELLS):
        v = res(m, 'overall', k); ax.bar(j, v, .7, color=SER[j], yerr=se(m, 'overall', k), capsize=3, ecolor=INK2, label=CL[m])
        if m != 'A':
            ypos_ = v + se(m, 'overall', k) + (.004 if k == 'recon_rmse' else .02)
            if k == 'recon_rmse': ypos_ = max(ypos_, AB['baselines']['linear_interpolation_recon']['overall']['mean'] + .006) if abs(ypos_ - AB['baselines']['linear_interpolation_recon']['overall']['mean']) < .012 else ypos_
            ax.text(j, ypos_, pct(v / res('A', 'overall', k) - 1), ha='center', fontsize=11, color=INK)
    if k == 'recon_rmse': ax.axhline(AB['baselines']['linear_interpolation_recon']['overall']['mean'], color=BASE, ls='--', lw=1.4); ax.text(3.45, AB['baselines']['linear_interpolation_recon']['overall']['mean'] + .002, 'linear interpolation', ha='right', fontsize=10, color=INK2)
    ax.set_xticks(range(4)); ax.set_xticklabels(['A\nlatent 8\nGRU 64', 'B\nlatent 16\nGRU 64', 'C\nlatent 8\nGRU 128', 'D\nlatent 16\nGRU 128']); ax.set_title(ttl); style(ax); ax.set_ylabel('RMSE (lower is better)')
axes[0].set_ylim(0, .115); axes[1].set_ylim(0, .72)
fig.suptitle('Controlled 2 x 2 ablation, all 750 test trajectories (labels: change vs A). Single training run per model.', fontsize=12.5, color=INK, y=1.02)
save(fig, 'fig5_ablation_overall.png')

# ---- Fig 6: forecast error versus time
et = AB['error_vs_time']; qt = np.array(et['query_t']); fig, ax = new_fig(10.5, 5.2)
sel_ = qt >= 6.0
for j, m in enumerate(CELLS):
    y = np.array(et[m]['overall']); ax.plot(qt[sel_], y[sel_], color=SER[j], lw=2.4, label=CL[m]); ax.text(10.05, y[-1], m, color=INK, fontsize=12, va='center', fontweight='bold')
ax.axvline(7, color=BASE, ls=':', lw=1.4); ax.text(7.03, ax.get_ylim()[1] * .97, 'forecast starts (t = 7)', fontsize=10, color=INK2, va='top')
ax.set_xlabel('time t'); ax.set_ylabel('RMSE across the 750 test trajectories at each time'); ax.set_xlim(6, 10.3); style(ax, 'both')
ax.set_title('Error versus time: larger models are better just after t = 7, worse later\n(single training run per model; before t = 7 the curve includes the observed times)', fontsize=12)
ax.legend(frameon=False, fontsize=10, loc='lower right')
save(fig, 'fig6_forecast_error_vs_time.png')

# ---- Fig 7: size of effects versus run-to-run variation (different experiments, one axis: relative change of mean forecast RMSE)
fig, ax = new_fig(11.5, 5.4)
rc = lambda m: AB['results'][m]['overall']['forecast_rmse']['mean'] / AB['results']['A']['overall']['forecast_rmse']['mean'] - 1
rep = AB['reproducibility_pairs']; vals = np.array(SS['across_seeds']['overall']['forecast_rmse']['values']); rel = vals / vals.mean() - 1
rows7 = [('2x2 ablation:\nB vs A (latent only)', [rc('B')], SER[0]), ('2x2 ablation:\nC vs A (GRU only)', [rc('C')], SER[0]), ('2x2 ablation:\nD vs A (both)', [rc('D')], SER[0]),
         ('Same architecture, two training runs:\nA vs earlier 8/64 run', [rep['A_vs_orig_baseline']['overall']['forecast_rmse']['relative_difference']], SER[1]),
         ('Same architecture, two training runs:\nD vs earlier 16/128 run', [rep['D_vs_orig_medium']['overall']['forecast_rmse']['relative_difference']], SER[1]),
         ('Five training seeds, same architecture\n(two-shape data): each run vs the mean', list(rel), SER[2])]
for i, (name, v, col) in enumerate(rows7):
    y = len(rows7) - 1 - i; v = np.array(v) * 100
    if len(v) > 1: ax.plot([v.min(), v.max()], [y, y], color=col, lw=3, alpha=.5)
    ax.scatter(v, [y] * len(v), s=80, color=col, zorder=3)
    if len(v) == 1: ax.text(v[0] + 2, y, f'{v[0]:+.1f}%', va='center', fontsize=11, color=INK)
    else: ax.text(v.max() + 2, y, f'{v.min():+.0f}% to {v.max():+.0f}%', va='center', fontsize=11, color=INK)
ax.axvline(0, color=BASE, lw=1); ax.set_yticks(range(len(rows7))); ax.set_yticklabels([r[0] for r in reversed(rows7)], fontsize=10.5); ax.set_xlabel('difference in mean forecast RMSE (%)'); style(ax, 'x'); ax.set_xlim(-45, 75)
ax.set_title('How big are the architecture effects compared with run-to-run variation?\n(different experiments, shown on one axis for scale only)', fontsize=12)
save(fig, 'fig7_effects_vs_seed_variation.png')

# ---- Fig 8: GeoLife transfer (dot plot with 95% CI, log axis)
g30 = GL['interpolation']['30']; order = [('linear', 'Linear interpolation'), ('cubic', 'Cubic spline'), ('pchip', 'PCHIP'), ('frozen_three_shapes', 'Synthetic-trained model (frozen)'),
                                         ('geolife_scratch', 'GeoLife-trained from scratch'), ('geolife_finetuned_from_three_shapes', 'Synthetic then fine-tuned on GeoLife')]
fig, ax = new_fig(11.2, 4.6)
for i, (k, name) in enumerate(order):
    y = len(order) - 1 - i; m_ = g30[k]['median']; lo, hi = g30[k]['median_ci95_user_bootstrap']; col = SER[2] if i < 3 else (SER[1] if i == 3 else SER[0])
    ax.plot([lo, hi], [y, y], color=col, lw=3, alpha=.55); ax.scatter([m_], [y], s=90, color=col, zorder=3); ax.text(hi * 1.12, y, f'{m_:.1f} m', va='center', fontsize=11, color=INK)
ax.set_xscale('log'); ax.set_yticks(range(len(order))); ax.set_yticklabels([o[1] for o in reversed(order)], fontsize=10.5); style(ax, 'x'); ax.set_xlim(3, 400)
ax.set_xlabel('median interpolation error in metres (log scale, lower is better; bars = 95% CI)')
ax.set_title('Real GPS (GeoLife), 1,000 held-out segments from 97 users, 30 observations each', fontsize=12)
save(fig, 'fig8_geolife_transfer.png')
