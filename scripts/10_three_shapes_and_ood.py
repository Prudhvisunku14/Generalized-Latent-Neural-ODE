#!/usr/bin/env python3
"""Phase 4 (three-shape shared model) and Phase 5 (out-of-distribution ellipses).

A. Three-shape model on the three-shape test split, per family (+ ellipse geometry).
B. Three-shape model vs two-shape models on the IDENTICAL circle/spiral test trajectories
   (two_shapes test split == circle/spiral part of three_shapes test split; verified).
C. OOD inference: two-shape models (never trained on ellipses) on the three-shape test ellipses,
   inputs normalised with the TWO-shape training statistics; compared with the three-shape model
   (in-distribution for ellipses) on the same ellipse observations. Evaluation only; no model
   was changed or selected using these results.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, torch
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from glo.evaluation import load_model, collect, by_family, summarize, t_ci95, baseline, geometry

p = argparse.ArgumentParser()
p.add_argument('--three', required=True, help='three-shape checkpoint')
p.add_argument('--two', nargs='+', required=True, help='two-shape checkpoints; FIRST = seed 42 (paired tests)')
p.add_argument('--out', default='results/validation/three_shapes')
a = p.parse_args()
torch.set_num_threads(4)
root = Path(__file__).resolve().parents[1]; data_root = root / 'data/processed'
out = root / a.out; out.mkdir(parents=True, exist_ok=True)
KEYS = ['recon_rmse', 'forecast_rmse', 'forecast_ade', 'forecast_fde', 'chamfer_full']
GREEN, BLUE, RED = '#2a9469', '#2578bc', '#e0533a'
m3, ck3 = load_model(root / a.three, 'cpu')
twos = [(pth,) + load_model(root / pth, 'cpu') for pth in a.two]
report = {'three_shape_checkpoint': a.three, 'three_shape_best_epoch': int(ck3['epoch']),
          'two_shape_checkpoints': a.two, 'protocol': 'test splits; n_obs=30 stored noisy observations; observation seed 123; posterior mean'}


def ellipse_geometry(res):
    """Shape of the full predicted trajectory about the true centre, compared with the truth's
    (computed identically): minor/major axis ratio, major-axis orientation, rotation direction."""
    c = res['params'][:, 3:5][:, None, :]
    fut = res['query_t'] > res['meta']['forecast_start']
    rows = []
    for P, T, om in zip(res['pred'] - c, res['truth'] - c, res['params'][:, 1]):
        def shape(X):
            w, v = np.linalg.eigh(np.cov(X.T)); return np.sqrt(w[0] / w[1]), np.arctan2(v[1, 1], v[0, 1])
        rp, ap = shape(P); rt, at = shape(T)
        dang = np.abs((ap - at + np.pi / 2) % np.pi - np.pi / 2)
        vel = np.gradient(P[fut], axis=0); cross = P[fut][:, 0] * vel[:, 1] - P[fut][:, 1] * vel[:, 0]
        rows.append((rp, rt, dang, np.sign(cross.mean()) == np.sign(om)))
    r = np.array(rows, dtype=float)
    return {'axis_ratio_true_mean': float(r[:, 1].mean()), 'axis_ratio_pred_mean': float(r[:, 0].mean()),
            'axis_ratio_abs_err_mean': float(np.abs(r[:, 0] - r[:, 1]).mean()),
            'axis_ratio_corr': float(np.corrcoef(r[:, 0], r[:, 1])[0, 1]),
            'major_axis_orientation_err_deg_median': float(np.degrees(np.median(r[:, 2]))),
            'forecast_rotation_direction_accuracy': float(r[:, 3].mean())}


# ---------------------------------------------------------------- A. three-shape model, in-distribution
R3 = collect(m3, ck3, data_root, 'three_shapes')
A = {'model': by_family(R3)}
for kind in ('const_pos', 'const_vel'):
    bm, bck = baseline(kind, 'three_shapes'); A[kind] = by_family(collect(bm, bck, data_root, 'three_shapes'))
g3 = geometry(R3)
A['rotation_direction_accuracy'] = {f: float((np.sign(g3['omega_hat']) == np.sign(g3['omega']))[R3['family'] == i].mean())
                                    for f, i in (('circle', 0), ('spiral', 1), ('ellipse', 2))}
A['spiral_growth_sign_accuracy'] = float((np.sign(g3['growth_hat']) == np.sign(g3['growth']))[R3['family'] == 1].mean())
A['ellipse_geometry'] = ellipse_geometry({**R3, **{k: R3[k][R3['family'] == 2] for k in ('pred', 'truth', 'params')}})
A['variance_ratio_pred_over_truth'] = {f: float(R3['pred'][R3['family'] == i].var(1).sum(-1).mean()
                                                / R3['truth'][R3['family'] == i].var(1).sum(-1).mean())
                                       for f, i in (('circle', 0), ('spiral', 1), ('ellipse', 2))}
report['A_three_shape_model_in_distribution'] = A

# ---------------------------------------------------------------- B. three-shape vs two-shape on identical circles/spirals
R3cs = collect(m3, ck3, data_root, 'two_shapes')            # same trajectories & observation draws as two-shape eval
R2 = [collect(m, ck, data_root, 'two_shapes') for _, m, ck in twos]
assert all(np.allclose(R3cs['obs'], r['obs'], atol=1e-5) for r in R2), 'observations differ'
B = {}
for f, fid in (('circle', 0), ('spiral', 1)):
    s = R3cs['family'] == fid
    B[f] = {}
    for k in KEYS:
        d = R3cs[k][s] - R2[0][k][s]
        two_vals = [float(r[k][s].mean()) for r in R2]
        B[f][k] = {'three_shape': float(R3cs[k][s].mean()),
                   'two_shape_seed42': float(R2[0][k][s].mean()),
                   'two_shape_across_seeds': t_ci95(two_vals) if len(two_vals) > 1 else {'values': two_vals},
                   'paired_diff_three_minus_two_seed42': float(d.mean()),
                   'paired_wilcoxon_p': float(stats.wilcoxon(R3cs[k][s], R2[0][k][s]).pvalue),
                   'relative_diff_pct_vs_seed42': float(100 * d.mean() / R2[0][k][s].mean())}
report['B_three_vs_two_shape_on_matching_circle_spiral'] = B

# ---------------------------------------------------------------- C. OOD ellipses
Rid = collect(m3, ck3, data_root, 'three_shapes', families=['ellipse'])
Rood = [collect(m, ck, data_root, 'three_shapes', families=['ellipse']) for _, m, ck in twos]
assert all(np.allclose(Rid['obs'], r['obs'], atol=1e-5) for r in Rood), 'ellipse observations differ'
C = {'two_shape_models_on_unseen_ellipses': {k: t_ci95([r[k].mean() for r in Rood]) if len(Rood) > 1
                                             else {'values': [float(Rood[0][k].mean())]} for k in KEYS},
     'three_shape_model_on_same_ellipses (in-distribution reference)': summarize(Rid),
     'two_shape_seed42_on_ellipses': summarize(Rood[0]),
     'ellipse_geometry_two_shape_seed42 (OOD)': ellipse_geometry(Rood[0]),
     'ellipse_geometry_three_shape (ID)': ellipse_geometry(Rid),
     'two_shape_seed42_in_distribution_reference': {f: {k: B[f][k]['two_shape_seed42'] for k in KEYS} for f in ('circle', 'spiral')}}
for kind in ('const_pos', 'const_vel'):
    bm, bck = baseline(kind, 'three_shapes'); C[f'{kind}_on_same_ellipses'] = summarize(collect(bm, bck, data_root, 'three_shapes', families=['ellipse']))
report['C_out_of_distribution_ellipses'] = C
(out / 'three_shapes_and_ood.json').write_text(json.dumps(report, indent=2))

# ---------------------------------------------------------------- figures
qt = R3['query_t']; fut = qt > R3['meta']['forecast_start']


def panel(ax, res, k, title, color=BLUE):
    tr, pr, ob = res['truth'][k], res['pred'][k], res['obs'][k]
    ax.plot(tr[:, 0], tr[:, 1], color=GREEN, lw=2.6, label='clean truth')
    ax.plot(pr[~fut, 0], pr[~fut, 1], color=color, lw=1.6, label='reconstruction')
    ax.plot(pr[fut, 0], pr[fut, 1], color=RED, lw=1.8, label='forecast')
    ax.scatter(ob[:, 0], ob[:, 1], s=11, color='k', zorder=3, label='observations')
    ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.15); ax.set_title(title, fontsize=9)


fig, ax = plt.subplots(3, 4, figsize=(18, 13.5))
for r_, (f, fid) in enumerate((('circle', 0), ('spiral', 1), ('ellipse', 2))):
    ids = np.flatnonzero(R3['family'] == fid)
    for c, k in enumerate(ids[:4]):
        panel(ax[r_, c], R3, k, f'{f} test #{k}: recon {R3["recon_rmse"][k]:.3f}, forecast {R3["forecast_rmse"][k]:.3f}')
ax[0, 0].legend(frameon=False, fontsize=7)
fig.suptitle('Three-shape shared Latent ODE (first 4 test trajectories of each family, not cherry-picked)')
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out / 'three_shape_gallery.png', dpi=120); plt.close(fig)

fig, ax = plt.subplots(2, 4, figsize=(18, 9))
order = np.argsort(Rid['forecast_rmse']); picks = [order[0], order[len(order) // 3], order[2 * len(order) // 3], order[-1]]
for c, k in enumerate(picks):
    panel(ax[0, c], Rood[0], k, f'OOD: two-shape model\nellipse #{k}: recon {Rood[0]["recon_rmse"][k]:.3f}, forecast {Rood[0]["forecast_rmse"][k]:.3f}', color='#8a5cc2')
    panel(ax[1, c], Rid, k, f'ID: three-shape model\nellipse #{k}: recon {Rid["recon_rmse"][k]:.3f}, forecast {Rid["forecast_rmse"][k]:.3f}')
    for rr in range(2):
        allp = np.concatenate([Rood[0]['pred'][k], Rid['pred'][k], Rid['truth'][k]]); lo, hi = allp.min(0) - .15, allp.max(0) + .15
        ax[rr, c].set_xlim(lo[0], hi[0]); ax[rr, c].set_ylim(lo[1], hi[1])
ax[0, 0].legend(frameon=False, fontsize=7)
fig.suptitle('Unseen ellipses: two-shape model (top, never trained on ellipses) vs three-shape model (bottom); '
             'same observations, columns = best / 33rd / 67th pct / worst for the three-shape model')
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out / 'ood_ellipses.png', dpi=120); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
for i, k in enumerate(('recon_rmse', 'forecast_rmse')):
    labels, vals, errs, cols = [], [], [], []
    for f in ('circle', 'spiral'):
        tw = B[f][k]['two_shape_across_seeds']
        labels += [f'{f}\ntwo-shape', f'{f}\nthree-shape']; vals += [np.mean(tw['values']), B[f][k]['three_shape']]
        errs += [tw.get('std', 0), 0]; cols += ['#9ab', BLUE]
    oo = C['two_shape_models_on_unseen_ellipses'][k]
    labels += ['ellipse\nOOD two-shape', 'ellipse\nthree-shape', 'ellipse\nconst. pos.']
    vals += [np.mean(oo['values']), C['three_shape_model_on_same_ellipses (in-distribution reference)'][k]['mean'],
             C['const_pos_on_same_ellipses'][k]['mean']]
    errs += [oo.get('std', 0), 0, 0]; cols += ['#8a5cc2', BLUE, '#ccc']
    ax[i].bar(range(len(vals)), vals, yerr=errs, color=cols, capsize=4)
    ax[i].set_xticks(range(len(vals))); ax[i].set_xticklabels(labels, fontsize=8); ax[i].set_title(f'test {k} (two-shape bars: mean ± SD over seeds)')
    ax[i].grid(alpha=.2, axis='y')
fig.tight_layout(); fig.savefig(out / 'three_vs_two_and_ood_bars.png', dpi=130); plt.close(fig)
print(json.dumps({'A': {k: A[k] for k in ('rotation_direction_accuracy', 'ellipse_geometry', 'variance_ratio_pred_over_truth')},
                  'A_metrics': {f: {k: round(A['model'][f][k]['mean'], 4) for k in KEYS} for f in A['model']},
                  'B': {f: {k: (round(B[f][k]['three_shape'], 4), round(B[f][k]['two_shape_seed42'], 4), f"{B[f][k]['paired_wilcoxon_p']:.2g}") for k in KEYS} for f in B},
                  'C_ood': {k: round(np.mean(C['two_shape_models_on_unseen_ellipses'][k]['values']), 4) for k in KEYS},
                  'C_id': {k: round(C['three_shape_model_on_same_ellipses (in-distribution reference)'][k]['mean'], 4) for k in KEYS}}, indent=1))
