#!/usr/bin/env python3
"""Phase 2: why are some forecasts poor?  (two-shape test split; evaluation only)

Per-trajectory analyses use the FIRST checkpoint (Model D, seed 42); sparsity / noise sweeps and
the cross-seed difficulty check use ALL checkpoints given (independent seeds).
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, torch
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from glo.evaluation import load_model, collect, geometry, horizon_error, summarize

p = argparse.ArgumentParser()
p.add_argument('--checkpoints', nargs='+', required=True)
p.add_argument('--out', default='results/validation/forecast_analysis')
p.add_argument('--n-obs-levels', type=int, nargs='+', default=[10, 15, 20, 30, 50])
p.add_argument('--noise-levels', type=float, nargs='+', default=[0.0, 0.025, 0.05, 0.1, 0.2])
p.add_argument('--threads', type=int, default=4)
a = p.parse_args()
torch.set_num_threads(a.threads)
root = Path(__file__).resolve().parents[1]; data_root = root / 'data/processed'
out = root / a.out; out.mkdir(parents=True, exist_ok=True)
models = [load_model(root / c, 'cpu') for c in a.checkpoints]
names = [Path(c).stem for c in a.checkpoints]
FAM = {'circle': 0, 'spiral': 1}
GREEN, BLUE, RED = '#2a9469', '#2578bc', '#e0533a'
report = {'checkpoints': a.checkpoints, 'split': 'test', 'primary_checkpoint': a.checkpoints[0]}

# ---------------------------------------------------------------- primary model, default protocol
m0, ck0 = models[0]
R = collect(m0, ck0, data_root, 'two_shapes')
qt = R['query_t']; cut = R['meta']['forecast_start']; fut = qt > cut
E = horizon_error(R)
G = geometry(R)

# 1. error vs prediction horizon
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
horizon = {}
for k, (f, fid) in enumerate(FAM.items()):
    e = E[R['family'] == fid]
    for q, ls, lab in ((50, '-', 'median'), (90, '--', '90th percentile')):
        ax[k].plot(qt, np.percentile(e, q, axis=0), ls, color=BLUE, label=lab)
    ax[k].plot(qt, e.mean(0), color='black', lw=1, label='mean')
    ax[k].axvline(cut, color='gray', ls=':'); ax[k].set_title(f'{f}: pointwise error vs time'); ax[k].set_xlabel('t')
    ax[k].grid(alpha=.2); ax[k].legend(frameon=False, fontsize=8)
    horizon[f] = {f'mean_err_at_t={t:.1f}': float(e[:, np.argmin(np.abs(qt - t))].mean()) for t in (3.5, 7.0, 8.0, 9.0, 10.0)}
ax[0].set_ylabel('Euclidean error (orig. units)')
fig.suptitle('Forecast error vs prediction horizon (Model D, seed 42; test split)')
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out / 'error_vs_horizon.png', dpi=140); plt.close(fig)
report['error_vs_horizon'] = horizon

# 2. geometry: radius drift, growth-rate error, angular-velocity error
geo = {}
for f, fid in FAM.items():
    s = R['family'] == fid
    d = {'angular_velocity_err_full_mean_abs': float(np.abs(G['omega_hat'][s] - G['omega'][s]).mean()),
         'angular_velocity_err_future_mean_abs': float(np.abs(G['omega_hat_future'][s] - G['omega'][s]).mean()),
         'angular_velocity_err_future_signed_mean': float((np.abs(G['omega_hat_future'][s]) - np.abs(G['omega'][s])).mean()),
         'radius_rel_err_context_mean': float(G['radius_rel_err_context'][s].mean()),
         'radius_rel_err_future_mean': float(G['radius_rel_err_future'][s].mean()),
         'radius_rel_drift_at_t10_mean_signed': float(G['radius_rel_drift_end'][s].mean()),
         'radius_rel_drift_at_t10_mean_abs': float(np.abs(G['radius_rel_drift_end'][s]).mean()),
         'fraction_radius_drift_inward_at_t10': float((G['radius_rel_drift_end'][s] < 0).mean())}
    if f == 'spiral':
        g, gh = G['growth'][s], G['growth_hat'][s]
        d.update({'growth_rate_abs_err_mean': float(np.abs(gh - g).mean()),
                  'growth_rate_err_signed_mean (|hat|-|true|)': float((np.abs(gh) - np.abs(g)).mean()),
                  'growth_sign_accuracy': float((np.sign(gh) == np.sign(g)).mean()),
                  'growth_rate_err_expanding': float(np.abs(gh - g)[g > 0].mean()),
                  'growth_rate_err_contracting': float(np.abs(gh - g)[g < 0].mean())})
    geo[f] = d
report['geometry'] = geo
fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
for f, fid in FAM.items():
    s = R['family'] == fid
    ax[0].scatter(np.abs(G['omega'][s]), np.abs(G['omega_hat_future'][s]) - np.abs(G['omega'][s]), s=9, label=f, alpha=.7)
    ax[1].hist(G['radius_rel_drift_end'][s], bins=30, alpha=.6, label=f)
ax[0].axhline(0, color='k', lw=.6); ax[0].set_xlabel('|ω| true'); ax[0].set_ylabel('|ω̂| − |ω| on t>7'); ax[0].set_title('Angular-velocity error (forecast window)')
ax[1].axvline(0, color='k', lw=.6); ax[1].set_xlabel('(r̂ − r)/r at t=10'); ax[1].set_title('Radius drift at end of forecast')
s = R['family'] == 1
ax[2].scatter(G['growth'][s], G['growth_hat'][s], s=10, color=RED, alpha=.7)
lim = [G['growth'][s].min() - .02, G['growth'][s].max() + .02]; ax[2].plot(lim, lim, 'k--', lw=.7)
ax[2].set_xlabel('true growth rate'); ax[2].set_ylabel('fitted growth rate of prediction'); ax[2].set_title('Spiral growth rate')
s = R['family'] == 0
ax[3].scatter(G['omega'][s], G['growth_hat'][s], s=10, color=GREEN, alpha=.7); ax[3].axhline(0, color='k', lw=.6)
ax[3].set_xlabel('ω'); ax[3].set_ylabel('fitted growth of predicted circle'); ax[3].set_title('Circles: spurious radial growth')
for x in ax[:2]: x.legend(frameon=False, fontsize=8)
for x in ax: x.grid(alpha=.2)
fig.tight_layout(); fig.savefig(out / 'geometry_errors.png', dpi=130); plt.close(fig)

# 3. which parameters / observation patterns go with high error (Spearman, per family)
ot = R['obs_t']
gaps = np.diff(ot, axis=1)
obs_feats = {'last_obs_time': ot[:, -1], 'largest_obs_gap': gaps.max(1),
             'n_obs_in_last_unit_before_cutoff': (ot > cut - 1).sum(1)}
assoc = {}
for f, fid in FAM.items():
    s = R['family'] == fid
    prm = R['params'][s]
    feats = {'radius_r0': prm[:, 0], 'abs_omega': np.abs(prm[:, 1]), 'omega_sign': np.sign(prm[:, 1]),
             'abs_centre_offset': np.linalg.norm(prm[:, 3:5], axis=1), 'phase': prm[:, 2]}
    if f == 'spiral':
        feats.update({'growth': prm[:, 5], 'abs_growth': np.abs(prm[:, 5]),
                      'final_radius r0*exp(10g)': prm[:, 0] * np.exp(10 * prm[:, 5])})
    feats.update({k: v[s] for k, v in obs_feats.items()})
    y = R['forecast_rmse'][s]
    rows = {}
    for k, v in feats.items():
        rho, pv = stats.spearmanr(v, y)
        rows[k] = {'spearman_rho': float(rho), 'p_value': float(pv)}
    assoc[f] = dict(sorted(rows.items(), key=lambda kv: -abs(kv[1]['spearman_rho'])))
report['forecast_error_associations (Spearman, forecast RMSE)'] = assoc
fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
for f, fid in FAM.items():
    s = R['family'] == fid; prm = R['params'][s]
    ax[0].scatter(prm[:, 0] * np.exp(10 * prm[:, 5]), R['forecast_rmse'][s], s=9, alpha=.7, label=f)
    ax[1].scatter(np.abs(prm[:, 1]), R['forecast_rmse'][s], s=9, alpha=.7, label=f)
s = R['family'] == 1
sc = ax[2].scatter(R['params'][s, 5], R['forecast_rmse'][s], c=R['params'][s, 0], s=12, cmap='viridis')
plt.colorbar(sc, ax=ax[2], label='r0')
ax[0].set_xlabel('radius at t=10'); ax[1].set_xlabel('|ω|'); ax[2].set_xlabel('spiral growth rate')
ax[2].set_title('Spirals (colour = r0)')
for x in ax: x.set_ylabel('forecast RMSE'); x.grid(alpha=.2)
ax[0].legend(frameon=False); ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(out / 'error_vs_parameters.png', dpi=130); plt.close(fig)

# 4. best / median / worst predictions (+ coordinates)
fig, ax = plt.subplots(4, 3, figsize=(15, 18))
for r0, (f, fid) in enumerate(FAM.items()):
    ids = np.flatnonzero(R['family'] == fid); order = ids[np.argsort(R['forecast_rmse'][ids])]
    for c, (k, tag) in enumerate(((order[0], 'best'), (order[len(order) // 2], 'median'), (order[-1], 'worst'))):
        A = ax[2 * r0, c]; tr, pr, ob = R['truth'][k], R['pred'][k], R['obs'][k]
        A.plot(tr[:, 0], tr[:, 1], color=GREEN, lw=2.6, label='clean truth')
        A.plot(pr[~fut, 0], pr[~fut, 1], color=BLUE, lw=1.6, label='reconstruction')
        A.plot(pr[fut, 0], pr[fut, 1], color=RED, lw=1.8, label='forecast')
        A.scatter(ob[:, 0], ob[:, 1], s=12, color='k', zorder=3, label='observations')
        A.set_aspect('equal', adjustable='datalim'); A.grid(alpha=.15)
        pm = R['params'][k]
        A.set_title(f'{f} {tag} (test #{k}) fc RMSE {R["forecast_rmse"][k]:.3f}\nr0={pm[0]:.2f} ω={pm[1]:.2f}'
                    + (f' g={pm[5]:.3f}' if fid == 1 else ''), fontsize=9)
        B = ax[2 * r0 + 1, c]
        for d, col in ((0, '#555'), (1, '#aaa')):
            B.plot(qt, tr[:, d], color=GREEN if d == 0 else '#7fc8a9', lw=2.2, label=f'{"xy"[d]} truth')
            B.plot(qt, pr[:, d], '--', color=BLUE if d == 0 else '#8fb9e0', lw=1.5, label=f'{"xy"[d]} pred')
        B.axvline(cut, color='gray', ls=':'); B.set_xlabel('t'); B.grid(alpha=.15)
        if c == 0: B.legend(frameon=False, fontsize=7, ncol=2)
ax[0, 0].legend(frameon=False, fontsize=7)
fig.suptitle('Best / median / worst forecasts by family (Model D, seed 42), with x(t), y(t) below each')
fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(out / 'best_median_worst.png', dpi=120); plt.close(fig)

# 5. cross-seed: are the same trajectories hard for every seed?
runs = [collect(m, ck, data_root, 'two_shapes') for m, ck in models]
if len(runs) > 1:
    M = np.stack([r['forecast_rmse'] for r in runs])
    rhos = [stats.spearmanr(M[i], M[j])[0] for i in range(len(runs)) for j in range(i + 1, len(runs))]
    worst_sets = [set(np.argsort(r['forecast_rmse'])[-30:]) for r in runs]
    overlap = [len(worst_sets[i] & worst_sets[j]) / 30 for i in range(len(runs)) for j in range(i + 1, len(runs))]
    report['cross_seed_difficulty'] = {
        'pairwise_spearman_of_per_trajectory_forecast_rmse': {'mean': float(np.mean(rhos)), 'min': float(np.min(rhos)), 'max': float(np.max(rhos))},
        'pairwise_overlap_of_worst_10pct_sets': {'mean': float(np.mean(overlap)), 'min': float(np.min(overlap))},
        'chance_overlap_of_worst_10pct': 0.1}

# 6. observation sparsity and measurement noise (all seeds)
def sweep(kind, levels):
    tab = {}
    for lv in levels:
        per = []
        for (m, ck) in models:
            kw = {'n_obs': lv} if kind == 'n_obs' else {'noise_std': lv}
            r = collect(m, ck, data_root, 'two_shapes', seed=777, **kw)
            per.append({f: summarize(r, r['family'] == fid) for f, fid in FAM.items()})
        tab[str(lv)] = {f: {k: {'mean_over_seeds': float(np.mean([q[f][k]['mean'] for q in per])),
                                'std_over_seeds': float(np.std([q[f][k]['mean'] for q in per], ddof=1)) if len(per) > 1 else 0.}
                            for k in ('recon_rmse', 'forecast_rmse', 'forecast_fde')} for f in FAM}
    return tab
report['sparsity_sweep (stored noise std 0.05)'] = sp = sweep('n_obs', a.n_obs_levels)
report['noise_sweep (n_obs 30, fresh noise, seed 777)'] = nz = sweep('noise', a.noise_levels)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
for A, tab, lv, xl in ((ax[0], sp, a.n_obs_levels, 'number of observations (t ≤ 7)'),
                       (ax[1], nz, a.noise_levels, 'observation noise std (training: 0.05)')):
    for f, col in (('circle', GREEN), ('spiral', RED)):
        for k, ls in (('recon_rmse', ':'), ('forecast_rmse', '-')):
            mu = [tab[str(v)][f][k]['mean_over_seeds'] for v in lv]; sd = [tab[str(v)][f][k]['std_over_seeds'] for v in lv]
            A.errorbar(lv, mu, yerr=sd, ls=ls, color=col, marker='o', ms=4, capsize=3, label=f'{f} {k.split("_")[0]}')
    A.set_xlabel(xl); A.set_ylabel('RMSE (orig. units)'); A.grid(alpha=.2); A.legend(frameon=False, fontsize=8)
ax[0].axvline(30, color='gray', ls=':'); ax[1].axvline(.05, color='gray', ls=':')
fig.suptitle(f'Sensitivity to sparsity and noise (mean ± std over {len(models)} seed(s); test split)')
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out / 'sparsity_and_noise.png', dpi=140); plt.close(fig)

(out / 'forecast_analysis.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in report if k not in ('checkpoints',)}, indent=1)[:6000])
