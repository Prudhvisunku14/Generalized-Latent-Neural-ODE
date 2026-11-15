#!/usr/bin/env python3
"""Conditioning study: M1 (unified Model D) vs M2 (parameter-matched unconditioned) vs M3
(dynamics-conditioned, dz/dt = f(z, c)), plus ablation M2aug (unconditioned, latent 12).

Test split used ONCE for the final comparison; probes for the code analysis are FIT ON THE
VALIDATION split and scored on test. All models see identical observation subsets (seed 123).
"""
import argparse, csv, glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, torch
from scipy import stats
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from glo.evaluation import load_model, collect, geometry, horizon_error, t_ci95, encode, decode, METRICS

p = argparse.ArgumentParser()
p.add_argument('--ckpt-dir', default='checkpoints/conditioning')
p.add_argument('--out', default='results/conditioning')
p.add_argument('--specialist-json', default='results/validation/shared_vs_separate/shared_vs_separate.json')
a = p.parse_args()
torch.set_num_threads(4)
root = Path(__file__).resolve().parents[1]; data_root = root / 'data/processed'
out = root / a.out; out.mkdir(parents=True, exist_ok=True)
MODELS = {'M1': 'm1_unified', 'M2': 'm2_param_matched', 'M3': 'm3_conditioned', 'M2aug': 'm2aug_latent12'}
DESC = {'M1': 'unified Latent ODE (Model D)', 'M2': 'unconditioned, dynamics width 67 (param-matched to M3)',
        'M3': 'dynamics-conditioned, c in R^4', 'M2aug': 'ablation: unconditioned, latent 12 (= M3 state size)'}
COL = {'M1': '#888888', 'M2': '#e0a030', 'M3': '#2578bc', 'M2aug': '#8a5cc2'}
FAM = {'circle': 0, 'spiral': 1}
GREEN, RED = '#2a9469', '#e0533a'
EXTRA = ['radius_abs_drift_t10', 'radius_rel_err_future', 'omega_err_future', 'growth_abs_err']


def seed_of(path): return int(Path(path).stem.split('seed')[-1])


runs = {}
for m, stem in MODELS.items():
    for path in sorted(glob.glob(str(root / a.ckpt_dir / f'{stem}_seed*.pt')), key=seed_of):
        model, ck = load_model(path, 'cpu')
        r = collect(model, ck, data_root, 'two_shapes')
        g = geometry(r)
        r['radius_abs_drift_t10'] = np.abs(g['radius_rel_drift_end'])
        r['radius_rel_err_future'] = g['radius_rel_err_future']
        r['omega_err_future'] = np.abs(g['omega_hat_future'] - g['omega'])
        r['growth_abs_err'] = np.abs(g['growth_hat'] - g['growth'])
        r['growth_hat'] = g['growth_hat']
        hist = list(csv.DictReader(open(Path(path).with_name(Path(path).stem + '_history.csv'))))
        runs.setdefault(m, {})[seed_of(path)] = {'res': r, 'ck': ck, 'model': model, 'hist': hist, 'path': path}
        print(m, seed_of(path), 'epoch', ck['epoch'], 'fc', round(float(r['forecast_rmse'].mean()), 4), flush=True)
ref = runs['M1'][next(iter(runs['M1']))]['res']
for m in runs:
    for s in runs[m]:
        assert np.allclose(runs[m][s]['res']['obs'], ref['obs']), 'observation subsets differ'
fam = ref['family']; qt = ref['query_t']; cut = ref['meta']['forecast_start']; fut = qt > cut
nparam = {m: sum(p_.numel() for p_ in next(iter(runs[m].values()))['model'].parameters()) for m in runs}


def famval(r, k, f):
    sel = fam == FAM[f]
    if k == 'growth_abs_err' and f == 'circle': return float('nan')
    if k in ('radius_abs_drift_t10', 'radius_rel_err_future') and f == 'spiral' and k == 'radius_abs_drift_t10': pass
    return float(r[k][sel].mean())


KEYS = METRICS + EXTRA
table = {m: {f: {k: t_ci95([famval(runs[m][s]['res'], k, f) for s in runs[m]]) for k in KEYS
                 if not (k == 'growth_abs_err' and f == 'circle')} for f in FAM} for m in runs}

# paired comparisons: seed-paired (same seed = same training data) and trajectory-paired (seed-averaged)
paired = {}
for other in [x for x in ('M1', 'M2', 'M2aug') if x in runs]:
    seeds = sorted(set(runs['M3']) & set(runs[other]))
    paired[f'M3_vs_{other}'] = {'seeds': seeds}
    for f in FAM:
        sel = fam == FAM[f]; d = {}
        for k in KEYS:
            if k == 'growth_abs_err' and f == 'circle': continue
            per_seed = np.array([famval(runs['M3'][s]['res'], k, f) - famval(runs[other][s]['res'], k, f) for s in seeds])
            traj3 = np.mean([runs['M3'][s]['res'][k][sel] for s in seeds], 0)
            trajo = np.mean([runs[other][s]['res'][k][sel] for s in seeds], 0)
            dt = traj3 - trajo
            rng = np.random.default_rng(0)
            boot = np.array([dt[rng.integers(0, len(dt), len(dt))].mean() for _ in range(5000)])
            d[k] = {'M3_mean': float(traj3.mean()), f'{other}_mean': float(trajo.mean()),
                    'diff_M3_minus_other': float(dt.mean()), 'rel_diff_pct': float(100 * dt.mean() / trajo.mean()),
                    'seed_paired_diffs': per_seed.tolist(),
                    'seed_paired_t_p': float(stats.ttest_1samp(per_seed, 0).pvalue) if len(seeds) > 1 else None,
                    'seeds_M3_better': int((per_seed < 0).sum()),
                    'traj_paired_bootstrap95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    'traj_paired_wilcoxon_p': float(stats.wilcoxon(traj3, trajo).pvalue)}
        paired[f'M3_vs_{other}'][f] = d

# gap to specialists (validation study, 3 specialist seeds)
spec = json.loads((root / a.specialist_json).read_text())
gap = {}
for f in FAM:
    sp = spec[f]['seed_level']['forecast_rmse']['specialist']['mean']
    m1 = table['M1'][f]['forecast_rmse']['mean']
    gap[f] = {'specialist_forecast_rmse': sp, **{f'{m}_forecast_rmse': table[m][f]['forecast_rmse']['mean'] for m in runs},
              **{f'fraction_of_M1_gap_closed_by_{m}': (m1 - table[m][f]['forecast_rmse']['mean']) / (m1 - sp) for m in runs if m != 'M1'}}

# ------------------------------------------------------------------ dynamics-code analysis (M3)
code = {}
probe_rows = []
if 'M3' in runs:
    def targets(params, family):
        om, g = params[:, 1], params[:, 5]
        return {'omega (signed)': ('reg', om, np.ones(len(om), bool)), '|omega|': ('reg', np.abs(om), np.ones(len(om), bool)),
                'radius r0': ('reg', params[:, 0], np.ones(len(om), bool)),
                'spiral growth rate': ('reg', g, family == 1),
                'rotation direction': ('clf', (om > 0).astype(int), np.ones(len(om), bool)),
                'family (circle vs spiral)': ('clf', family.astype(int), np.ones(len(om), bool)),
                'spiral expanding vs contracting': ('clf', (g > 0).astype(int), family == 1)}
    reps_by_seed = {}
    for s, R in runs['M3'].items():
        ev = encode(R['model'], R['ck'], data_root, 'two_shapes', split='val', seed=321)
        et = encode(R['model'], R['ck'], data_root, 'two_shapes', split='test')
        reps = {'c (M3)': (ev['cond'].numpy(), et['cond'].numpy()), 'z0 mean (M3)': (ev['mu'].numpy(), et['mu'].numpy()),
                '[z0, c] (M3)': (np.hstack([ev['mu'], ev['cond']]), np.hstack([et['mu'], et['cond']]))}
        if s in runs['M1']:
            e1v = encode(runs['M1'][s]['model'], runs['M1'][s]['ck'], data_root, 'two_shapes', split='val', seed=321)
            e1t = encode(runs['M1'][s]['model'], runs['M1'][s]['ck'], data_root, 'two_shapes', split='test')
            reps['z0 mean (M1)'] = (e1v['mu'].numpy(), e1t['mu'].numpy())
        reps_by_seed[s] = (reps, ev, et)
        tv, tt = targets(ev['params'], ev['family']), targets(et['params'], et['family'])
        for rep, (Xv, Xt) in reps.items():
            for name, (kind, yv, mv) in tv.items():
                _, yt, mt = tt[name]
                if kind == 'reg':
                    pr = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(Xv[mv], yv[mv])
                    score = float(pr.score(Xt[mt], yt[mt]))
                else:
                    pr = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, max_iter=2000)).fit(Xv[mv], yv[mv])
                    score = float(pr.score(Xt[mt], yt[mt]))
                probe_rows.append({'seed': s, 'representation': rep, 'target': name,
                                   'metric': 'R2' if kind == 'reg' else 'accuracy', 'test_score': score})
    probes = {}
    for row in probe_rows:
        probes.setdefault(row['target'], {}).setdefault(row['representation'], []).append(row['test_score'])
    code['linear_probes (fit on val, scored on test; mean [min,max] over seeds)'] = {
        t: {r: {'mean': float(np.mean(v)), 'min': float(np.min(v)), 'max': float(np.max(v))} for r, v in d.items()} for t, d in probes.items()}

    # code usage / ablation and sensitivity (all M3 seeds)
    usage = {}
    for s, R in runs['M3'].items():
        reps, ev, et = reps_by_seed[s]
        mu, c = et['mu'], et['cond']
        truth = R['res']['truth']; base = R['res']['pred']
        def frmse(pred, sel=None):
            e = np.sqrt(((pred - truth)[:, fut] ** 2).mean(axis=(1, 2)))
            return {f: float(e[fam == i].mean()) for f, i in FAM.items()}
        cmean = c.mean(0, keepdim=True).expand_as(c)
        rng = np.random.default_rng(s)
        perm_within = torch.arange(len(c))
        for i in FAM.values():
            idx = np.flatnonzero(fam == i); perm_within[idx] = torch.from_numpy(rng.permutation(idx))
        perm_all = torch.from_numpy(rng.permutation(len(c)))
        usage[s] = {'posterior_c (baseline)': frmse(base),
                    'c replaced by mean code': frmse(decode(R['model'], mu, cmean, et['query_t'], et['mean'], et['scale'])),
                    'c shuffled within family': frmse(decode(R['model'], mu, c[perm_within], et['query_t'], et['mean'], et['scale'])),
                    'c shuffled across all': frmse(decode(R['model'], mu, c[perm_all], et['query_t'], et['mean'], et['scale']))}
    code['code_usage: forecast RMSE when c is altered (z0 kept)'] = {
        cond_: {f: {'mean': float(np.mean([usage[s][cond_][f] for s in usage])), 'per_seed': [usage[s][cond_][f] for s in usage]}
                for f in FAM} for cond_ in next(iter(usage.values()))}

    # sensitivity along the val-fitted growth-rate probe direction (seed 42 or first)
    s0 = 42 if 42 in runs['M3'] else next(iter(runs['M3']))
    R = runs['M3'][s0]; reps, ev, et = reps_by_seed[s0]
    sp = ev['family'] == 1
    sc = StandardScaler().fit(ev['cond'].numpy()[sp])
    rg = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sc.transform(ev['cond'].numpy()[sp]), ev['params'][sp, 5])
    direction = torch.tensor(rg.coef_ / np.linalg.norm(rg.coef_) * sc.scale_, dtype=torch.float32)
    direction = direction / direction.norm()
    cstd = float(et['cond'].std(0).mean())
    sens = {}
    from glo.evaluation import geometry as geo_fn
    for f, i in FAM.items():
        idx = np.flatnonzero(fam == i)[:60]
        rows = []
        for k in (-2, -1, 0, 1, 2):
            cc = (et['cond'][idx] + k * cstd * direction).clamp(-1, 1)
            pr = decode(R['model'], et['mu'][idx], cc, et['query_t'], et['mean'], et['scale'])
            gg = geo_fn({'pred': pr, 'query_t': qt, 'params': et['params'][idx], 'meta': et['meta']})
            rows.append({'shift_in_code_std': k, 'fitted_growth_mean': float(gg['growth_hat'].mean()),
                         'abs_omega_hat_mean': float(np.abs(gg['omega_hat']).mean()),
                         'mean_radius_ratio_t10': float(np.mean(1 + gg['radius_rel_drift_end']))})
        sens[f] = rows
    code[f'sensitivity: shift c along val-fitted spiral-growth probe direction (seed {s0}, 60 test trajectories/family)'] = sens

    # code swap: circle z0 + codes from expanding / contracting spirals
    ci = np.flatnonzero(fam == 0)[:60]; sp_t = np.flatnonzero(fam == 1)
    exp_c = et['cond'][sp_t[et['params'][sp_t, 5] > 0]].mean(0, keepdim=True)
    con_c = et['cond'][sp_t[et['params'][sp_t, 5] < 0]].mean(0, keepdim=True)
    swap = {}
    for lab, cc in (('own code', et['cond'][ci]), ('mean code of expanding spirals', exp_c.expand(len(ci), -1)),
                    ('mean code of contracting spirals', con_c.expand(len(ci), -1))):
        pr = decode(R['model'], et['mu'][ci], cc, et['query_t'], et['mean'], et['scale'])
        gg = geo_fn({'pred': pr, 'query_t': qt, 'params': et['params'][ci], 'meta': et['meta']})
        swap[lab] = {'fitted_growth_mean': float(gg['growth_hat'].mean()), 'fraction_growth_positive': float((gg['growth_hat'] > 0).mean())}
    code[f'code_swap: circle z0 with spiral codes (seed {s0}, 60 test circles)'] = swap

report = {'models': {m: {'description': DESC[m], 'n_parameters': nparam[m], 'seeds': sorted(runs[m]),
                         'best_epochs': [int(runs[m][s]['ck']['epoch']) for s in sorted(runs[m])]} for m in runs},
          'protocol': 'two_shapes test split (300); n_obs=30 stored noisy observations; observation seed 123 for every model; '
                      'posterior mean z0; seed-level mean/SD/95% t-CI; seed-paired t-test and trajectory-paired '
                      '(seed-averaged) bootstrap/Wilcoxon', 'seed_level': table, 'paired': paired,
          'gap_to_specialists': gap, 'dynamics_code': code}
(out / 'conditioning_study.json').write_text(json.dumps(report, indent=2))
with open(out / 'probe_scores.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['seed', 'representation', 'target', 'metric', 'test_score']); w.writeheader(); w.writerows(probe_rows)

# markdown table
L = ['| Model | Params | Family | Recon RMSE | Forecast RMSE | ADE | FDE | Chamfer | |r̂−r|/r @t=10 | ω err (t>7) | growth err |', '|' + '---|' * 11]
for m in runs:
    for f in FAM:
        t = table[m][f]
        cell = lambda k: f"{t[k]['mean']:.4f} ± {t[k]['std']:.4f}" if k in t else '–'
        L.append(f'| {m} | {nparam[m]} | {f} | ' + ' | '.join(cell(k) for k in ['recon_rmse', 'forecast_rmse', 'forecast_ade', 'forecast_fde', 'chamfer_full', 'radius_abs_drift_t10', 'omega_err_future', 'growth_abs_err']) + ' |')
(out / 'comparison_table.md').write_text('\n'.join(L) + '\n')

# ------------------------------------------------------------------ figures
# 1. architecture diagram
fig, ax = plt.subplots(figsize=(15, 4.6)); ax.set_xlim(0, 15); ax.set_ylim(0, 4.6); ax.axis('off')
def box(x, y, w, h, text, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06', fc=col, ec='#333', lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=9)
def arrow(x0, y0, x1, y1): ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', lw=1.4))
box(0.1, 1.7, 1.9, 1.2, '30 noisy, irregular\nobservations\n(x, y, t/T, Δt/T)\nt ≤ 7', '#f2f2f2')
box(2.6, 1.7, 2.0, 1.2, 'GRU encoder\n(hidden 64)\nreads newest → oldest', '#dde8f5')
box(5.3, 3.0, 2.0, 1.0, 'μ, log σ²  ∈ R⁸\nz₀ = μ + σ ⊙ ε', '#dde8f5')
box(5.3, 0.6, 2.0, 1.0, 'c = tanh(W h) ∈ R⁴\n(deterministic,\nconstant in time)', '#fde3c8')
box(8.0, 1.7, 2.6, 1.2, 'Conditioned ODE\ndz/dt = f_θ(z, c)\nMLP 12→64→64→8 (tanh)\nRK4, step 0.1, t∈[0,10]', '#d9f0e3')
box(11.3, 1.7, 1.6, 1.2, 'Decoder g_ψ(z)\nMLP 8→64→64→2', '#dde8f5')
box(13.4, 1.7, 1.5, 1.2, 'x̂(t), t∈[0,10]\nrecon (≤7)\nforecast (>7)', '#f2f2f2')
arrow(2.0, 2.3, 2.6, 2.3); arrow(4.6, 2.6, 5.3, 3.4); arrow(4.6, 2.0, 5.3, 1.1); arrow(7.3, 3.4, 8.0, 2.6)
arrow(7.3, 1.1, 8.0, 2.0); arrow(10.6, 2.3, 11.3, 2.3); arrow(12.9, 2.3, 13.4, 2.3)
ax.text(7.5, 4.45, 'M3: dynamics-conditioned Latent ODE. Loss: −E_q[log N(x_obs | x̂, 0.05²)] + KL(q(z₀) ‖ N(0, I)); c has no KL term',
        ha='center', fontsize=10)
ax.text(7.5, 0.1, 'M1/M2 are identical without the c branch (f_θ(z), MLP 8→64→64→8 for M1, 8→67→67→8 for M2). '
        'No family labels or future coordinates are inputs.', ha='center', fontsize=8.5, color='#444')
fig.savefig(out / 'fig1_architecture_m3.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# 2. loss curves (median over seeds, min-max band)
fig, ax = plt.subplots(1, 4, figsize=(21, 4.4))
for m in runs:
    H = [runs[m][s]['hist'] for s in runs[m]]; n = min(len(h) for h in H); ep = np.arange(1, n + 1)
    for i, k in enumerate(('train_loss', 'val_elbo_loss', 'val_kl', 'val_forecast_rmse')):
        v = np.array([[float(r[k]) for r in h[:n]] for h in H])
        ax[i].plot(ep, np.median(v, 0), color=COL[m], label=f'{m}'); ax[i].fill_between(ep, v.min(0), v.max(0), color=COL[m], alpha=.15)
for i, t in enumerate(('train −ELBO (sampled z₀)', 'validation −ELBO (z₀ = μ; selection criterion)', 'validation KL (nats)', 'validation forecast RMSE')):
    ax[i].set_title(t); ax[i].set_xlabel('epoch'); ax[i].grid(alpha=.2); ax[i].legend(frameon=False, fontsize=8)
ax[0].set_yscale('symlog'); ax[1].set_yscale('symlog'); ax[3].set_yscale('log')
fig.suptitle('Training curves: median over seeds, band = min–max'); fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out / 'fig2_loss_curves.png', dpi=130); plt.close(fig)

# 3/4. M3 reconstruction galleries + 9 best/median/worst
if 'M3' in runs:
    s0 = 42 if 42 in runs['M3'] else next(iter(runs['M3'])); R3 = runs['M3'][s0]['res']
    def panel(A, R, k, title, col='#2578bc'):
        A.plot(R['truth'][k][:, 0], R['truth'][k][:, 1], color=GREEN, lw=2.6, label='clean truth')
        A.plot(R['pred'][k][~fut, 0], R['pred'][k][~fut, 1], color=col, lw=1.6, label='reconstruction (t ≤ 7)')
        A.plot(R['pred'][k][fut, 0], R['pred'][k][fut, 1], color=RED, lw=1.8, label='forecast (t > 7)')
        A.scatter(R['obs'][k][:, 0], R['obs'][k][:, 1], s=11, color='k', zorder=3, label='noisy observations')
        A.set_aspect('equal', adjustable='datalim'); A.grid(alpha=.15); A.set_title(title, fontsize=9)
    for f, num in (('circle', 3), ('spiral', 4)):
        ids = np.flatnonzero(fam == FAM[f])[:6]
        fig, ax = plt.subplots(2, 3, figsize=(14, 9.4))
        for A, k in zip(ax.flat, ids):
            panel(A, R3, k, f'{f} test #{k}: recon {R3["recon_rmse"][k]:.3f}, forecast {R3["forecast_rmse"][k]:.3f}')
        ax.flat[0].legend(frameon=False, fontsize=7)
        fig.suptitle(f'M3 (seed {s0}): first 6 {f}s of the test split (not cherry-picked)')
        fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out / f'fig{num}_{f}_reconstruction_M3.png', dpi=130); plt.close(fig)
    fig, ax = plt.subplots(2, 3, figsize=(14, 9.4))
    for r_, f in enumerate(FAM):
        ids = np.flatnonzero(fam == FAM[f]); o = ids[np.argsort(R3['forecast_rmse'][ids])]
        for c_, (k, tag) in enumerate(((o[0], 'best'), (o[len(o) // 2], 'median'), (o[-1], 'worst'))):
            pm = ref['params'][k]
            panel(ax[r_, c_], R3, k, f'{f} {tag} #{k}: forecast {R3["forecast_rmse"][k]:.3f}\nr0={pm[0]:.2f} ω={pm[1]:.2f}' + (f' g={pm[5]:.3f}' if f == 'spiral' else ''))
    ax[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(f'M3 (seed {s0}): best / median / worst forecasts'); fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / 'fig9_best_median_worst_M3.png', dpi=130); plt.close(fig)

# 5. M1 vs M2 vs M3 on identical trajectories (seed 42), same axes
s0 = 42
ids = list(np.flatnonzero(fam == 0)[:2]) + list(np.flatnonzero(fam == 1)[:2])
shown = [m for m in ('M1', 'M2', 'M3') if m in runs and s0 in runs[m]]
fig, ax = plt.subplots(len(ids), len(shown), figsize=(4.6 * len(shown), 4.4 * len(ids)), squeeze=False)
for r_, k in enumerate(ids):
    allp = np.concatenate([runs[m][s0]['res']['pred'][k] for m in shown] + [ref['truth'][k]]); lo, hi = allp.min(0) - .15, allp.max(0) + .15
    for c_, m in enumerate(shown):
        R = runs[m][s0]['res']; A = ax[r_, c_]
        A.plot(R['truth'][k][:, 0], R['truth'][k][:, 1], color=GREEN, lw=2.6, label='truth')
        A.plot(R['pred'][k][~fut, 0], R['pred'][k][~fut, 1], color=COL[m], lw=1.6, label='reconstruction')
        A.plot(R['pred'][k][fut, 0], R['pred'][k][fut, 1], color=RED, lw=1.8, label='forecast')
        A.scatter(R['obs'][k][:, 0], R['obs'][k][:, 1], s=10, color='k', zorder=3)
        A.set_xlim(lo[0], hi[0]); A.set_ylim(lo[1], hi[1]); A.set_aspect('equal'); A.grid(alpha=.15)
        A.set_title(f'{m} | {"circle" if fam[k] == 0 else "spiral"} #{k}\nrecon {R["recon_rmse"][k]:.3f}, forecast {R["forecast_rmse"][k]:.3f}', fontsize=9)
ax[0, 0].legend(frameon=False, fontsize=7)
fig.suptitle(f'M1 vs M2 vs M3 (seed {s0}): identical test trajectories, observations and axis limits')
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out / 'fig5_model_comparison.png', dpi=120); plt.close(fig)

# 6. forecast error vs horizon (mean over seeds)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
hz = {}
for i, f in enumerate(FAM):
    for m in runs:
        E = np.mean([horizon_error(runs[m][s]['res'])[fam == FAM[f]].mean(0) for s in runs[m]], 0)
        sd = np.std([horizon_error(runs[m][s]['res'])[fam == FAM[f]].mean(0) for s in runs[m]], 0, ddof=1)
        ax[i].plot(qt, E, color=COL[m], label=m); ax[i].fill_between(qt, E - sd, E + sd, color=COL[m], alpha=.15)
        hz.setdefault(f, {})[m] = {f't={t}': float(E[np.argmin(np.abs(qt - t))]) for t in (7.0, 8.0, 9.0, 10.0)}
    ax[i].axvline(cut, color='gray', ls=':'); ax[i].set_title(f'{f}: mean pointwise error (± SD over seeds)'); ax[i].set_xlabel('t'); ax[i].grid(alpha=.2); ax[i].legend(frameon=False)
fig.tight_layout(); fig.savefig(out / 'fig6_error_vs_horizon.png', dpi=130); plt.close(fig)
report['error_vs_horizon'] = hz

# 7. radius / angular-velocity / growth errors (per-seed means)
fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
for i, (k, f, t) in enumerate((('radius_abs_drift_t10', 'circle', 'circle |radius drift| at t=10'), ('omega_err_future', 'circle', 'circle |ω̂−ω| (t>7)'),
                               ('omega_err_future', 'spiral', 'spiral |ω̂−ω| (t>7)'), ('growth_abs_err', 'spiral', 'spiral |ĝ−g|'))):
    for j, m in enumerate(runs):
        v = [famval(runs[m][s]['res'], k, f) for s in runs[m]]
        ax[i].scatter(np.full(len(v), j) + np.linspace(-.12, .12, len(v)), v, color=COL[m], s=26, zorder=3)
        ax[i].errorbar(j + .3, np.mean(v), yerr=np.std(v, ddof=1) if len(v) > 1 else 0, fmt='s', color='k', capsize=3, ms=4)
    ax[i].set_xticks(range(len(runs))); ax[i].set_xticklabels(list(runs)); ax[i].set_title(t); ax[i].grid(alpha=.2)
fig.suptitle('Geometric forecast errors (dots = seeds; black = mean ± SD)'); fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(out / 'fig7_radius_angular_growth_errors.png', dpi=130); plt.close(fig)

# 8. dynamics-code visualisation (seed 42)
if 'M3' in runs:
    reps, ev, et = reps_by_seed[s0]
    C = et['cond'].numpy(); P2 = PCA(2).fit(ev['cond'].numpy()); Z = P2.transform(C)
    prm = et['params']; fm = et['family']
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    ax[0].scatter(Z[fm == 0, 0], Z[fm == 0, 1], s=10, color=GREEN, label='circle'); ax[0].scatter(Z[fm == 1, 0], Z[fm == 1, 1], s=10, color=RED, label='spiral')
    ax[0].legend(frameon=False); ax[0].set_title('c (PCA fit on val), by family')
    sc_ = ax[1].scatter(Z[:, 0], Z[:, 1], c=prm[:, 1], cmap='coolwarm', s=10); plt.colorbar(sc_, ax=ax[1], label='ω'); ax[1].set_title('c, coloured by signed ω')
    s_ = fm == 1
    sc_ = ax[2].scatter(Z[s_, 0], Z[s_, 1], c=prm[s_, 5], cmap='PiYG', s=12); plt.colorbar(sc_, ax=ax[2], label='growth g'); ax[2].set_title('spirals only: c coloured by growth rate')
    for k in (-2, -1, 0, 1, 2):
        pass
    rows = report['dynamics_code'][next(x for x in report['dynamics_code'] if x.startswith('sensitivity'))]
    for f, col in (('circle', GREEN), ('spiral', RED)):
        ax[3].plot([r['shift_in_code_std'] for r in rows[f]], [r['fitted_growth_mean'] for r in rows[f]], marker='o', color=col, label=f)
    ax[3].axhline(0, color='k', lw=.6); ax[3].set_xlabel('shift of c along growth-probe direction (code SDs)'); ax[3].set_ylabel('mean fitted growth of prediction')
    ax[3].set_title('Sensitivity: does moving c change radial growth?'); ax[3].legend(frameon=False)
    for A in ax: A.grid(alpha=.2)
    fig.suptitle(f'M3 dynamics code c ∈ R⁴ on test trajectories (seed {s0}); PC variance explained {', '.join(f'{v:.2f}' for v in P2.explained_variance_ratio_)}')
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out / 'fig8_dynamics_code.png', dpi=130); plt.close(fig)

(out / 'conditioning_study.json').write_text(json.dumps(report, indent=2))
print((out / 'comparison_table.md').read_text())
print(json.dumps({'gap': gap, 'paired_forecast': {k: {f: {kk: v[f]['forecast_rmse'][kk] for kk in ('diff_M3_minus_other', 'rel_diff_pct', 'seed_paired_t_p', 'seeds_M3_better', 'traj_paired_bootstrap95', 'traj_paired_wilcoxon_p')} for f in FAM} for k, v in paired.items()}}, indent=1))
