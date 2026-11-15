#!/usr/bin/env python3
"""Complete held-out test evaluation of an improved checkpoint vs the original one.

* Both models (and the baselines) see IDENTICAL observation subsets (same generator seed).
* Metrics in original coordinate units, per family: recon RMSE (unobserved context points),
  forecast RMSE / MAE / ADE / FDE (t > cutoff), Chamfer (full trajectory).
* Baselines: constant-position and constant-velocity forecasts from the last observations;
  linear interpolation of the observations for the reconstruction interval.
* Posterior mean vs sampled posterior; prediction-vs-truth variance; geometric checks using
  the known generating parameters (radius, rotation direction, angular velocity, spiral growth).
"""
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from glo.model import LatentODE
from glo.pipeline import (TrajectoryDataset, create_loader, make_query_indices, prepare_batch,
                          infer_test_metrics, seed_everything)

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--original', default='checkpoints/two_shapes_cond0_best.pt')
p.add_argument('--history', default='')
p.add_argument('--out', default='results/improved_two_shapes')
p.add_argument('--n-obs', type=int, default=30)
p.add_argument('--n-query', type=int, default=200)
p.add_argument('--seed', type=int, default=123)
p.add_argument('--n-samples', type=int, default=10)
p.add_argument('--split', default='test', choices=['val', 'test'])
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = root / a.out; out.mkdir(parents=True, exist_ok=True)
seed_everything(a.seed)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CIRCLE, SPIRAL = 0, 1


def load(path):
    path = Path(path); path = path if path.is_absolute() else root / path
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = LatentODE(**ck['model']).to(dev); m.load_state_dict(ck['state_dict']); m.eval()
    if m.solver == 'dopri5':           # evaluate with a tighter tolerance than training
        m.rtol, m.atol = 1e-5, 1e-6
    return m, ck


new_model, new_ck = load(a.checkpoint)
old_model, old_ck = load(a.original)
data = TrajectoryDataset(root / 'data/processed', new_ck['experiment'], a.split)
meta = data.meta
params = torch.tensor(data.arrays['params'])
qidx = make_query_indices(meta['n_points'], a.n_query)
time = torch.tensor(data.arrays['time'], dtype=torch.float32)
gen = torch.Generator().manual_seed(a.seed)
cut = meta['forecast_start']
batches = [prepare_batch(b, time, meta, qidx, a.n_obs, dev, generator=gen)
           for b in create_loader(data, batch_size=50, shuffle=False)]


def to_orig(pred_norm, P):
    return pred_norm * P['scale'] + P['mean']


def baselines(P):
    """Constant position / constant velocity on the query grid, using observations only."""
    obs = to_orig(P['obs_xy'], P); ot = P['obs_t']; qt = P['query_t']
    B, Q = obs.shape[0], len(qt)
    interp = torch.empty(B, Q, 2, device=obs.device)
    for i in range(B):                                       # linear interp. inside context
        for c in range(2):
            interp[i, :, c] = torch.from_numpy(np.interp(qt.cpu().numpy(), ot[i].cpu().numpy(),
                                                        obs[i, :, c].cpu().numpy())).to(obs.device)
    fut = qt > cut
    last, last_t = obs[:, -1], ot[:, -1]
    vel = (obs[:, -1] - obs[:, -2]) / (ot[:, -1] - ot[:, -2]).clamp_min(1e-3)[:, None]
    cp, cv = interp.clone(), interp.clone()
    cp[:, fut] = last[:, None, :]
    cv[:, fut] = last[:, None, :] + vel[:, None, :] * (qt[fut][None, :, None] - last_t[:, None, None])
    return cp, cv


def geometry(pred, P, prm):
    """Per-trajectory geometric diagnostics relative to the KNOWN generating centre."""
    qt = P['query_t']; c = prm[:, 3:5].to(pred.device)
    R0, omega, growth = prm[:, 0].to(pred.device), prm[:, 1].to(pred.device), prm[:, 5].to(pred.device)
    rel = pred - c[:, None]; r = rel.norm(dim=-1)
    ang = torch.from_numpy(np.unwrap(torch.atan2(rel[..., 1], rel[..., 0]).cpu().numpy(), axis=1)).to(pred.device)
    T = qt - qt.mean()

    def slope(y):
        return (y * T).sum(-1) / T.square().sum()

    true_r = R0[:, None] * torch.exp(growth[:, None] * qt[None])
    fut = qt > cut
    return {'omega_hat': slope(ang), 'omega': omega,
            'growth_hat': slope(torch.log(r.clamp_min(1e-4))), 'growth': growth,
            'radius_rel_err': ((r - true_r).abs() / true_r).mean(-1),
            'radius_rel_err_future': ((r[:, fut] - true_r[:, fut]).abs() / true_r[:, fut]).mean(-1),
            'circle_radius_cv': r.std(-1) / r.mean(-1)}


records = {k: [] for k in ['new', 'new_sampled', 'old', 'const_pos', 'const_vel']}
geo = {'new': [], 'old': [], 'truth': []}
var = []
keep = {'truth': [], 'obs': [], 'obs_t': [], 'new': [], 'old': [], 'family': []}
start = 0
with torch.no_grad():
    for P in batches:
        B = P['obs_xy'].shape[0]; prm = params[start:start + B]; start += B
        fam = P['family'].cpu()
        preds = {}
        for name, m in (('new', new_model), ('old', old_model)):
            o = m(P['obs_xy'], P['obs_t'], P['query_t'], sample=False)['prediction']
            preds[name] = o
            mets = infer_test_metrics(o, P, cut)
            records[name] += [{'family': int(f), **{k: float(v[i]) for k, v in mets.items()}} for i, f in enumerate(fam)]
        # sampled posterior: average each metric over K independent z0 samples
        acc = None
        for _ in range(a.n_samples):
            o = new_model(P['obs_xy'], P['obs_t'], P['query_t'], sample=True)['prediction']
            mets = infer_test_metrics(o, P, cut)
            acc = mets if acc is None else {k: acc[k] + v for k, v in mets.items()}
        records['new_sampled'] += [{'family': int(f), **{k: float(v[i]) / a.n_samples for k, v in acc.items()}} for i, f in enumerate(fam)]
        cp, cv = baselines(P)
        for name, b in (('const_pos', cp), ('const_vel', cv)):
            mets = infer_test_metrics((b - P['mean']) / P['scale'], P, cut)
            records[name] += [{'family': int(f), **{k: float(v[i]) for k, v in mets.items()}} for i, f in enumerate(fam)]
        truth = P['raw_truth']
        for name in ('new', 'old'):
            g = geometry(to_orig(preds[name], P), P, prm)
            geo[name] += [{'family': int(f), **{k: float(v[i]) for k, v in g.items()}} for i, f in enumerate(fam)]
        g = geometry(truth, P, prm)
        geo['truth'] += [{'family': int(f), **{k: float(v[i]) for k, v in g.items()}} for i, f in enumerate(fam)]
        for name in ('new', 'old'):
            pv = to_orig(preds[name], P).var(1).sum(-1); tv = truth.var(1).sum(-1)
            if name == 'new':
                var += [{'family': int(f), 'pred_var_new': float(pv[i]), 'truth_var': float(tv[i])} for i, f in enumerate(fam)]
            else:
                for i in range(B): var[-B + i]['pred_var_old'] = float(pv[i])
        keep['truth'].append(truth.cpu()); keep['obs'].append(to_orig(P['obs_xy'], P).cpu())
        keep['obs_t'].append(P['obs_t'].cpu()); keep['family'].append(fam)
        keep['new'].append(to_orig(preds['new'], P).cpu()); keep['old'].append(to_orig(preds['old'], P).cpu())
K = {k: torch.cat(v) for k, v in keep.items()}
QT = batches[0]['query_t'].cpu().numpy()

KEYS = ['recon_rmse', 'forecast_rmse', 'forecast_mae', 'forecast_ade', 'forecast_fde', 'chamfer_full']
FAM = {'circle': CIRCLE, 'spiral': SPIRAL}


def summarize(entries):
    return {'n': len(entries), **{k: {'mean': float(np.mean([e[k] for e in entries])),
                                      'std': float(np.std([e[k] for e in entries])),
                                      'median': float(np.median([e[k] for e in entries]))} for k in KEYS}}


def by_family(rec):
    return {'overall': summarize(rec), **{f: summarize([e for e in rec if e['family'] == i]) for f, i in FAM.items()}}


def geo_summary(rows, fid):
    r = [e for e in rows if e['family'] == fid]
    om = np.array([e['omega'] for e in r]); omh = np.array([e['omega_hat'] for e in r])
    gr = np.array([e['growth'] for e in r]); grh = np.array([e['growth_hat'] for e in r])
    s = {'rotation_direction_accuracy': float(np.mean(np.sign(om) == np.sign(omh))),
         'angular_velocity_abs_err_mean': float(np.mean(np.abs(omh - om))),
         'radius_rel_err_mean': float(np.mean([e['radius_rel_err'] for e in r])),
         'radius_rel_err_future_mean': float(np.mean([e['radius_rel_err_future'] for e in r]))}
    if fid == CIRCLE:
        s['radius_coef_of_variation_mean (0 = perfect circle)'] = float(np.mean([e['circle_radius_cv'] for e in r]))
        s['fitted_growth_abs_mean (0 = no radial drift)'] = float(np.mean(np.abs(grh)))
    else:
        s['growth_sign_accuracy (expansion vs contraction)'] = float(np.mean(np.sign(gr) == np.sign(grh)))
        s['growth_rate_abs_err_mean'] = float(np.mean(np.abs(grh - gr)))
        s['growth_rate_corr'] = float(np.corrcoef(gr, grh)[0, 1])
    return s


def var_summary(fid):
    r = [e for e in var if e['family'] == fid]
    tv = np.mean([e['truth_var'] for e in r])
    return {'truth_temporal_variance_mean': float(tv),
            'improved_pred_variance_mean': float(np.mean([e['pred_var_new'] for e in r])),
            'original_pred_variance_mean': float(np.mean([e['pred_var_old'] for e in r])),
            'improved_variance_ratio': float(np.mean([e['pred_var_new'] for e in r]) / tv),
            'original_variance_ratio': float(np.mean([e['pred_var_old'] for e in r]) / tv)}


# "average-trajectory" check: RMSE of predicting the per-family mean test trajectory
avg = {}
for f, i in FAM.items():
    tr = K['truth'][K['family'] == i]
    avg[f] = float((tr - tr.mean(0, keepdim=True)).square().mean().sqrt())

# Solver cross-check: re-integrate the improved model with tight-tolerance dopri5.
solver_check = None
if new_model.solver != 'dopri5':
    P = batches[0]
    with torch.no_grad():
        ref_pred = new_model(P['obs_xy'], P['obs_t'], P['query_t'], sample=False)['prediction']
        s0, st0 = new_model.solver, new_model.step_size
        new_model.solver, new_model.step_size, new_model.rtol, new_model.atol = 'dopri5', None, 1e-6, 1e-7
        tight = new_model(P['obs_xy'], P['obs_t'], P['query_t'], sample=False)['prediction']
        new_model.solver, new_model.step_size = s0, st0
    diff = ((ref_pred - tight) * P['scale']).norm(dim=-1)
    solver_check = {'trained_solver': f'{s0} step={st0}', 'reference': 'dopri5 rtol=1e-6 atol=1e-7',
                    'max_pointwise_diff': float(diff.max()), 'mean_pointwise_diff': float(diff.mean())}

report = {
    'checkpoint': a.checkpoint, 'solver_cross_check': solver_check, 'original_checkpoint_path': a.original, 'split': a.split,
    'checkpoint_epoch': new_ck['epoch'], 'solver_backend': new_ck['config'].get('solver_backend'),
    'model': new_ck['model'], 'n_obs': a.n_obs, 'n_query': a.n_query, 'forecast_start': cut,
    'metric_units': 'original coordinate units; per-trajectory metrics then averaged',
    'improved_posterior_mean': by_family(records['new']),
    'improved_sampled_posterior': by_family(records['new_sampled']),
    'original_checkpoint': by_family(records['old']),
    'geometry': {name: {f: geo_summary(geo[name], i) for f, i in FAM.items()} for name in ('truth', 'new', 'old')},
    'variance': {f: var_summary(i) for f, i in FAM.items()},
    'rmse_if_predicting_family_mean_trajectory': avg,
}
(out / 'evaluation.json').write_text(json.dumps(report, indent=2))
bl = {name: by_family(records[k]) for name, k in (('latent_ode_improved', 'new'), ('latent_ode_original', 'old'),
                                                    ('constant_position', 'const_pos'), ('constant_velocity', 'const_vel'))}
bl['note'] = ('Baselines use linear interpolation of the 30 noisy observations for the reconstruction '
              'interval; forecast holds the last observation (constant position) or extrapolates the '
              'velocity of the last two observations (constant velocity).')
(out / 'baseline_comparison.json').write_text(json.dumps(bl, indent=2))
with open(out / 'per_trajectory_metrics.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['index', 'family', 'model'] + KEYS)
    for name in ('new', 'old', 'const_pos', 'const_vel'):
        for i, e in enumerate(records[name]): w.writerow([i, e['family'], name] + [e[k] for k in KEYS])

# ------------------------------------------------------------------ figures
GREEN, BLUE, RED = '#2a9469', '#2578bc', '#e0533a'
fut = QT > cut
idx_f = {f: torch.nonzero(K['family'] == i).flatten().tolist() for f, i in FAM.items()}


def traj_panel(ax, k, pred, title):
    tr, ob = K['truth'][k].numpy(), K['obs'][k].numpy()
    ax.plot(tr[:, 0], tr[:, 1], color=GREEN, lw=2.6, label='Clean ground truth', zorder=1)
    ax.plot(pred[~fut, 0], pred[~fut, 1], color=BLUE, lw=1.7, label='Reconstruction (t≤7)', zorder=2)
    ax.plot(pred[fut, 0], pred[fut, 1], color=RED, lw=1.9, label='Forecast (t>7)', zorder=2)
    ax.scatter(ob[:, 0], ob[:, 1], s=14, color='black', alpha=.8, label='Noisy observations', zorder=3)
    ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.15); ax.set_title(title, fontsize=9)


for f in FAM:
    ks = idx_f[f][:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.2))
    for ax, k in zip(axes.flat, ks):
        e = records['new'][k]
        traj_panel(ax, k, K['new'][k].numpy(), f'{f} test #{k} | recon RMSE {e["recon_rmse"]:.3f} | fc RMSE {e["forecast_rmse"]:.3f}')
    axes.flat[0].legend(fontsize=7, frameon=False, loc='best')
    fig.suptitle(f'{f.capitalize()}: improved Latent ODE on the first 6 {f}s of the test split (not cherry-picked)')
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(out / f'{f}_reconstruction.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for r, k in enumerate(ks[:3]):
        for c, nm in enumerate('xy'):
            ax = axes[r, c]
            ax.plot(QT, K['truth'][k][:, c], color=GREEN, lw=2.4, label=f'{nm}(t) truth')
            ax.plot(QT[~fut], K['new'][k][~fut, c], color=BLUE, lw=1.6, label=f'{nm}̂(t) reconstruction')
            ax.plot(QT[fut], K['new'][k][fut, c], color=RED, lw=1.8, label=f'{nm}̂(t) forecast')
            ax.scatter(K['obs_t'][k], K['obs'][k][:, c], s=10, color='black', label='observations', zorder=3)
            ax.axvline(cut, color='gray', ls=':'); ax.grid(alpha=.15); ax.set_ylabel(f'{nm}  (test #{k})')
            if r == 0: ax.legend(fontsize=7, frameon=False, ncol=2)
    axes[-1, 0].set_xlabel('t'); axes[-1, 1].set_xlabel('t')
    fig.suptitle(f'{f.capitalize()}: coordinate predictions vs time (timing check)')
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(out / f'{f}_coordinates.png', dpi=150); plt.close(fig)

# before / after on identical trajectories with identical axis limits per row
rows = idx_f['circle'][:2] + idx_f['spiral'][:2]
fig, axes = plt.subplots(len(rows), 2, figsize=(10, 4.6 * len(rows)))
for r, k in enumerate(rows):
    fam = 'circle' if int(K['family'][k]) == CIRCLE else 'spiral'
    for c, (name, lab) in enumerate((('old', 'Original checkpoint (epoch %d)' % old_ck['epoch']),
                                     ('new', 'Improved checkpoint (epoch %d)' % new_ck['epoch']))):
        e = records[name][k]
        traj_panel(axes[r, c], k, K[name][k].numpy(), f'{lab}\n{fam} test #{k}: recon {e["recon_rmse"]:.3f}, forecast {e["forecast_rmse"]:.3f}')
    allp = torch.cat([K['truth'][k], K['old'][k], K['new'][k], K['obs'][k]])
    lo, hi = allp.min(0).values.numpy() - .15, allp.max(0).values.numpy() + .15
    for c in range(2):
        axes[r, c].set_xlim(lo[0], hi[0]); axes[r, c].set_ylim(lo[1], hi[1]); axes[r, c].set_aspect('equal')
axes[0, 0].legend(fontsize=7, frameon=False)
fig.suptitle('Before vs after: same test trajectories, same observations, same axis limits')
fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(out / 'before_after_comparison.png', dpi=140); plt.close(fig)

# worst cases (honesty check)
fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
for c, f in enumerate(FAM):
    ii = idx_f[f]; order = sorted(ii, key=lambda k: records['new'][k]['forecast_rmse'])
    for j, (k, tag) in enumerate(((order[len(order) // 2], 'median'), (order[-1], 'worst'))):
        e = records['new'][k]
        traj_panel(axes[2 * c + j], k, K['new'][k].numpy(), f'{f} {tag} forecast #{k}\nrecon {e["recon_rmse"]:.3f}, forecast {e["forecast_rmse"]:.3f}')
fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(out / 'median_and_worst_cases.png', dpi=130); plt.close(fig)

if a.history:
    import pandas as pd
    h = pd.read_csv(root / a.history)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    axes[0].plot(h.epoch, h.train_loss, label='train (β-weighted)'); axes[0].plot(h.epoch, h.val_elbo_loss, label='val (β=1, posterior mean)')
    axes[0].set_yscale('symlog'); axes[0].set_title('Total objective (−ELBO per trajectory)')
    axes[1].plot(h.epoch, h.train_nll, label='train'); axes[1].plot(h.epoch, h.val_nll, label='val')
    axes[1].set_yscale('symlog'); axes[1].set_title('Reconstruction NLL')
    axes[2].plot(h.epoch, h.train_kl, label='train'); axes[2].plot(h.epoch, h.val_kl, label='val')
    ax2 = axes[2].twinx(); ax2.plot(h.epoch, h.beta, color='gray', ls=':', label='β'); ax2.set_ylabel('β')
    axes[2].set_title('KL divergence (nats)')
    axes[3].plot(h.epoch, h.val_context_rmse, label='val context RMSE'); axes[3].plot(h.epoch, h.val_forecast_rmse, label='val forecast RMSE')
    axes[3].set_yscale('log'); axes[3].set_title('Validation RMSE (orig. units)')
    best = int(new_ck['epoch'])
    for ax in axes:
        ax.axvline(best, color='k', ls='--', lw=.8); ax.set_xlabel('epoch'); ax.grid(alpha=.2); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(out / 'loss_curves.png', dpi=140); plt.close(fig)

print(json.dumps({k: report[k] for k in ('improved_posterior_mean', 'variance')}, indent=1)[:4000])
print('Saved to', out)
