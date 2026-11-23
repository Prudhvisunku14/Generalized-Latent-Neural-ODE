#!/usr/bin/env python3
"""Five-shape study: evaluation only (no weights are changed; checksums are asserted).

  python scripts/31_evaluate_five_shapes.py zeroshot   # frozen three-shape model on all five test families (+ baselines)
  python scripts/31_evaluate_five_shapes.py compare    # five-shape model vs frozen three-shape model, paired
  python scripts/31_evaluate_five_shapes.py figures

Protocol (identical to results/validation/three_shapes): test split, 30 stored noisy observations at t <= 7 drawn with
observation seed 123 in dataset order, 200-point query grid, posterior mean z0. Every model sees the SAME observation
timestamps and values (asserted). Each model's inputs are normalised with its own training-set constants.
recon_rmse   = RMSE over the context timestamps (t <= 7) that were NOT observed (interpolation).
forecast_rmse = RMSE over all timestamps with 7 < t <= 10 (extrapolation). Both per trajectory, original units.
Baselines: linear interpolation of the observations for reconstruction; hold-last-observation (const_pos) and
constant velocity from the last two observations (const_vel) for forecasting.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, torch
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from glo.evaluation import load_model, collect, by_family, baseline
from glo.inference import sha256_file

ap = argparse.ArgumentParser(); ap.add_argument('stages', nargs='+', choices=['zeroshot', 'compare', 'figures']); args = ap.parse_args()
torch.set_num_threads(4)
root = Path(__file__).resolve().parents[1]; data_root = root / 'data/processed'
OUT = root / 'results/five_shapes'; TAB = OUT / 'tables'; FIG = OUT / 'figures'
for d in (TAB, FIG): d.mkdir(parents=True, exist_ok=True)
THREE = root / 'saved_models/three_shapes_unified/best_model.pt'
FIVE = root / 'checkpoints/five_shapes/five_shapes_seed42.pt'
FAMS = ['circle', 'spiral', 'ellipse', 'figure_eight', 'sinusoid']
SEEN3 = {'circle', 'spiral', 'ellipse'}
KEYS = ['recon_rmse', 'forecast_rmse']
EXISTING = root / 'results/validation/three_shapes/three_shapes_and_ood.json'


def three_sha_ok():
    return sha256_file(THREE) == json.loads((THREE.parent / 'metadata.json').read_text())['sha256']


def per_traj(res):
    names = {v: k for k, v in res['meta']['family_to_id'].items()}
    return {'ids': res['ids'].tolist(), 'family': [names[int(f)] for f in res['family']],
            **{k: res[k].astype(float).tolist() for k in KEYS}}


def same_observations(a, b):
    return (np.array_equal(a['obs_t'], b['obs_t']) and np.array_equal(a['obs_mask'], b['obs_mask'])
            and np.allclose(a['obs'], b['obs'], atol=1e-5) and np.array_equal(a['ids'], b['ids']))


def run_three_and_baselines():
    m3, ck3 = load_model(THREE, 'cpu')
    R3 = collect(m3, ck3, data_root, 'five_shapes')
    B = {k: collect(*baseline(k, 'five_shapes'), data_root, 'five_shapes') for k in ('const_pos', 'const_vel')}
    assert all(same_observations(R3, b) for b in B.values()), 'baselines saw different observations'
    return m3, ck3, R3, B


def stage_zeroshot():
    assert three_sha_ok()
    m3, ck3, R3, B = run_three_and_baselines()
    # Control: the frozen model on the ORIGINAL three-shape test split must reproduce the stored validation results.
    old = by_family(collect(m3, ck3, data_root, 'three_shapes'))
    stored = json.loads(EXISTING.read_text())['A_three_shape_model_in_distribution']['model']
    repro = {f: {k: [old[f][k]['mean'], stored[f][k]['mean']] for k in KEYS} for f in stored}
    max_dev = max(abs(a - b) for f in repro.values() for a, b in f.values())
    out = {'model': str(THREE.relative_to(root)), 'best_epoch': int(ck3['epoch']), 'model_sha256_ok': True,
           'trained_on': 'circle, spiral, ellipse (three_shapes)', 'normalisation': 'three_shapes training constants',
           'protocol': 'five_shapes test split (750 = 150 per family), 30 stored noisy observations, observation seed 123, posterior mean',
           'reproduces_stored_three_shape_results': {'max_abs_deviation': max_dev, 'values_new_vs_stored': repro},
           'three_shape_model': by_family(R3), 'const_pos': by_family(B['const_pos']), 'const_vel': by_family(B['const_vel']),
           'per_trajectory': {'three_shape_model': per_traj(R3), 'const_pos': per_traj(B['const_pos']), 'const_vel': per_traj(B['const_vel'])}}
    assert max_dev < 1e-4, f'frozen model does not reproduce stored results ({max_dev})'
    (OUT / 'zeroshot_three_shape_model.json').write_text(json.dumps(out, indent=2))
    T = ['| Family | status for the three-shape model | n | Recon. RMSE mean (median) | linear-interp. recon. RMSE mean (median) | '
         'Forecast RMSE mean (median) | hold-last forecast RMSE mean | const.-velocity forecast RMSE mean |', '|---|---|---|---|---|---|---|---|']
    for f in FAMS + ['overall']:
        a, p, v = out['three_shape_model'][f], out['const_pos'][f], out['const_vel'][f]
        st = 'all five' if f == 'overall' else ('seen in training' if f in SEEN3 else '**unseen (zero-shot)**')
        T.append(f"| {f} | {st} | {a['n']} | {a['recon_rmse']['mean']:.4f} ({a['recon_rmse']['median']:.4f}) | "
                 f"{p['recon_rmse']['mean']:.4f} ({p['recon_rmse']['median']:.4f}) | {a['forecast_rmse']['mean']:.4f} ({a['forecast_rmse']['median']:.4f}) | "
                 f"{p['forecast_rmse']['mean']:.4f} | {v['forecast_rmse']['mean']:.4f} |")
    (TAB / 'table_zeroshot_three_shape_model.md').write_text('\n'.join(T) + '\n'); print('\n'.join(T)); print('reproduction max deviation', max_dev)
    assert three_sha_ok()


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.
    for r, i in enumerate(order):
        run = max(run, min(1., (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def stage_compare():
    assert three_sha_ok(); five_sha = sha256_file(FIVE)
    m3, ck3, R3, B = run_three_and_baselines()
    m5, ck5 = load_model(FIVE, 'cpu')
    R5 = collect(m5, ck5, data_root, 'five_shapes')
    assert same_observations(R3, R5), 'three- and five-shape models saw different observations'
    assert np.isfinite(R5['pred']).all() and np.isfinite(R3['pred']).all()
    hist = np.genfromtxt(FIVE.with_name(FIVE.stem + '_history.csv'), delimiter=',', names=True)
    out = {'five_shape_checkpoint': str(FIVE.relative_to(root)), 'five_shape_best_epoch': int(ck5['epoch']),
           'five_shape_epochs_run': int(hist['epoch'].max()), 'five_shape_best_val_neg_elbo': float(ck5['val_loss']),
           'five_shape_sha256': five_sha, 'three_shape_model': str(THREE.relative_to(root)),
           'protocol': 'five_shapes test split; identical observations for every model (asserted); observation seed 123; posterior mean',
           'observations_identical_across_models': True, 'predictions_finite': True,
           'five_shape_model': by_family(R5), 'three_shape_model': by_family(R3),
           'const_pos': by_family(B['const_pos']), 'const_vel': by_family(B['const_vel']), 'paired': {}}
    tests = []
    for f in FAMS:
        s = np.array([n == f for n in per_traj(R5)['family']])
        for k in KEYS:
            x, y = R5[k][s], R3[k][s]; d = x - y
            ci = stats.t.interval(.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
            p = float(stats.wilcoxon(x, y).pvalue)
            out['paired'][f'{f}/{k}'] = {'n': int(s.sum()), 'five_mean': float(x.mean()), 'three_mean': float(y.mean()),
                                         'mean_difference_five_minus_three': float(d.mean()), 'ci95': [float(ci[0]), float(ci[1])],
                                         'relative_change_of_mean': float(x.mean() / y.mean() - 1),
                                         'five_better_fraction': float((x < y).mean()), 'wilcoxon_p': p}
            tests.append((f'{f}/{k}', p))
    for (key, _), a in zip(tests, holm(np.array([p for _, p in tests]))):
        out['paired'][key]['wilcoxon_p_holm'] = float(a)
    out['per_trajectory'] = {'five_shape_model': per_traj(R5), 'three_shape_model': per_traj(R3)}
    (OUT / 'compare_three_vs_five.json').write_text(json.dumps(out, indent=2))
    np.savez_compressed(OUT / 'predictions_test.npz', ids=R5['ids'], family=R5['family'], query_t=R5['query_t'], truth=R5['truth'],
                        obs=R5['obs'], obs_t=R5['obs_t'], pred_five=R5['pred'], pred_three=R3['pred'],
                        pred_const_vel=B['const_vel']['pred'])
    T = ['| Family | three-shape model (frozen) | five-shape model | Forecast RMSE: three-shape | Forecast RMSE: five-shape | '
         'difference [95% CI] | five-shape better on | Holm p | Recon. RMSE: three-shape | Recon. RMSE: five-shape | Holm p (recon.) |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for f in FAMS:
        P, Q = out['paired'][f'{f}/forecast_rmse'], out['paired'][f'{f}/recon_rmse']
        T.append(f"| {f} | {'trained on' if f in SEEN3 else '**unseen**'} | trained on | {P['three_mean']:.4f} | {P['five_mean']:.4f} | "
                 f"{P['mean_difference_five_minus_three']:+.4f} [{P['ci95'][0]:+.4f}, {P['ci95'][1]:+.4f}] | {P['five_better_fraction'] * 100:.0f}% | "
                 f"{P['wilcoxon_p_holm']:.2g} | {Q['three_mean']:.4f} | {Q['five_mean']:.4f} | {Q['wilcoxon_p_holm']:.2g} |")
    o3, o5 = out['three_shape_model']['overall'], out['five_shape_model']['overall']
    T.append(f"| overall (750) | | | {o3['forecast_rmse']['mean']:.4f} | {o5['forecast_rmse']['mean']:.4f} | | | | "
             f"{o3['recon_rmse']['mean']:.4f} | {o5['recon_rmse']['mean']:.4f} | |")
    T += ['', '| Family | five-shape recon. RMSE mean (median) | linear interp. | five-shape forecast RMSE mean (median) | hold-last | const. velocity |',
          '|---|---|---|---|---|---|']
    for f in FAMS + ['overall']:
        a, p, v = out['five_shape_model'][f], out['const_pos'][f], out['const_vel'][f]
        T.append(f"| {f} | {a['recon_rmse']['mean']:.4f} ({a['recon_rmse']['median']:.4f}) | {p['recon_rmse']['mean']:.4f} | "
                 f"{a['forecast_rmse']['mean']:.4f} ({a['forecast_rmse']['median']:.4f}) | {p['forecast_rmse']['mean']:.4f} | {v['forecast_rmse']['mean']:.4f} |")
    (TAB / 'table_three_vs_five.md').write_text('\n'.join(T) + '\n'); print('\n'.join(T))
    assert three_sha_ok() and sha256_file(FIVE) == five_sha


def stage_figures():
    z = np.load(OUT / 'predictions_test.npz'); C = json.loads((OUT / 'compare_three_vs_five.json').read_text())
    qt = z['query_t']; fut = qt > 7
    # Figure A: two test trajectories per family (the first two in test order, not chosen by error)
    fig, axes = plt.subplots(5, 2, figsize=(10, 22))
    for r, f in enumerate(FAMS):
        for c, k in enumerate(np.flatnonzero(z['family'] == r)[:2]):
            ax = axes[r, c]
            ax.plot(*z['truth'][k].T, color='#999', lw=3, alpha=.6, label='ground truth')
            for P, col, lab in ((z['pred_three'][k], '#d1495b', 'three-shape model'), (z['pred_five'][k], '#2578bc', 'five-shape model')):
                ax.plot(*P[~fut].T, color=col, lw=1.4, label=f'{lab}: reconstruction (t <= 7)')
                ax.plot(*P[fut].T, color=col, lw=1.4, ls='--', label=f'{lab}: forecast (t > 7)')
            ax.scatter(*z['obs'][k].T, s=10, color='k', zorder=4, label='30 observations')
            ax.plot(*z['truth'][k][fut][0], 'ko', mfc='none', ms=9)
            ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.15)
            ax.set_title(f"{f}{' (unseen by three-shape model)' if f not in SEEN3 else ''} / test #{c + 1}", fontsize=10)
            if r == 0 and c == 0: ax.legend(fontsize=7, frameon=False, loc='best')
    fig.suptitle('Test trajectories: frozen three-shape vs five-shape model (open circle = forecast start, t = 7)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .985)); fig.savefig(FIG / 'examples_three_vs_five.png', dpi=120); plt.close(fig)
    # Figure B: per-family forecast and reconstruction RMSE with baselines
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6)); x = np.arange(len(FAMS)); w = .2
    for ax, k, bl in ((axes[0], 'recon_rmse', [('const_pos', 'linear interpolation', '#3a9d5d')]),
                      (axes[1], 'forecast_rmse', [('const_pos', 'hold last', '#3a9d5d'), ('const_vel', 'constant velocity', '#e0a030')])):
        series = [('three_shape_model', 'three-shape model (frozen)', '#d1495b'), ('five_shape_model', 'five-shape model', '#2578bc')] + bl
        for i, (m, lab, col) in enumerate(series):
            ax.bar(x + (i - (len(series) - 1) / 2) * w, [C[m][f][k]['mean'] for f in FAMS], w, color=col, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([f + ('\n(unseen by 3-shape)' if f not in SEEN3 else '') for f in FAMS], fontsize=8)
        ax.set_ylabel(f"mean {k.replace('_', ' ').replace('rmse', 'RMSE')} (data units)"); ax.legend(frameon=False, fontsize=8); ax.grid(axis='y', alpha=.2)
    axes[0].set_title('Reconstruction of withheld context points (t <= 7)'); axes[1].set_title('Forecast (7 < t <= 10)')
    fig.tight_layout(); fig.savefig(FIG / 'per_family_rmse_three_vs_five.png', dpi=130); plt.close(fig)
    # Figure C: forecast error vs time, per family
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.8), sharey=True)
    for r, f in enumerate(FAMS):
        s = z['family'] == r
        for P, col, lab in ((z['pred_three'], '#d1495b', 'three-shape'), (z['pred_five'], '#2578bc', 'five-shape'), (z['pred_const_vel'], '#e0a030', 'const. velocity')):
            e = np.linalg.norm(P[s] - z['truth'][s], axis=-1)
            axes[r].plot(qt, np.median(e, 0), color=col, label=lab)
        axes[r].axvline(7, color='gray', ls=':'); axes[r].set_title(f); axes[r].set_xlabel('t'); axes[r].set_yscale('log'); axes[r].grid(alpha=.2)
    axes[0].set_ylabel('median Euclidean error'); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / 'error_vs_time_three_vs_five.png', dpi=130); plt.close(fig)
    print('figures written to', FIG)


for s in args.stages:
    globals()[f'stage_{s}']()
