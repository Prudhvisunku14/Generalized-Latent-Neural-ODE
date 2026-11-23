#!/usr/bin/env python3
"""Phase 11 (conditional): package the GeoLife-trained Latent ODEs into saved_models/geolife_trained/ and compare them with
the frozen synthetic model and the baselines on the SAME held-out test segments and observation subsets as scripts/21.

  python scripts/23_geolife_trained_models.py package evaluate
The original three-shape model and every pre-existing checkpoint stay untouched (checked).
"""
import argparse, datetime, json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd, torch, yaml
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from glo import geolife as gl, geolife_evaluation as ge, inference as inf
from glo.model import LatentODE

torch.set_num_threads(1)
root = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument('stages', nargs='+', choices=['package', 'evaluate']); args = ap.parse_args()
C = yaml.safe_load(open(root / 'configs/geolife.yaml')); E = C['experiment']
RES = root / 'results/geolife'; TAB = RES / 'tables'; FIG = RES / 'figures'
OUT = root / 'saved_models/geolife_trained'
RUNS = {'scratch': ('checkpoints/geolife/geolife_scratch_seed42.pt', 'trained from random initialisation on GeoLife (dev users)'),
        'finetuned_from_three_shapes': ('checkpoints/geolife/geolife_finetune_from_three_shapes_seed42.pt',
                                        'three_shapes_unified weights fine-tuned on GeoLife (dev users), lr 3e-4')}
EVAL_STRATEGY = gl.NormStrategy('S1_obsmean_rms_1.0', 'obs_mean', 'obs_rms', 1.0)   # the training convention, fixed in advance


def frozen_snapshot():
    files = sorted((root / 'checkpoints').rglob('*.pt')) + sorted((root / 'saved_models').rglob('best_model.pt'))
    return {str(p.relative_to(root)): inf.sha256_file(p) for p in files if 'geolife' not in str(p.relative_to(root))}


def stage_package():
    dmeta = json.loads((root / 'data/geolife/processed/geolife_regular/metadata.json').read_text())
    for name, (src, desc) in RUNS.items():
        src = root / src; ck = torch.load(src, map_location='cpu', weights_only=True)
        hist = pd.read_csv(src.with_name(src.stem + '_history.csv'))
        d = OUT / name; d.mkdir(parents=True, exist_ok=True)
        dst = d / 'best_model.pt'
        if dst.exists() and inf.sha256_file(dst) != inf.sha256_file(src):
            sys.exit(f'{dst} exists and differs from {src}; refusing to overwrite')
        shutil.copy2(src, dst); assert inf.sha256_file(dst) == inf.sha256_file(src)
        m = LatentODE(**ck['model']); m.load_state_dict(ck['state_dict'])
        n_par = sum(p.numel() for p in m.parameters())
        (d / 'model_config.json').write_text(json.dumps({'architecture': ck['model'], 'n_parameters': n_par,
                                                         'module_repr': str(m).splitlines()}, indent=2))
        (d / 'preprocessing.json').write_text(json.dumps({
            'input_columns': ['t', 'x', 'y'], 'normalization_mean': dmeta['normalization_mean'], 'normalization_scale': dmeta['normalization_scale'],
            'normalization_source': 'data/geolife/processed/geolife_regular/metadata.json (train split, AFTER per-segment normalisation)',
            'per_segment_gps_normalisation': 'local metres (AEQD at the first record) -> (xy - mean of observations) / RMS distance of observations about it; '
                                             'trained with the same map computed from the 140 context records',
            'time_scale_T': 10.0, 'time_horizon': 10.0, 'context_end': 7.0, 'training_n_obs': 30,
            'encoder_features': '[x_norm, y_norm, t/T, dt/T] per observation', 'integration_origin_t': 0.0,
            'latent_state_used': 'posterior mean z0 = mu (deterministic)', 'training_observation_noise_std': 0.05,
            'coordinate_abs_max_train': float(np.abs(np.load(root / 'data/geolife/processed/geolife_regular/train.npz')['xy']).max())}, indent=2))
        best = hist.loc[hist.val_elbo_loss.idxmin()]
        (d / 'metadata.json').write_text(json.dumps({
            'model_name': f'geolife_trained/{name}', 'description': desc, 'n_parameters': n_par, 'training_seed': 42,
            'best_checkpoint_epoch': int(ck['epoch']), 'epochs_run': int(hist.epoch.max()), 'epochs_configured': 300,
            'selection': 'minimum validation -ELBO on the validation USERS (no test user involved)', 'best_val_neg_elbo': float(best.val_elbo_loss),
            'training_dataset': 'geolife_regular', 'training_users': dmeta['users']['train'], 'validation_users': dmeta['users']['val'],
            'split_counts': dmeta['split_counts'], 'original_checkpoint': str(src.relative_to(root)), 'sha256': inf.sha256_file(dst),
            'training_config': ck['config']['training'], 'torch_version': torch.__version__,
            'packaged_on': datetime.date.today().isoformat(), 'licence_note': 'trained on GeoLife (MSR-LA, non-commercial; do not redistribute)'}, indent=2, default=str))
        lm = inf.load_model_folder(d)                     # full verification path
        print(name, 'epoch', ck['epoch'], 'params', lm.n_parameters, 'sha', lm.sha256[:12])


def load_split(name):
    return pd.read_csv(root / f'data/geolife/splits/{name}_segments.csv'), np.load(root / f'data/geolife/processed/{name}_segments.npz')


def obs_seed(i, n):
    return int(E['observation_seed_base'] + 1_000_003 * n + i)


def stage_evaluate():
    meta, arr = load_split('test')
    ns = json.loads((TAB / 'normalization_strategies.json').read_text()); c = ns['strategies'][ns['selected']]['config']
    frozen_strat = gl.NormStrategy(c['name'], c['center'], c['spread'], c['target'], c['rotate'], tuple(c['offset']))
    models = {'frozen_three_shapes': (inf.load_packaged_model(E['model'], root / 'saved_models'), frozen_strat),
              'frozen_three_shapes_S1': (inf.load_packaged_model(E['model'], root / 'saved_models'), EVAL_STRATEGY)}
    for name in RUNS:
        models[f'geolife_{name}'] = (inf.load_model_folder(OUT / name), EVAL_STRATEGY)
    rows = []
    for n in E['density_levels']:
        for i in range(len(meta)):
            task = gl.build_task(arr['t_unix'][i], arr['lat'][i], arr['lon'][i], n, obs_seed(i, n), E['context_end'])
            r = {'n_obs': n, 'segment_id': meta.segment_id[i], 'user': str(meta.user[i]).zfill(3)}
            base = ge.evaluate_interpolation(task, None, None, methods=('linear', 'cubic', 'pchip'))
            for m in ('linear', 'cubic', 'pchip'):
                r[f'interp_{m}'] = base[m]['rmse_xy']
            for mn, (lm, st) in models.items():
                r[f'interp_{mn}'] = ge.evaluate_interpolation(task, lm, st, methods=('latent_ode',))['latent_ode']['rmse_xy']
            if n == E['n_obs']:
                for P in ('A', 'B'):
                    fb = ge.evaluate_forecast(task, None, None, P, methods=('hold_last', 'const_velocity'))
                    r[f'fc{P}_hold_last'] = fb['hold_last']['rmse_xy']; r[f'fc{P}_const_velocity'] = fb['const_velocity']['rmse_xy']
                    for mn, (lm, st) in models.items():
                        r[f'fc{P}_{mn}'] = ge.evaluate_forecast(task, lm, st, P, methods=('latent_ode',))['latent_ode']['rmse_xy']
            rows.append(r)
        print('n_obs', n, 'done', flush=True)
    df = pd.DataFrame(rows); df.to_csv(RES / 'main_evaluation/geolife_trained_per_segment.csv', index=False)
    M = ['linear', 'cubic', 'pchip'] + list(models)
    out = {'test_segments': int(len(meta)), 'test_users': int(meta.user.nunique()),
           'strategies': {'frozen_three_shapes': frozen_strat.__dict__, 'others': EVAL_STRATEGY.__dict__},
           'note': 'same segments and observation subsets as results/geolife/main_evaluation (seeds identical); RMSE in metres; '
                   'frozen_three_shapes_S1 = the frozen model with the GeoLife-trained models\' normalisation (control for normalisation)',
           'interpolation': {}, 'forecast_n30': {}}
    for n, g in df.groupby('n_obs'):
        out['interpolation'][str(n)] = {m: {'median': float(g[f'interp_{m}'].median()), 'mean': float(g[f'interp_{m}'].mean()),
                                            'median_ci95_user_bootstrap': list(ge.cluster_bootstrap(g[f'interp_{m}'].to_numpy(), g.user.to_numpy(), np.median)),
                                            'win_rate_vs_linear': float((g[f'interp_{m}'] < g.interp_linear).mean()),
                                            'win_rate_vs_pchip': float((g[f'interp_{m}'] < g.interp_pchip).mean()),
                                            'geometric_mean_ratio_vs_linear': float(np.exp(np.log(g[f'interp_{m}'] / g.interp_linear).mean()))} for m in M}
    g = df[df.n_obs == E['n_obs']]
    from scipy.stats import wilcoxon
    out['paired_n30'] = {}
    for a in [f'geolife_{k}' for k in RUNS]:
        for b in ('linear', 'pchip', 'frozen_three_shapes'):
            x, y = g[f'interp_{a}'].to_numpy(), g[f'interp_{b}'].to_numpy(); lr = np.log(x / y)
            ci = ge.cluster_bootstrap(lr, g.user.to_numpy(), np.mean)
            out['paired_n30'][f'{a}_vs_{b}'] = {'geometric_mean_ratio': float(np.exp(lr.mean())), 'ci95': [float(np.exp(ci[0])), float(np.exp(ci[1]))],
                                                'win_rate': float((x < y).mean()), 'wilcoxon_p': float(wilcoxon(x, y).pvalue)}
    for P in ('A', 'B'):
        out['forecast_n30'][P] = {m: {'median': float(g[f'fc{P}_{m}'].median()),
                                      'win_rate_vs_const_velocity': float((g[f'fc{P}_{m}'] < g[f'fc{P}_const_velocity']).mean())}
                                  for m in ['hold_last', 'const_velocity'] + list(models)}
    (RES / 'main_evaluation/geolife_trained_summary.json').write_text(json.dumps(out, indent=2))
    LAB = {'linear': 'Linear', 'cubic': 'Cubic spline', 'pchip': 'PCHIP', 'frozen_three_shapes': 'Frozen synthetic (S3)',
           'frozen_three_shapes_S1': 'Frozen synthetic (S1)', 'geolife_scratch': 'GeoLife-trained (scratch)',
           'geolife_finetuned_from_three_shapes': 'Synthetic → GeoLife fine-tuned', 'hold_last': 'Hold last', 'const_velocity': 'Constant velocity'}
    I = out['interpolation'][str(E['n_obs'])]
    T = ['| Model (30 observations, held-out test users) | Interp. RMSE median [95% CI] (m) | mean (m) | win rate vs linear | win rate vs PCHIP | geo-mean ratio vs linear | Forecast A median (m) | Forecast B median (m) |',
         '|---|---|---|---|---|---|---|---|']
    for m in M:
        T.append(f"| {LAB[m]} | {I[m]['median']:.2f} [{I[m]['median_ci95_user_bootstrap'][0]:.2f}, {I[m]['median_ci95_user_bootstrap'][1]:.2f}] | {I[m]['mean']:.2f} | "
                 f"{I[m]['win_rate_vs_linear'] * 100:.1f}% | {I[m]['win_rate_vs_pchip'] * 100:.1f}% | {I[m]['geometric_mean_ratio_vs_linear']:.3f} | "
                 + (f"{out['forecast_n30']['A'][m]['median']:.1f} | {out['forecast_n30']['B'][m]['median']:.1f} |" if m in out['forecast_n30']['A'] else '— | — |'))
    for k in ('hold_last', 'const_velocity'):
        T.append(f"| {LAB[k]} (forecast baseline) | — | — | — | — | — | {out['forecast_n30']['A'][k]['median']:.1f} | {out['forecast_n30']['B'][k]['median']:.1f} |")
    T += ['', '| Paired (30 obs) | geometric-mean RMSE ratio [95% user-bootstrap CI] | win rate | Wilcoxon p |', '|---|---|---|---|']
    for k, v in out['paired_n30'].items():
        T.append(f"| {k} | {v['geometric_mean_ratio']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v['win_rate'] * 100:.1f}% | {v['wilcoxon_p']:.2g} |")
    T += ['', '| N obs | ' + ' | '.join(LAB[m] for m in M) + ' |', '|' + '---|' * (len(M) + 1)]
    for n in sorted(out['interpolation'], key=int):
        T.append(f'| {n} | ' + ' | '.join(f"{out['interpolation'][n][m]['median']:.2f}" for m in M) + ' |')
    (TAB / 'table5_synthetic_vs_geolife_trained.md').write_text('\n'.join(T) + '\n'); print('\n'.join(T))
    fig, ax = plt.subplots(figsize=(8, 5)); L = sorted(int(k) for k in out['interpolation'])
    cols = {'linear': '#2578bc', 'cubic': '#e0a030', 'pchip': '#3a9d5d', 'frozen_three_shapes': '#d1495b', 'frozen_three_shapes_S1': '#f08a8a',
            'geolife_scratch': '#222222', 'geolife_finetuned_from_three_shapes': '#8a5cc2'}
    for m in M:
        ax.plot(L, [out['interpolation'][str(n)][m]['median'] for n in L], marker='o', color=cols[m], label=LAB[m])
    ax.set_yscale('log'); ax.set_xticks(L); ax.axvline(30, color='gray', ls=':'); ax.set_xlabel('number of observations'); ax.set_ylabel('median interpolation RMSE (m)')
    ax.legend(frameon=False, fontsize=8); ax.set_title('Figure 9. Synthetic-pretrained vs GeoLife-trained Latent ODEs (held-out test users)', fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / 'fig9_synthetic_vs_geolife_trained.png', dpi=140); plt.close(fig)


if __name__ == '__main__':
    snap = frozen_snapshot()
    for s in args.stages:
        globals()[f'stage_{s}']()
    assert frozen_snapshot() == snap
    print(f'pre-existing weights unchanged ({len(snap)} files)')
