#!/usr/bin/env python3
"""Synthetic-to-real transfer: the FROZEN three-shape Latent ODE vs linear / cubic interpolation on real GeoLife GPS.

Stages (run in this order; each reads what the previous ones wrote):
  prepare    user-disjoint dev/test split and segment sampling from data/geolife/processed/segment_index.csv
  pilot      10 development segments, full per-trajectory outputs (a-priori default normalisation S1)
  normsel    normalisation strategy selection + sensitivity analyses on DEVELOPMENT users only
  main       held-out TEST users: interpolation (primary) and forecasting (secondary), 30 observations
  density    held-out TEST users: 10/20/30/40/50 observations (robustness)
  figures    figures and tables from the saved results
No model is trained or modified; weights are checksummed before and after every stage.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from glo import geolife as gl, geolife_evaluation as ge, inference as inf

torch.set_num_threads(1)
root = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('stages', nargs='+', choices=['prepare', 'pilot', 'normsel', 'main', 'density', 'figures', 'all'])
ap.add_argument('--config', default=str(root / 'configs/geolife.yaml'))
args = ap.parse_args()
stages = ['prepare', 'pilot', 'normsel', 'main', 'density', 'figures'] if 'all' in args.stages else args.stages
C = yaml.safe_load(open(args.config))
E = C['experiment']; S = C['splits']
RES = root / 'results/geolife'; FIG = RES / 'figures'; TAB = RES / 'tables'; PROC = root / 'data/geolife/processed'; SPL = root / 'data/geolife/splits'
for d in (RES, FIG, TAB, PROC, SPL, RES / 'pilot', RES / 'main_evaluation'):
    d.mkdir(parents=True, exist_ok=True)
SAVED = root / 'saved_models'
COL = {'latent_ode': '#d1495b', 'linear': '#2578bc', 'cubic': '#e0a030', 'pchip': '#3a9d5d', 'variable_nobs_experimental': '#8a5cc2',
       'hold_last': '#777777', 'const_velocity': '#2578bc'}
LABEL = {'latent_ode': 'Latent ODE (frozen, three-shape)', 'linear': 'Linear', 'cubic': 'Cubic spline', 'pchip': 'PCHIP',
         'variable_nobs_experimental': 'Latent ODE (variable-count model)', 'hold_last': 'Hold last', 'const_velocity': 'Constant velocity'}


def weights_snapshot():
    files = sorted((root / 'checkpoints').rglob('*.pt')) + sorted(SAVED.rglob('best_model.pt'))
    # Phase 11 outputs (GeoLife-trained models, possibly still being written) are not part of the frozen set
    return {str(p.relative_to(root)): inf.sha256_file(p) for p in files if 'geolife' not in p.relative_to(root).parts[1]}


def obs_seed(idx, n_obs):
    return int(E['observation_seed_base'] + 1_000_003 * n_obs + idx)


def load_split(name):
    meta = pd.read_csv(SPL / f'{name}_segments.csv')
    arr = np.load(PROC / f'{name}_segments.npz')
    return meta, arr


def task_of(meta, arr, i, n_obs=None):
    n_obs = n_obs or E['n_obs']
    return gl.build_task(arr['t_unix'][i], arr['lat'][i], arr['lon'][i], n_obs, obs_seed(i, n_obs), E['context_end'],
                         meta={'segment_id': meta.segment_id[i], 'user': str(meta.user[i]).zfill(3)})


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


# ======================================================================================================= prepare
def stage_prepare():
    idx = pd.read_csv(PROC / 'segment_index.csv', dtype={'user': str})
    elig = idx[idx.eligible].reset_index(drop=True)
    users = np.array(sorted(elig.user.unique()))
    rng = np.random.default_rng(S['user_split_seed'])
    perm = rng.permutation(users)
    n_dev = int(round(S['dev_user_fraction'] * len(users)))
    split_users = {'dev': sorted(perm[:n_dev]), 'test': sorted(perm[n_dev:])}
    assert not set(split_users['dev']) & set(split_users['test'])
    arch = gl.GeoLifeArchive(root / C['archive'])
    rs = np.random.default_rng(S['sampling_seed'])
    info = {'n_eligible_windows': int(len(elig)), 'n_users_with_eligible_windows': int(len(users))}
    for name, target in (('dev', S['n_dev_segments']), ('test', S['n_test_segments'])):
        pool = elig[elig.user.isin(split_users[name])]
        picked = []
        for u, g in pool.groupby('user', sort=True):
            k = min(S['max_segments_per_user'], len(g))
            picked.append(g.iloc[np.sort(rs.choice(len(g), k, replace=False))])
        picked = pd.concat(picked)
        if len(picked) > target:
            picked = picked.iloc[np.sort(rs.choice(len(picked), target, replace=False))]
        picked = picked.sample(frac=1.0, random_state=S['sampling_seed']).reset_index(drop=True)   # random order: pilot = first 10
        picked.insert(0, 'segment_id', [f'{name}_{i:04d}' for i in range(len(picked))])
        T = np.zeros((len(picked), 200), np.int64); LA = np.zeros((len(picked), 200)); LO = np.zeros((len(picked), 200))
        cache_lab, modes = {}, []
        for i, w in picked.iterrows():
            t, la, lo = arch.load_clean(f'Geolife Trajectories 1.3/Data/{w.user}/Trajectory/{w.file}')
            T[i], LA[i], LO[i] = t[w.start:w.stop], la[w.start:w.stop], lo[w.start:w.stop]
            assert T[i][0] == w.t0_unix and T[i][-1] == w.t1_unix
            if w.user not in cache_lab:
                cache_lab[w.user] = arch.labels(w.user)
            modes.append(gl.label_for_window(cache_lab[w.user], int(T[i][0]), int(T[i][-1])))
        picked['transport_mode_label'] = modes
        picked.to_csv(SPL / f'{name}_segments.csv', index=False)
        np.savez_compressed(PROC / f'{name}_segments.npz', t_unix=T, lat=LA, lon=LO)
        info[name] = {'n_users_in_split': len(split_users[name]), 'n_users_sampled': int(picked.user.nunique()),
                      'n_eligible_windows_available': int(len(pool)), 'n_segments': int(len(picked)),
                      'n_source_files': int(picked.groupby(['user', 'file']).ngroups),
                      'n_with_transport_label': int(sum(m is not None for m in modes)),
                      'transport_labels': pd.Series([m for m in modes if m]).value_counts().to_dict(), 'users': split_users[name]}
        print(name, {k: v for k, v in info[name].items() if k != 'users'})
    jdump(info, RES / 'splits_summary.json')


# ========================================================================================================= pilot
def save_segment_outputs(d, task, preds_i, fc_preds, fc_idx, metrics, title):
    d.mkdir(parents=True, exist_ok=True)
    o, q = task.obs_idx, task.interp_idx

    def wcsv(path, idx, xy, extra=None):
        df = pd.DataFrame({'t_model': task.t_model[idx], 't_elapsed_s': task.t_sec[idx], 'x_m': xy[:, 0], 'y_m': xy[:, 1]})
        if extra is not None:
            df['region'] = extra
        df.to_csv(path, index=False, float_format='%.4f')
    wcsv(d / 'observations.csv', o, task.xy_m[o])
    gt_idx = np.concatenate([q, fc_idx]); wcsv(d / 'withheld_ground_truth.csv', gt_idx, task.xy_m[gt_idx],
                                               ['interpolation'] * len(q) + ['forecast'] * len(fc_idx))
    lode_all = np.concatenate([preds_i['latent_ode'], fc_preds['latent_ode']])
    wcsv(d / 'latent_ode_predictions.csv', gt_idx, lode_all, ['interpolation'] * len(q) + ['forecast'] * len(fc_idx))
    wcsv(d / 'linear_predictions.csv', q, preds_i['linear']); wcsv(d / 'cubic_predictions.csv', q, preds_i['cubic'])
    jdump(metrics, d / 'metrics.json')
    plot_comparison(d / 'interpolation_comparison.png', task, preds_i, fc_preds, fc_idx, title)
    for k, nm in ((0, 'x'), (1, 'y')):
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(task.t_sec / 60, task.xy_m[:, k], color='#bbbbbb', lw=3, label='recorded GPS (all records)', zorder=1)
        ax.scatter(task.t_sec[q] / 60, task.xy_m[q, k], s=6, color='k', alpha=.5, label='withheld (targets)', zorder=2)
        for m in ('linear', 'cubic', 'latent_ode'):
            ax.plot(task.t_sec[q] / 60, preds_i[m][:, k], color=COL[m], lw=1.3, label=LABEL[m], zorder=3)
        ax.plot(task.t_sec[fc_idx] / 60, fc_preds['latent_ode'][:, k], color=COL['latent_ode'], lw=1.3, ls='--', label='Latent ODE forecast')
        ax.scatter(task.t_sec[o] / 60, task.xy_m[o, k], s=28, facecolor='w', edgecolor='k', zorder=4, label='30 observations')
        ax.axvline(task.t_sec[task.t_model <= 7].max() / 60, color='gray', ls=':', lw=1)
        ax.set_xlabel('elapsed time (min)'); ax.set_ylabel(f'{nm} (m)'); ax.set_title(f'{title}: {nm} vs time', fontsize=9)
        ax.legend(frameon=False, fontsize=7, ncol=3); fig.tight_layout(); fig.savefig(d / f'{nm}_vs_time.png', dpi=130); plt.close(fig)


def plot_comparison(path, task, preds_i, fc_preds, fc_idx, title, ax=None, legend=True):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 6.5))
    o, q = task.obs_idx, task.interp_idx
    ax.plot(task.xy_m[:, 0], task.xy_m[:, 1], color='#cccccc', lw=3, zorder=1, label='recorded GPS')
    ax.scatter(task.xy_m[q, 0], task.xy_m[q, 1], s=5, color='k', alpha=.45, zorder=2, label='withheld positions')
    for m in ('linear', 'cubic', 'latent_ode'):
        ax.plot(preds_i[m][:, 0], preds_i[m][:, 1], color=COL[m], lw=1.3, zorder=3,
                label=f"{LABEL[m]} (RMSE {ge.error_metrics(preds_i[m], task.xy_m[q])['rmse_xy']:.1f} m)")
    if fc_preds is not None and len(fc_idx):
        ax.plot(fc_preds['latent_ode'][:, 0], fc_preds['latent_ode'][:, 1], color=COL['latent_ode'], lw=1, ls='--', zorder=3, label='Latent ODE forecast')
    ax.scatter(task.xy_m[o, 0], task.xy_m[o, 1], s=30, facecolor='w', edgecolor='k', zorder=4, label=f'{len(o)} observations')
    ax.plot(*task.xy_m[0], 'k^', ms=8, zorder=5)
    ax.set_aspect('equal', 'datalim'); ax.set_xlabel('east (m)'); ax.set_ylabel('north (m)'); ax.set_title(title, fontsize=9)
    if legend:
        ax.legend(frameon=False, fontsize=7)
    if own:
        fig = ax.figure; fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def stage_pilot():
    meta, arr = load_split('dev')
    lm = inf.load_packaged_model(E['model'], SAVED)
    strat = gl.DEFAULT_STRATEGY
    rows = []
    for i in range(E['n_pilot']):
        task = task_of(meta, arr, i)
        ri, pi = ge.evaluate_interpolation(task, lm, strat, return_preds=True)
        rf, pf, fidx = ge.evaluate_forecast(task, lm, strat, 'A', return_preds=True)
        d = ge.descriptors(task)
        m = {'segment_id': meta.segment_id[i], 'user': str(meta.user[i]).zfill(3), 'source_file': meta.file[i],
             'records': [int(meta.start[i]), int(meta.stop[i])], 'duration_s': float(meta.duration_s[i]),
             'normalisation_strategy': strat.__dict__, 'n_observations': int(len(task.obs_idx)),
             'n_interpolation_targets': int(len(task.interp_idx)), 'n_context_tail_not_scored': int(len(task.ctx_tail_idx)),
             'interpolation_metrics_m': ri, 'forecast_metrics_m_protocol_A': rf, 'descriptors': d,
             'shape_category': ge.shape_category(d), 'speed_category': ge.speed_category(d)}
        title = f"pilot {i + 1}: user {m['user']}, {meta.duration_s[i] / 60:.1f} min, {d['context_path_m']:.0f} m context path"
        save_segment_outputs(RES / 'pilot' / f'trajectory_{i + 1:03d}', task, pi, pf, fidx, m, title)
        rows.append({'trajectory': f'trajectory_{i + 1:03d}', 'segment_id': m['segment_id'], 'user': m['user'], 'category': m['shape_category'],
                     'speed_mps': d['mean_speed_mps'], 'n_targets': len(task.interp_idx),
                     **{f'{k}_rmse_m': ri[k]['rmse_xy'] for k in ge.INTERP_METHODS},
                     'lode_forecast_rmse_m': rf['latent_ode']['rmse_xy'], 'const_velocity_forecast_rmse_m': rf['const_velocity']['rmse_xy']})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(RES / 'pilot/pilot_summary.csv', index=False, float_format='%.3f')
    (RES / 'pilot/pilot_summary.md').write_text(df.to_markdown(index=False, floatfmt='.2f'))


# ======================================================================================================= normsel
def synthetic_max_radius_median():
    d = np.load(root / 'data/processed/three_shapes/train.npz'); ctx = d['time'] <= 7
    c = d['xy'][:, ctx]; r = np.linalg.norm(c - c.mean(1, keepdims=True), axis=-1)
    return float(np.median(np.sqrt((r ** 2).mean(1)))), float(np.median(r.max(1))), [float(v) for v in np.percentile(np.sqrt((r ** 2).mean(1)), [5, 95])]


def interp_median(meta, arr, lm, strat, n=None):
    vals = []
    for i in range(len(meta) if n is None else n):
        vals.append(ge.evaluate_interpolation(task_of(meta, arr, i), lm, strat, methods=('latent_ode',))['latent_ode']['rmse_xy'])
    return np.array(vals)


def stage_normsel():
    meta, arr = load_split('dev')
    lm = inf.load_packaged_model(E['model'], SAVED)
    rms_med, max_med, rms_p = synthetic_max_radius_median()
    cands = [gl.DEFAULT_STRATEGY,
             gl.NormStrategy('S2_obsmean_rms_p5', 'obs_mean', 'obs_rms', round(rms_p[0], 3)),
             gl.NormStrategy('S3_obsmean_rms_p95', 'obs_mean', 'obs_rms', round(rms_p[1], 3)),
             gl.NormStrategy('S4_firstobs_rms_1.0', 'first_obs', 'obs_rms', 1.0),
             gl.NormStrategy('S5_obsmean_max', 'obs_mean', 'obs_max', round(max_med, 3))]
    lin = np.array([ge.evaluate_interpolation(task_of(meta, arr, i), lm, cands[0], methods=('linear',))['linear']['rmse_xy'] for i in range(len(meta))])
    out = {'split': 'dev', 'n_segments': int(len(meta)), 'criterion': 'lowest MEDIAN per-segment interpolation RMSE (m) of the frozen Latent ODE on development users (declared before running)',
           'synthetic_reference': {'context_rms_radius_median': rms_med, 'context_rms_radius_p5_p95': rms_p, 'context_max_radius_median': max_med},
           'linear_median_rmse_m': float(np.median(lin)), 'strategies': {}}
    for s in cands:
        v = interp_median(meta, arr, lm, s)
        out['strategies'][s.name] = {'config': s.__dict__, 'median_rmse_m': float(np.median(v)), 'mean_rmse_m': float(np.mean(v)),
                                     'win_rate_vs_linear': float(np.mean(v < lin))}
        print(s.name, out['strategies'][s.name], flush=True)
    best = min(out['strategies'], key=lambda k: out['strategies'][k]['median_rmse_m'])
    out['selected'] = best
    sel = next(s for s in cands if s.name == best)
    # ---- sensitivity (dev only) with the selected strategy: what an equivariant method would leave unchanged
    sens = {'rotation_deg': {}, 'translation_model_units': {}, 'target_scale': {}}
    base = interp_median(meta, arr, lm, sel)
    for deg in (0, 45, 90, 180, 270):
        v = base if deg == 0 else interp_median(meta, arr, lm, gl.NormStrategy(f'rot{deg}', sel.center, sel.spread, sel.target, np.radians(deg)))
        sens['rotation_deg'][str(deg)] = {'median_rmse_m': float(np.median(v)), 'median_abs_change_vs_0_pct': float(np.median(np.abs(v / base - 1)) * 100)}
    for off in (0.0, 0.5, 1.0, 2.0):
        v = base if off == 0 else interp_median(meta, arr, lm, gl.NormStrategy(f'off{off}', sel.center, sel.spread, sel.target, 0.0, (off, 0.0)))
        sens['translation_model_units'][str(off)] = {'median_rmse_m': float(np.median(v)), 'median_abs_change_vs_0_pct': float(np.median(np.abs(v / base - 1)) * 100)}
    for f in (0.25, 0.5, 1.0, 2.0, 4.0):
        v = base if f == 1.0 else interp_median(meta, arr, lm, gl.NormStrategy(f'scale{f}', sel.center, sel.spread, sel.target * f))
        sens['target_scale'][str(f)] = {'target': sel.target * f, 'median_rmse_m': float(np.median(v)), 'median_abs_change_vs_1_pct': float(np.median(np.abs(v / base - 1)) * 100)}
    dl = np.array([ge.descriptors(task_of(meta, arr, i))['context_path_m'] for i in range(len(meta))])
    edges = np.percentile(dl, [0, 33.33, 66.67, 100])
    sens['path_length_terciles'] = {}
    for a, b in zip(edges[:-1], edges[1:]):
        k = (dl >= a) & (dl <= b)
        sens['path_length_terciles'][f'{a:.0f}-{b:.0f} m'] = {'n': int(k.sum()), 'latent_ode_median_rmse_m': float(np.median(base[k])),
                                                             'linear_median_rmse_m': float(np.median(lin[k])),
                                                             'latent_ode_median_relative_rmse': float(np.median(base[k] / dl[k]))}
    out['sensitivity_selected_strategy'] = sens
    jdump(out, TAB / 'normalization_strategies.json')
    print('selected', best); print(json.dumps(sens, indent=1))


def selected_strategy():
    j = json.loads((TAB / 'normalization_strategies.json').read_text())
    c = j['strategies'][j['selected']]['config']
    return gl.NormStrategy(c['name'], c['center'], c['spread'], c['target'], c['rotate'], tuple(c['offset']))


# ========================================================================================================== main
def stage_main():
    meta, arr = load_split('test')
    lm = inf.load_packaged_model(E['model'], SAVED)
    strat = selected_strategy()
    rows, gaps = [], {'segment': [], 'gap_s': [], 'pos': [], **{m: [] for m in ge.INTERP_METHODS}}
    t0 = time.time()
    for i in range(len(meta)):
        task = task_of(meta, arr, i)
        ri, pi = ge.evaluate_interpolation(task, lm, strat, return_preds=True)
        fa = ge.evaluate_forecast(task, lm, strat, 'A'); fb = ge.evaluate_forecast(task, lm, strat, 'B')
        d = ge.descriptors(task)
        r = {'segment_id': meta.segment_id[i], 'user': str(meta.user[i]).zfill(3), 'file': meta.file[i], 'mode': meta.transport_mode_label[i],
             'duration_s': meta.duration_s[i], 'n_interp_targets': ri['latent_ode']['n'], 'n_ctx_tail': len(task.ctx_tail_idx),
             'scale_m_per_model_unit': ri['scale_m_per_model_unit'], 'obs_rms_spread_m': ri['obs_rms_spread_m'], **d,
             'shape_category': ge.shape_category(d), 'speed_category': ge.speed_category(d)}
        for m in ge.INTERP_METHODS:
            for k in ('rmse_xy', 'mae_xy', 'ade', 'median_error', 'p90_error', 'max_error', 'inference_s'):
                r[f'interp_{m}_{k}'] = ri[m][k]
        for P, f in (('A', fa), ('B', fb)):
            r[f'fc{P}_n'] = f['n_targets']; r[f'fc{P}_horizon_s'] = f['horizon_s']
            for m in ge.FORECAST_METHODS:
                for k in ('rmse_xy', 'ade', 'fde'):
                    r[f'fc{P}_{m}_{k}'] = f[m][k]
        rows.append(r)
        g, pos = ge.interpolation_target_gaps(task); tq = task.xy_m[task.interp_idx]
        gaps['segment'] += [i] * len(g); gaps['gap_s'] += list(g); gaps['pos'] += list(pos)
        for m in ge.INTERP_METHODS:
            gaps[m] += list(np.linalg.norm(pi[m] - tq, axis=1))
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(meta)}  {time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows); df.to_csv(RES / 'main_evaluation/per_segment_metrics.csv', index=False)
    np.savez_compressed(RES / 'main_evaluation/per_target_errors.npz', **{k: np.asarray(v) for k, v in gaps.items()})
    summarize_main(df)


def paired(df, a, b, metric='rmse_xy', prefix='interp'):
    from scipy.stats import wilcoxon
    x = df[f'{prefix}_{a}_{metric}'].to_numpy(); y = df[f'{prefix}_{b}_{metric}'].to_numpy()
    lr = np.log(x / y)
    ci = ge.cluster_bootstrap(lr, df.user.to_numpy(), np.mean)
    return {'comparison': f'{a} vs {b}', 'n': int(len(x)), 'geometric_mean_ratio': float(np.exp(lr.mean())),
            'geometric_mean_ratio_95ci_user_cluster_bootstrap': [float(np.exp(ci[0])), float(np.exp(ci[1]))],
            'median_ratio': float(np.median(x / y)), 'win_rate_a_lower': float(np.mean(x < y)),
            'wilcoxon_p': float(wilcoxon(x, y).pvalue), 'median_difference_m': float(np.median(x - y))}


def agg(df, cols):
    out = {}
    for c in cols:
        v = df[c].to_numpy()
        out[c] = {'mean': float(v.mean()), 'median': float(np.median(v)), 'p90': float(np.percentile(v, 90)),
                  'mean_95ci_user_cluster_bootstrap': list(ge.cluster_bootstrap(v, df.user.to_numpy(), np.mean)),
                  'median_95ci_user_cluster_bootstrap': list(ge.cluster_bootstrap(v, df.user.to_numpy(), np.median))}
    return out


def summarize_main(df):
    M = ge.INTERP_METHODS
    summ = {'n_segments': int(len(df)), 'n_users': int(df.user.nunique()), 'n_source_files': int(df.groupby(['user', 'file']).ngroups),
            'normalisation_strategy': selected_strategy().__dict__,
            'interpolation_targets_total': int(df.n_interp_targets.sum()), 'context_tail_records_not_scored_total': int(df.n_ctx_tail.sum()),
            'interpolation': {m: agg(df, [f'interp_{m}_{k}' for k in ('rmse_xy', 'mae_xy', 'ade', 'median_error', 'p90_error', 'max_error')]) for m in M},
            'interpolation_relative_rmse_model_units': {m: agg(df.assign(rel=df[f'interp_{m}_rmse_xy'] / df.scale_m_per_model_unit), ['rel'])['rel'] for m in M},
            'inference_time_ms_per_segment': {m: {'median': float(df[f'interp_{m}_inference_s'].median() * 1e3),
                                                  'mean': float(df[f'interp_{m}_inference_s'].mean() * 1e3)} for m in M},
            'paired_tests': {f'latent_ode_vs_{b}': paired(df, 'latent_ode', b) for b in ('linear', 'cubic', 'pchip')}
            | {'cubic_vs_linear': paired(df, 'cubic', 'linear'), 'pchip_vs_linear': paired(df, 'pchip', 'linear')},
            'note_multiple_comparisons': 'five paired Wilcoxon tests; Holm-adjusted p-values in paired_tests_holm',
            'forecast': {}}
    ps = sorted(summ['paired_tests'].items(), key=lambda kv: kv[1]['wilcoxon_p'])
    summ['paired_tests_holm'] = {k: min(1.0, v['wilcoxon_p'] * (len(ps) - j)) for j, (k, v) in enumerate(ps)}
    for P in ('A', 'B'):
        summ['forecast'][P] = {'protocol': 'segment duration mapped to [0,10]; targets 7<t<=10 (end time known)' if P == 'A' else
                               'last observation mapped to t=7; targets = all later records with t<=10 (end time NOT used)',
                               'n_targets_total': int(df[f'fc{P}_n'].sum()), 'median_horizon_s': float(df[f'fc{P}_horizon_s'].median()),
                               **{m: agg(df, [f'fc{P}_{m}_{k}' for k in ('rmse_xy', 'ade', 'fde')]) for m in ge.FORECAST_METHODS},
                               'paired': {f'latent_ode_vs_{b}': paired(df, 'latent_ode', b, prefix=f'fc{P}') for b in ('hold_last', 'const_velocity')}}
    strata = {}
    for col in ('shape_category', 'speed_category', 'mode'):
        strata[col] = {}
        for k, g in df.groupby(col):
            if len(g) < 10:
                continue
            strata[col][str(k)] = {'n': int(len(g)), 'n_users': int(g.user.nunique()),
                                   **{m: float(g[f'interp_{m}_rmse_xy'].median()) for m in M},
                                   'latent_ode_win_rate_vs_linear': float((g.interp_latent_ode_rmse_xy < g.interp_linear_rmse_xy).mean()),
                                   'geometric_mean_ratio_lode_linear': float(np.exp(np.log(g.interp_latent_ode_rmse_xy / g.interp_linear_rmse_xy).mean()))}
    e = np.percentile(df.mean_obs_gap_s, [0, 100 / 3, 200 / 3, 100])
    q = pd.cut(df.mean_obs_gap_s, e, labels=[f'{a:.1f}-{b:.1f} s' for a, b in zip(e[:-1], e[1:])], include_lowest=True)
    strata['mean_observation_gap_tercile'] = {str(k): {'n': int(len(g)), **{m: float(g[f'interp_{m}_rmse_xy'].median()) for m in M},
                                                      'latent_ode_win_rate_vs_linear': float((g.interp_latent_ode_rmse_xy < g.interp_linear_rmse_xy).mean())}
                                              for k, g in df.groupby(q, observed=True)}
    summ['strata_median_interp_rmse_m'] = strata
    jdump(summ, RES / 'main_evaluation/summary.json')
    for m in M:
        a = summ['interpolation'][m]
        print(f"{m:11s} interp RMSE median {a[f'interp_{m}_rmse_xy']['median']:.2f} m  mean {a[f'interp_{m}_rmse_xy']['mean']:.2f} m  "
              f"ADE median {a[f'interp_{m}_ade']['median']:.2f} m")
    print(json.dumps(summ['paired_tests'], indent=1))


# ======================================================================================================= density
def stage_density():
    meta, arr = load_split('test')
    strat = selected_strategy()
    models = {'latent_ode': inf.load_packaged_model(E['model'], SAVED),
              'variable_nobs_experimental': inf.load_packaged_model('variable_nobs_experimental', SAVED)}
    rows = []
    for n in E['density_levels']:
        for i in range(len(meta)):
            task = task_of(meta, arr, i, n)
            r = ge.evaluate_interpolation(task, models['latent_ode'], strat, methods=('latent_ode', 'linear', 'cubic', 'pchip'))
            rv = ge.evaluate_interpolation(task, models['variable_nobs_experimental'], strat, methods=('latent_ode',))
            rows.append({'n_obs': n, 'segment_id': meta.segment_id[i], 'user': str(meta.user[i]).zfill(3), 'n_targets': r['linear']['n'],
                         **{f'{m}_rmse_xy': r[m]['rmse_xy'] for m in ('latent_ode', 'linear', 'cubic', 'pchip')},
                         'variable_nobs_experimental_rmse_xy': rv['latent_ode']['rmse_xy'],
                         **{f'{m}_ade': r[m]['ade'] for m in ('latent_ode', 'linear', 'cubic', 'pchip')},
                         'variable_nobs_experimental_ade': rv['latent_ode']['ade']})
        print('n_obs', n, 'done', flush=True)
    df = pd.DataFrame(rows); df.to_csv(RES / 'main_evaluation/density_per_segment.csv', index=False)
    M = ('latent_ode', 'variable_nobs_experimental', 'linear', 'cubic', 'pchip')
    out = {'n_segments': int(meta.shape[0]), 'normalisation_strategy': strat.__dict__,
           'note': 'the same test segments at every level; observation subsets drawn independently per level (t=0 always observed); '
                   'targets = withheld context records between first and last observation, so they differ between levels',
           'levels': {}}
    for n, g in df.groupby('n_obs'):
        out['levels'][str(n)] = {m: {'median_rmse_m': float(g[f'{m}_rmse_xy'].median()), 'mean_rmse_m': float(g[f'{m}_rmse_xy'].mean()),
                                     'median_ci95': list(ge.cluster_bootstrap(g[f'{m}_rmse_xy'].to_numpy(), g.user.to_numpy(), np.median)),
                                     'median_ade_m': float(g[f'{m}_ade'].median()),
                                     'win_rate_vs_linear': float((g[f'{m}_rmse_xy'] < g.linear_rmse_xy).mean())} for m in M}
        out['levels'][str(n)]['median_targets'] = float(g.n_targets.median())
    jdump(out, RES / 'main_evaluation/density_summary.json')
    for n in out['levels']:
        print(n, {m: round(out['levels'][n][m]['median_rmse_m'], 2) for m in M})


# ======================================================================================================= figures
def stage_figures():
    from scipy.stats import spearmanr
    df = pd.read_csv(RES / 'main_evaluation/per_segment_metrics.csv', dtype={'user': str})
    summarize_main(df)
    summ = json.loads((RES / 'main_evaluation/summary.json').read_text())
    meta, arr = load_split('test'); lm = inf.load_packaged_model(E['model'], SAVED); strat = selected_strategy()
    M = ge.INTERP_METHODS

    def preds_for(i):
        task = task_of(meta, arr, i)
        _, pi = ge.evaluate_interpolation(task, lm, strat, return_preds=True)
        _, pf, fidx = ge.evaluate_forecast(task, lm, strat, 'A', return_preds=True)
        return task, pi, pf, fidx

    rng = np.random.default_rng(11)
    pick = rng.choice(len(meta), 3, replace=False)                 # random, not chosen by result
    # Figure 2: GPS record and the 30 observations
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4))
    for a, i in zip(axs, pick):
        task = task_of(meta, arr, i); o = task.obs_idx
        a.plot(task.xy_m[:, 0], task.xy_m[:, 1], color='#bbbbbb', lw=3, label='recorded GPS (200 records)')
        sc = a.scatter(task.xy_m[:, 0], task.xy_m[:, 1], c=task.t_model, s=6, cmap='viridis', zorder=2)
        a.scatter(task.xy_m[o, 0], task.xy_m[o, 1], s=40, facecolor='none', edgecolor='r', lw=1.4, zorder=3, label='30 observations (t ≤ 7)')
        a.set_aspect('equal', 'datalim'); a.set_title(f'{meta.segment_id[i]} (user {str(meta.user[i]).zfill(3)})', fontsize=9); a.legend(frameon=False, fontsize=7)
        plt.colorbar(sc, ax=a, fraction=0.04).set_label('model time t', fontsize=8)
    fig.suptitle('Figure 2. Randomly drawn held-out GPS segments and their 30 observed positions (local metres)')
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / 'fig2_gps_with_observations.png', dpi=140); plt.close(fig)
    # Figure 3: methods on the same random segments
    fig, axs = plt.subplots(1, 3, figsize=(17, 6))
    for a, i in zip(axs, pick):
        task, pi, pf, fidx = preds_for(i)
        plot_comparison(None, task, pi, None, [], f'{meta.segment_id[i]} ({df.shape_category[i]}, {df.mean_speed_mps[i]:.1f} m/s)', ax=a)
    fig.suptitle('Figure 3. Linear vs cubic spline vs frozen Latent ODE interpolation on randomly drawn held-out segments')
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / 'fig3_method_comparison.png', dpi=140); plt.close(fig)
    # Figure 4: best / median / worst by relative Latent ODE error
    rel = (df.interp_latent_ode_rmse_xy / df.scale_m_per_model_unit).to_numpy(); order = np.argsort(rel)
    sel = {'best': order[0], 'median': order[len(order) // 2], 'worst': order[-1]}
    fig, axs = plt.subplots(1, 3, figsize=(17, 6))
    for a, (lab, i) in zip(axs, sel.items()):
        task, pi, pf, fidx = preds_for(i)
        plot_comparison(None, task, pi, pf, fidx, f'{lab}: {meta.segment_id[i]}, LODE RMSE {df.interp_latent_ode_rmse_xy[i]:.1f} m '
                                                  f'({rel[i]:.3f} model units)', ax=a)
    fig.suptitle('Figure 4. Best, median and worst frozen Latent ODE interpolation (ranked by RMSE in model units = metres / per-segment scale)')
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / 'fig4_best_median_worst.png', dpi=140); plt.close(fig)
    # Figure 5: error distributions
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    bins = np.logspace(np.log10(max(df[[f'interp_{m}_rmse_xy' for m in M]].min().min(), 0.1)), np.log10(df[[f'interp_{m}_rmse_xy' for m in M]].max().max()), 50)
    for m in M:
        v = np.sort(df[f'interp_{m}_rmse_xy'])
        ax[0].plot(v, np.arange(1, len(v) + 1) / len(v), color=COL[m], label=f'{LABEL[m]} (median {np.median(v):.1f} m)')
    ax[0].set_xscale('log'); ax[0].set_xlabel('per-segment interpolation RMSE (m)'); ax[0].set_ylabel('cumulative fraction of segments'); ax[0].legend(frameon=False, fontsize=8)
    lr = np.log10(df.interp_latent_ode_rmse_xy / df.interp_linear_rmse_xy)
    ax[1].hist(lr, bins=60, color=COL['latent_ode']); ax[1].axvline(0, color='k', lw=1)
    ax[1].set_xlabel('log10(Latent ODE RMSE / linear RMSE)  (< 0: Latent ODE better)'); ax[1].set_ylabel('segments')
    ax[1].set_title(f"Latent ODE better on {np.mean(lr < 0) * 100:.1f}% of {len(df)} segments", fontsize=9)
    fig.suptitle(f'Figure 5. Interpolation error distribution, held-out test users ({len(df)} segments, {df.user.nunique()} users, 30 observations)')
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(FIG / 'fig5_error_distribution.png', dpi=140); plt.close(fig)
    # Figure 6: observation density
    den = json.loads((RES / 'main_evaluation/density_summary.json').read_text())
    L = sorted(int(k) for k in den['levels']); fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in ('latent_ode', 'variable_nobs_experimental', 'linear', 'cubic', 'pchip'):
        med = [den['levels'][str(n)][m]['median_rmse_m'] for n in L]
        lo_ = [den['levels'][str(n)][m]['median_ci95'][0] for n in L]; hi_ = [den['levels'][str(n)][m]['median_ci95'][1] for n in L]
        ax.plot(L, med, marker='o', color=COL[m], label=LABEL[m]); ax.fill_between(L, lo_, hi_, color=COL[m], alpha=.15)
    ax.axvline(30, color='gray', ls=':'); ax.set_yscale('log'); ax.set_xticks(L); ax.set_xlabel('number of observations'); ax.set_ylabel('median interpolation RMSE (m)')
    ax.legend(frameon=False, fontsize=8); ax.set_title('Figure 6. Interpolation error vs observation count (held-out test; 95% user-bootstrap CI)', fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / 'fig6_error_vs_observation_count.png', dpi=140); plt.close(fig)
    # Figure 7: strata
    st = summ['strata_median_interp_rmse_m']
    groups = [('shape_category', ['nearly_straight', 'curved', 'frequent_turning']),
              ('speed_category', ['slow_<2.5mps', 'medium_2.5-10mps', 'fast_>10mps']),
              ('mean_observation_gap_tercile', list(st['mean_observation_gap_tercile'].keys()))]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for a, (g, keys) in zip(axs, groups):
        keys = [k for k in keys if k in st[g]]; x = np.arange(len(keys)); w = 0.2
        for j, m in enumerate(M):
            a.bar(x + (j - 1.5) * w, [st[g][k][m] for k in keys], w, color=COL[m], label=LABEL[m])
        a.set_xticks(x); a.set_xticklabels([f"{k}\n(n={st[g][k]['n']})" for k in keys], fontsize=8); a.set_yscale('log')
        a.set_ylabel('median interpolation RMSE (m)'); a.set_title(g.replace('_', ' ') + (' (mean gap between observations)' if 'gap' in g else ''), fontsize=10)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Figure 7. Interpolation error by trajectory characteristic (held-out test; descriptors from the GPS record, documented thresholds)')
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(FIG / 'fig7_characteristics.png', dpi=140); plt.close(fig)
    # Figure 8: error vs gap
    pt = np.load(RES / 'main_evaluation/per_target_errors.npz')
    edges = np.unique(np.percentile(pt['gap_s'], np.linspace(0, 100, 11)))
    k = np.clip(np.digitize(pt['gap_s'], edges[1:-1]), 0, len(edges) - 2)
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in M:
        med = [np.median(pt[m][k == b]) for b in range(len(edges) - 1)]
        ctr = [np.median(pt['gap_s'][k == b]) for b in range(len(edges) - 1)]
        ax.plot(ctr, med, marker='o', color=COL[m], label=LABEL[m])
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('length of the observation gap containing the target (s; decile bins)')
    ax.set_ylabel('median positional error (m)'); ax.legend(frameon=False, fontsize=8)
    rho = {m: float(spearmanr(pt['gap_s'], pt[m]).statistic) for m in M}
    ax.set_title('Figure 8. Error vs observation gap (' + ', '.join(f'{m} ρ={v:.2f}' for m, v in rho.items()) + ')', fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / 'fig8_error_vs_gap.png', dpi=140); plt.close(fig)
    write_tables(df, summ, den, rho)


def write_tables(df, summ, den, rho):
    ds = json.loads((RES / 'dataset_statistics.json').read_text()); sp = json.loads((RES / 'splits_summary.json').read_text())
    ns = json.loads((TAB / 'normalization_strategies.json').read_text())
    T1 = ['| Quantity | Value |', '|---|---|',
          f"| Users / trajectory files | {ds['n_users']} / {ds['n_trajectory_files']:,} |",
          f"| Data lines / clean records | {ds['n_data_lines']:,} / {ds['n_clean_records']:,} |",
          f"| Malformed lines / invalid coordinates | {ds['n_malformed_lines']} / {ds['n_invalid_coordinates']} |",
          f"| Duplicate timestamps (dropped) / backward timestamps | {ds['n_duplicate_timestamps']:,} / {ds['n_backward_timestamps']} |",
          f"| Sampling interval p5 / p50 / p95 / p99 (s) | {ds['sampling_interval_s_percentiles_all_steps']['p5']} / {ds['sampling_interval_s_percentiles_all_steps']['p50']} / {ds['sampling_interval_s_percentiles_all_steps']['p95']} / {ds['sampling_interval_s_percentiles_all_steps']['p99']} |",
          f"| Steps with interval ≤ 5 s / > 60 s | {ds['fraction_steps_dt_le_5s'] * 100:.1f}% / {ds['fraction_steps_dt_gt_max_gap'] * 100:.2f}% |",
          f"| Records per file median (p5–p95) | {ds['records_per_file']['p50']:.0f} ({ds['records_per_file']['p5']:.0f}–{ds['records_per_file']['p95']:.0f}) |",
          f"| Candidate / eligible 200-record segments | {ds['n_candidate_windows']:,} / {ds['n_eligible_windows']:,} |",
          f"| Users with eligible segments | {ds['n_users_with_eligible_windows']} |",
          f"| Eligible segment duration median (p5–p95) | {ds['eligible_window_duration_s']['p50'] / 60:.1f} min ({ds['eligible_window_duration_s']['p5'] / 60:.1f}–{ds['eligible_window_duration_s']['p95'] / 60:.1f}) |",
          f"| Eligible segment path length median (p5–p95) | {ds['eligible_window_path_length_m']['p50']:.0f} m ({ds['eligible_window_path_length_m']['p5']:.0f}–{ds['eligible_window_path_length_m']['p95']:.0f}) |",
          f"| Dev split: users / segments | {sp['dev']['n_users_sampled']} / {sp['dev']['n_segments']} |",
          f"| Test split: users / segments | {sp['test']['n_users_sampled']} / {sp['test']['n_segments']} |"]
    sc = C['segments']; st = summ['normalisation_strategy']
    T2 = ['| Setting | Value |', '|---|---|',
          '| Projection | spherical azimuthal equidistant at the segment\'s first record (always observed), radius = WGS-84 Gaussian radius at that latitude |',
          f"| Run breaks | gap > {sc['max_gap_s']:g} s or step speed > {sc['max_speed_mps']:g} m/s |",
          f"| Segment | {sc['window_points']} consecutive cleaned records, non-overlapping, extent ≥ {sc['min_extent_m']:g} m, ≥ {sc['min_context_points']} context and ≥ {sc['min_future_points']} future records |",
          '| Time (protocol A, primary) | t = 10 (t_real − t_start) / (t_end − t_start); context t ≤ 7, forecast 7 < t ≤ 10; irregular spacing kept |',
          '| Time (protocol B, forecasting) | t = 7 (t_real − t_start) / (t_last_obs − t_start); no knowledge of the segment end |',
          f"| Observations | {E['n_obs']}: the first record + {E['n_obs'] - 1} uniformly random other context records; seed = {E['observation_seed_base']} + 1000003·N + segment index |",
          '| Interpolation targets | all withheld context records between the first and last observation (recorded GPS positions) |',
          f"| Coordinate transform | {st['name']}: centre = {st['center']}, spread = {st['spread']}, mapped to {st['target']} model units; fitted on the observations only |",
          f"| Strategy selection | {ns['criterion']}; {ns['n_segments']} dev segments |",
          f"| Splits | users split 70/30 test/dev (seed {S['user_split_seed']}); ≤ {S['max_segments_per_user']} segments per user |",
          f"| Model | {E['model']} (frozen, posterior mean z0, RK4 step 0.1) via glo.inference.predict |"]
    M = ge.INTERP_METHODS; I = summ['interpolation']
    T3 = ['| Method | RMSE median [95% CI] (m) | RMSE mean (m) | MAE median (m) | ADE median (m) | median err. (m) | p90 err. (m) | max err. (m) | win rate vs linear | time / segment (ms) |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for m in M:
        r = I[m]; ci = r[f'interp_{m}_rmse_xy']['median_95ci_user_cluster_bootstrap']
        wr = '—' if m == 'linear' else f"{summ['paired_tests'].get(f'latent_ode_vs_linear' if m == 'latent_ode' else f'{m}_vs_linear')['win_rate_a_lower'] * 100:.1f}%"
        T3.append(f"| {LABEL[m]} | {r[f'interp_{m}_rmse_xy']['median']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {r[f'interp_{m}_rmse_xy']['mean']:.2f} | "
                  f"{r[f'interp_{m}_mae_xy']['median']:.2f} | {r[f'interp_{m}_ade']['median']:.2f} | {r[f'interp_{m}_median_error']['median']:.2f} | "
                  f"{r[f'interp_{m}_p90_error']['median']:.2f} | {r[f'interp_{m}_max_error']['median']:.2f} | {wr} | {summ['inference_time_ms_per_segment'][m]['median']:.2f} |")
    T3 += ['', '| Paired comparison (per-segment RMSE) | geometric-mean ratio [95% user-bootstrap CI] | median ratio | win rate (first lower) | Wilcoxon p (Holm) |', '|---|---|---|---|---|']
    for k, v in summ['paired_tests'].items():
        T3.append(f"| {v['comparison']} | {v['geometric_mean_ratio']:.3f} [{v['geometric_mean_ratio_95ci_user_cluster_bootstrap'][0]:.3f}, {v['geometric_mean_ratio_95ci_user_cluster_bootstrap'][1]:.3f}] | "
                  f"{v['median_ratio']:.3f} | {v['win_rate_a_lower'] * 100:.1f}% | {summ['paired_tests_holm'][k]:.2g} |")
    T3 += ['', '| Forecast (secondary) | protocol | targets | median horizon (s) | Latent ODE RMSE median (m) | hold-last (m) | constant velocity (m) | LODE win rate vs const. vel. |', '|---|---|---|---|---|---|---|---|']
    for P in ('A', 'B'):
        F = summ['forecast'][P]
        T3.append(f"| | {P} | {F['n_targets_total']:,} | {F['median_horizon_s']:.0f} | {F['latent_ode'][f'fc{P}_latent_ode_rmse_xy']['median']:.1f} | "
                  f"{F['hold_last'][f'fc{P}_hold_last_rmse_xy']['median']:.1f} | {F['const_velocity'][f'fc{P}_const_velocity_rmse_xy']['median']:.1f} | "
                  f"{F['paired']['latent_ode_vs_const_velocity']['win_rate_a_lower'] * 100:.1f}% |")
    DM = ('latent_ode', 'variable_nobs_experimental', 'linear', 'cubic', 'pchip')
    T4 = ['| N obs | median targets | ' + ' | '.join(LABEL[m] for m in DM) + ' | LODE win rate vs linear |', '|' + '---|' * (len(DM) + 3)]
    for n in sorted(den['levels'], key=int):
        L_ = den['levels'][n]
        T4.append(f"| {n} | {L_['median_targets']:.0f} | " + ' | '.join(f"{L_[m]['median_rmse_m']:.2f}" for m in DM) + f" | {L_['latent_ode']['win_rate_vs_linear'] * 100:.1f}% |")
    T4.append(''); T4.append('Median per-segment interpolation RMSE (m), held-out test users.')
    for nm, T in (('table1_dataset_statistics', T1), ('table2_preprocessing', T2), ('table3_interpolation_performance', T3), ('table4_observation_counts', T4)):
        (TAB / f'{nm}.md').write_text('\n'.join(T) + '\n')
    jdump({'spearman_error_vs_gap': rho}, TAB / 'gap_correlation.json')
    print('\n'.join(T3)); print('\n'.join(T4))


if __name__ == '__main__':
    snap0 = weights_snapshot()
    for s in stages:
        print(f'===== {s}', flush=True); t = time.time()
        globals()[f'stage_{s}']()
        print(f'===== {s} done in {time.time() - t:.0f}s', flush=True)
    assert weights_snapshot() == snap0, 'a checkpoint changed!'
    print(f'weights unchanged ({len(snap0)} checkpoint files checked)')
