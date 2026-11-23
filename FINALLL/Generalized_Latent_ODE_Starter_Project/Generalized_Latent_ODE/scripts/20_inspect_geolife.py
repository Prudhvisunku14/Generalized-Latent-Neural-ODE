#!/usr/bin/env python3
"""Phase 1: audit the actual GeoLife 1.3 archive (read in place, file by file) and build the index of eligible
200-record segments under the thresholds in configs/geolife.yaml. No model is used here.

Outputs
  results/geolife/dataset_statistics.json      dataset-level summary
  results/geolife/dataset_statistics.csv       one row per .plt file
  data/geolife/processed/segment_index.csv     every candidate window with its eligibility decision (GeoLife-derived)
  results/geolife/figures/fig1_*.png, audit_*.png
  reports/geolife_dataset_audit.md
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import yaml
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from glo import geolife as gl

root = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--config', default=str(root / 'configs/geolife.yaml'))
ap.add_argument('--workers', type=int, default=16)
args = ap.parse_args()
cfg_all = yaml.safe_load(open(args.config))
cfg = gl.SegmentConfig(**cfg_all['segments'])
zip_path = root / cfg_all['archive']
res_dir = root / 'results/geolife'; fig_dir = res_dir / 'figures'; proc = root / 'data/geolife/processed'
for d in (res_dir, fig_dir, proc):
    d.mkdir(parents=True, exist_ok=True)

DT_BINS = np.logspace(-1, 7, 161)            # seconds (figure only)
DT_MAX_EXACT = 100_000                       # exact integer-second counts up to this value (larger values pooled)
SP_BINS = np.logspace(-3, 4, 141)            # m/s
_arch = None


def _init():
    global _arch
    _arch = gl.GeoLifeArchive(zip_path)


def audit_file(name):
    p = gl.parse_plt(_arch.read(name))
    keep, dup, back = gl.clean_time_order(p.t_unix)
    t, la, lo = p.t_unix[keep], p.lat[keep], p.lon[keep]
    row = {'user': gl.GeoLifeArchive.user_of(name), 'file': Path(name).name, 'n_lines': p.n_lines,
           'n_malformed': p.n_malformed, 'n_invalid_coord': p.n_invalid_coord, 'n_duplicate_time': dup,
           'n_backward_time': back, 'n_datenum_mismatch': p.n_datenum_mismatch, 'header_ok': p.header_ok, 'n_clean': len(t)}
    dt_hist = np.zeros(len(DT_BINS) - 1, np.int64); sp_hist = np.zeros(len(SP_BINS) - 1, np.int64)
    dt_exact = np.zeros(DT_MAX_EXACT + 1, np.int64)
    windows = []
    if len(t) >= 2:
        dt = np.diff(t).astype(float); step = gl.haversine(la[:-1], lo[:-1], la[1:], lo[1:]); sp = step / dt
        dt_hist += np.histogram(dt, DT_BINS)[0]; dt_exact += np.bincount(np.minimum(dt.astype(np.int64), DT_MAX_EXACT), minlength=DT_MAX_EXACT + 1); sp_hist += np.histogram(sp, SP_BINS)[0]
        runs = gl.continuous_runs(t, la, lo, cfg)
        inrun = np.zeros(len(dt), bool)
        for a, b in runs:
            inrun[a:b - 1] = True
        row.update(start_unix=int(t[0]), end_unix=int(t[-1]), duration_s=float(t[-1] - t[0]), dt_median_s=float(np.median(dt)),
                   dt_p95_s=float(np.percentile(dt, 95)), dt_max_s=float(dt.max()), n_gaps_gt_max=int((dt > cfg.max_gap_s).sum()),
                   n_jumps_gt_max_speed=int((sp > cfg.max_speed_mps).sum()), path_length_m=float(step.sum()),
                   path_length_within_runs_m=float(step[inrun].sum()), n_runs=len(runs),
                   n_runs_ge_window=sum(1 for a, b in runs if b - a >= cfg.window_points))
        for s, e, ok, why in gl.candidate_windows(t, la, lo, cfg):
            x, y = gl.aeqd_forward(la[s:e], lo[s:e], la[s], lo[s])
            st = np.hypot(np.diff(x), np.diff(y)); dur = float(t[e - 1] - t[s])
            windows.append({'user': row['user'], 'file': row['file'], 'start': s, 'stop': e, 'eligible': ok, 'reason': why,
                            't0_unix': int(t[s]), 't1_unix': int(t[e - 1]), 'duration_s': dur,
                            'extent_m': float(np.hypot(x, y).max()), 'path_length_m': float(st.sum()),
                            'mean_speed_mps': float(st.sum() / dur), 'median_dt_s': float(np.median(np.diff(t[s:e])))})
    else:
        row.update(start_unix=None, end_unix=None, duration_s=0.0, dt_median_s=None, dt_p95_s=None, dt_max_s=None, n_gaps_gt_max=0,
                   n_jumps_gt_max_speed=0, path_length_m=0.0, path_length_within_runs_m=0.0, n_runs=0, n_runs_ge_window=0)
    row['n_windows'] = len(windows); row['n_eligible_windows'] = sum(w['eligible'] for w in windows)
    return row, dt_hist, dt_exact, sp_hist, windows


if __name__ == '__main__':
    t_start = time.time()
    sha = gl.file_sha256(zip_path)
    if sha != cfg_all['archive_sha256']:
        sys.exit(f'archive checksum {sha} differs from the configured {cfg_all["archive_sha256"]}')
    arch = gl.GeoLifeArchive(zip_path)
    names = arch.plt_names
    users = sorted({gl.GeoLifeArchive.user_of(n) for n in names})
    print(f'{len(names)} .plt files, {len(users)} users, {len(arch.label_names)} label files', flush=True)
    rows, windows = [], []
    dt_hist = np.zeros(len(DT_BINS) - 1, np.int64); sp_hist = np.zeros(len(SP_BINS) - 1, np.int64)
    dt_exact = np.zeros(DT_MAX_EXACT + 1, np.int64)
    with Pool(args.workers, initializer=_init) as pool:
        for i, (r, dh, de, sh, w) in enumerate(pool.imap(audit_file, names, chunksize=32)):
            rows.append(r); windows += w; dt_hist += dh; dt_exact += de; sp_hist += sh
            if (i + 1) % 4000 == 0:
                print(f'  {i + 1}/{len(names)}', flush=True)

    import pandas as pd
    df = pd.DataFrame(rows); dfw = pd.DataFrame(windows)
    df.to_csv(res_dir / 'dataset_statistics.csv', index=False)
    dfw.to_csv(proc / 'segment_index.csv', index=False)

    def q(s, ps=(5, 25, 50, 75, 95)):
        s = pd.Series(s).dropna()
        return {f'p{p}': float(np.percentile(s, p)) for p in ps} | {'mean': float(s.mean()), 'max': float(s.max())}

    def exact_pct(counts, ps=(5, 25, 50, 75, 95, 99, 99.9)):
        c = np.cumsum(counts) / counts.sum()
        return {f'p{p}': int(np.searchsorted(c, p / 100)) for p in ps}

    def hist_pct(hist, bins, ps=(5, 25, 50, 75, 95, 99)):
        c = np.cumsum(hist) / hist.sum()
        return {f'p{p}': float(bins[1:][np.searchsorted(c, p / 100)]) for p in ps}

    elig = dfw[dfw.eligible]
    per_user_elig = elig.groupby('user').size()
    summary = {
        'archive': str(zip_path.relative_to(root)), 'archive_sha256': sha,
        'n_users': len(users), 'n_users_with_labels': len(arch.label_names), 'n_trajectory_files': len(names),
        'n_data_lines': int(df.n_lines.sum()), 'n_malformed_lines': int(df.n_malformed.sum()),
        'n_invalid_coordinates': int(df.n_invalid_coord.sum()), 'n_duplicate_timestamps': int(df.n_duplicate_time.sum()),
        'n_backward_timestamps': int(df.n_backward_time.sum()), 'n_files_with_backward_time': int((df.n_backward_time > 0).sum()),
        'n_datenum_vs_string_mismatch_gt_1s': int(df.n_datenum_mismatch.sum()), 'n_files_bad_header': int((~df.header_ok).sum()),
        'n_clean_records': int(df.n_clean.sum()),
        'records_per_file': q(df.n_clean), 'duration_per_file_s': q(df.duration_s),
        'path_length_per_file_m': q(df.path_length_m), 'total_path_length_km_within_runs': float(df.path_length_within_runs_m.sum() / 1e3),
        'total_duration_h': float(df.duration_s.sum() / 3600),
        'n_steps_after_cleaning': int(dt_exact.sum()),
        'sampling_interval_s_percentiles_all_steps': exact_pct(dt_exact),
        'sampling_interval_s_counts': {f'{k}s': int(dt_exact[k]) for k in (1, 2, 3, 4, 5)},
        'fraction_steps_dt_le_5s': float(dt_exact[:6].sum() / dt_exact.sum()),
        'fraction_steps_dt_gt_max_gap': float(dt_exact[int(cfg.max_gap_s) + 1:].sum() / dt_exact.sum()),
        'step_speed_mps_percentiles_all_steps_log_bin_upper_edge': hist_pct(sp_hist, SP_BINS),
        'fraction_steps_speed_gt_max_speed': float(sp_hist[SP_BINS[:-1] >= cfg.max_speed_mps].sum() / sp_hist.sum()),
        'n_steps_gt_max_speed': int(df.n_jumps_gt_max_speed.sum()), 'n_gaps_gt_max_gap': int(df.n_gaps_gt_max.sum()),
        'segment_thresholds': cfg.__dict__,
        'n_candidate_windows': int(len(dfw)), 'window_reasons': {k: int(v) for k, v in dfw.reason.value_counts().items()},
        'n_eligible_windows': int(len(elig)), 'n_users_with_eligible_windows': int(per_user_elig.size),
        'n_files_with_eligible_windows': int(elig.groupby(['user', 'file']).ngroups),
        'eligible_windows_per_user': q(per_user_elig), 'eligible_window_duration_s': q(elig.duration_s),
        'eligible_window_path_length_m': q(elig.path_length_m), 'eligible_window_extent_m': q(elig.extent_m),
        'eligible_window_mean_speed_mps': q(elig.mean_speed_mps), 'eligible_window_median_dt_s': q(elig.median_dt_s),
        'runtime_s': round(time.time() - t_start, 1),
    }
    (res_dir / 'dataset_statistics.json').write_text(json.dumps(summary, indent=2))
    np.savez(res_dir / 'audit_histograms.npz', dt_bins=DT_BINS, dt_hist=dt_hist, dt_exact=dt_exact, sp_bins=SP_BINS, sp_hist=sp_hist)

    # ------------------------------------------------------------------ figures (all from actual records)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(df.n_clean[df.n_clean > 0], bins=np.logspace(0, 5.5, 60), color='#2578bc'); ax[0].set_xscale('log')
    ax[0].set_xlabel('records per trajectory file'); ax[0].set_ylabel('files'); ax[0].set_title('Trajectory length')
    ax[1].stairs(dt_hist / dt_hist.sum(), DT_BINS, fill=True, color='#2578bc'); ax[1].set_xscale('log')
    ax[1].axvline(cfg.max_gap_s, color='r', ls='--', lw=1, label=f'run break > {cfg.max_gap_s:g} s'); ax[1].legend(frameon=False)
    ax[1].set_xlabel('interval between consecutive records (s)'); ax[1].set_ylabel('fraction of steps'); ax[1].set_title('Sampling interval')
    ax[2].hist(df.duration_s[df.duration_s > 0] / 60, bins=np.logspace(-2, 4, 60), color='#2578bc'); ax[2].set_xscale('log')
    ax[2].set_xlabel('duration (min)'); ax[2].set_ylabel('files'); ax[2].set_title('Trajectory duration')
    fig.suptitle(f'GeoLife 1.3: {len(names):,} files, {summary["n_clean_records"]:,} clean records, {len(users)} users')
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(fig_dir / 'audit_distributions.png', dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.stairs(sp_hist / sp_hist.sum(), SP_BINS, fill=True, color='#8a5cc2'); ax.set_xscale('log')
    ax.axvline(cfg.max_speed_mps, color='r', ls='--', lw=1, label=f'jump break > {cfg.max_speed_mps:g} m/s'); ax.legend(frameon=False)
    ax.set_xlabel('step speed (m/s)'); ax.set_ylabel('fraction of steps'); ax.set_title('Speed between consecutive records')
    fig.tight_layout(); fig.savefig(fig_dir / 'audit_step_speed.png', dpi=140); plt.close(fig)

    # Figure 1: eligible segments drawn at random (seeded), not chosen by appearance
    rng = np.random.default_rng(0)
    pick = elig.iloc[rng.choice(len(elig), 12, replace=False)]
    fig, axs = plt.subplots(3, 4, figsize=(15, 11))
    for a, (_, w) in zip(axs.flat, pick.iterrows()):
        t, la, lo = arch.load_clean(f'Geolife Trajectories 1.3/Data/{w.user}/Trajectory/{w.file}')
        s, e = int(w.start), int(w.stop)
        x, y = gl.aeqd_forward(la[s:e], lo[s:e], la[s], lo[s])
        sc = a.scatter(x, y, c=(t[s:e] - t[s]) / 60, s=5, cmap='viridis'); a.plot(x[0], y[0], 'k^', ms=7)
        a.set_aspect('equal', 'datalim'); a.set_title(f'user {w.user}  {w.duration_s / 60:.1f} min  {w.path_length_m:.0f} m', fontsize=9)
        a.tick_params(labelsize=7); plt.colorbar(sc, ax=a, fraction=0.04).set_label('min', fontsize=7)
    fig.suptitle('Figure 1. Twelve randomly drawn eligible GeoLife segments (200 records each; local metres, ▲ = start; colour = elapsed time)')
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(fig_dir / 'fig1_geolife_examples.png', dpi=140); plt.close(fig)
    print(json.dumps({k: summary[k] for k in ('n_users', 'n_trajectory_files', 'n_data_lines', 'n_clean_records', 'n_candidate_windows',
                                              'n_eligible_windows', 'n_users_with_eligible_windows', 'window_reasons', 'runtime_s')}, indent=1))
