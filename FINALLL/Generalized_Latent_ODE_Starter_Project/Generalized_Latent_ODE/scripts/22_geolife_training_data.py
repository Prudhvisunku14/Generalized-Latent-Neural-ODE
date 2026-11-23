#!/usr/bin/env python3
"""Phase 11 data: GeoLife training/validation set for training the SAME Latent ODE architecture on real GPS.

Only DEVELOPMENT users (scripts/21 'prepare') are used, split by user into train/val; the held-out test users are never
touched. Only exactly regularly sampled 200-record windows are used, so the records lie on the same uniform 200-point
time grid as the synthetic data (t = 10 i / 199) and no training target is created by interpolation. Each window is
centred on the mean of its 140 context records (t <= 7) and scaled so their RMS distance about it is 1.0 model unit.
Output: data/geolife/processed/geolife_regular/{train,val}.npz + metadata.json (GeoLife-derived; not in git).
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd, yaml
from glo import geolife as gl

root = Path(__file__).resolve().parents[1]
C = yaml.safe_load(open(root / 'configs/geolife.yaml')); R = C['real_training']
sp = json.loads((root / 'results/geolife/splits_summary.json').read_text())
dev_users = np.array(sorted(sp['dev']['users'])); test_users = set(sp['test']['users'])
rng = np.random.default_rng(R['user_split_seed']); perm = rng.permutation(dev_users)
n_val = int(round(R['val_user_fraction'] * len(dev_users)))
users = {'val': sorted(perm[:n_val]), 'train': sorted(perm[n_val:])}
assert not (set(users['train']) | set(users['val'])) & test_users
idx = pd.read_csv(root / 'data/geolife/processed/segment_index.csv', dtype={'user': str}); idx = idx[idx.eligible]
arch = gl.GeoLifeArchive(root / C['archive'])
out = root / 'data/geolife/processed/geolife_regular'; out.mkdir(parents=True, exist_ok=True)
t_grid = np.linspace(0, 10, 200, dtype=np.float32)
rs = np.random.default_rng(R['sampling_seed']); assembled = {}; info = {}
for split in ('train', 'val'):
    cap = R[f'max_windows_per_user_{split}']; X, ids, params = [], [], []
    for u in users[split]:
        rows = []
        for f, g in idx[idx.user == u].groupby('file'):
            t, la, lo = arch.load_clean(f'Geolife Trajectories 1.3/Data/{u}/Trajectory/{f}')
            for _, w in g.iterrows():
                tt = t[w.start:w.stop]; d = np.diff(tt)
                if (d == d[0]).all():
                    rows.append((f, int(w.start), tt, la[w.start:w.stop], lo[w.start:w.stop], int(d[0])))
        if len(rows) > cap:
            rows = [rows[i] for i in np.sort(rs.choice(len(rows), cap, replace=False))]
        for f, s, tt, la_, lo_, dt in rows:
            assert np.allclose(gl.model_time_full(tt), t_grid, atol=1e-5)
            x, y = gl.aeqd_forward(la_, lo_, la_[0], lo_[0]); xy = np.stack([x, y], 1)
            ctx = xy[:140]; c = ctx.mean(0); spread = float(np.sqrt((np.linalg.norm(ctx - c, axis=1) ** 2).mean()))
            X.append(((xy - c) / spread).astype(np.float32)); ids.append(f'{u}/{f}/{s}'); params.append([spread, float(tt[-1] - tt[0]), dt])
    X = np.stack(X); n = len(X)
    masks = np.zeros((n, 200), bool)
    for k in range(n):
        masks[k, np.concatenate([[0], rs.choice(np.arange(1, 140), 29, replace=False)])] = True
    order = rs.permutation(n)
    assembled[split] = dict(xy=X[order], noisy_xy=X[order], family=np.zeros(n, np.int64), ids=np.array(ids)[order],
                            params=np.array(params)[order], default_obs_mask=masks[order])
    np.savez_compressed(out / f'{split}.npz', time=t_grid, **assembled[split])
    info[split] = {'n_users': len(users[split]), 'n_windows': n, 'per_user_max': int(pd.Series([i.split('/')[0] for i in ids]).value_counts().max()),
                   'sampling_interval_s_counts': {str(k): int(v) for k, v in pd.Series([p[2] for p in params]).value_counts().items()},
                   'rms_spread_m_median': float(np.median([p[0] for p in params]))}
tr = assembled['train']['xy'].astype(np.float64); mean = tr.mean(axis=(0, 1)); scale = float(np.sqrt(np.mean((tr - mean) ** 2)))
meta = {'dataset_name': 'geolife_regular', 'families': ['gps'], 'family_to_id': {'gps': 0},
        'parameter_names': ['rms_spread_m', 'duration_s', 'sampling_interval_s'], 'n_points': 200, 't_end': 10.0,
        'forecast_start': 7.0, 'default_n_obs': 30, 'base_noise_std': 0.05, 'seed': R['sampling_seed'],
        'normalization_mean': mean.tolist(), 'normalization_scale': scale,
        'split_counts': {k: v['n_windows'] for k, v in info.items()}, 'users': users, 'split_info': info,
        'data_note': 'Real GeoLife GPS (development users only), exactly regular 200-record windows, per-window context-normalised; '
                     'xy = noisy_xy = recorded GPS (no clean truth exists). No test split: evaluation uses the held-out test users of scripts/21.'}
(out / 'metadata.json').write_text(json.dumps(meta, indent=2))
print(json.dumps({k: meta[k] for k in ('split_counts', 'normalization_mean', 'normalization_scale', 'split_info')}, indent=1))
