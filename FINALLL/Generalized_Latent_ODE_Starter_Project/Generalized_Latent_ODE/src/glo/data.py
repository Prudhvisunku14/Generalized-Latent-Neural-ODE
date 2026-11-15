"""Reproducible synthetic trajectory families and dataset management.

All generated labels represent time-ordered 2-D trajectories, not unordered shapes.
A single master sample belongs to exactly one train/val/test split, irrespective
of whether it is later used in the two- or three-family experiment.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

FAMILY_TO_ID = {"circle": 0, "spiral": 1, "ellipse": 2}
PARAM_NAMES = ["radius_or_r0", "omega", "phase", "cx", "cy", "growth", "a", "b", "angle"]


def sample_trajectory(family: str, t: np.ndarray, rng: np.random.Generator):
    """Return clean [T, 2] trajectory and a fixed-length, finite parameter vector.

    omega's sign encodes clockwise/counterclockwise motion. Spirals are
    logarithmic here; original Chen et al. example uses a different generator.
    """
    if family not in FAMILY_TO_ID:
        raise ValueError(f"Unknown family: {family}")
    radius = float(rng.uniform(.55, 1.5))
    omega = float(rng.choice([-1, 1]) * rng.uniform(.75, 1.45))
    phase = float(rng.uniform(-np.pi, np.pi))
    cx, cy = rng.uniform(-.35, .35, size=2).tolist()
    growth = 0.0
    a = b = radius
    alpha = 0.0
    angle = omega * t + phase
    if family == 'circle':
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
    elif family == 'spiral':
        growth = float(rng.choice([-1, 1]) * rng.uniform(.055, .12))
        r_t = radius * np.exp(growth * t)
        x = cx + r_t * np.cos(angle)
        y = cy + r_t * np.sin(angle)
    else:
        a = float(rng.uniform(.85, 1.65))
        b = float(a * rng.uniform(.40, .78))
        alpha = float(rng.uniform(0, np.pi))
        xx, yy = a * np.cos(angle), b * np.sin(angle)
        x = cx + np.cos(alpha) * xx - np.sin(alpha) * yy
        y = cy + np.sin(alpha) * xx + np.cos(alpha) * yy
    params = np.asarray([radius, omega, phase, cx, cy, growth, a, b, alpha], dtype=np.float32)
    xy = np.column_stack([x, y]).astype(np.float32)
    return xy, params


def _family_splits(family, n, t, noise_std, n_obs, forecast_start, seed):
    rng = np.random.default_rng(seed + 1009 * (FAMILY_TO_ID[family] + 1))
    xs, pars = [], []
    for _ in range(n):
        xy, p = sample_trajectory(family, t, rng)
        xs.append(xy)
        pars.append(p)
    xy = np.stack(xs)
    noisy_xy = (xy + rng.normal(0, noise_std, size=xy.shape)).astype(np.float32)
    cutoff_candidates = np.flatnonzero(t <= forecast_start)
    if n_obs > len(cutoff_candidates):
        raise ValueError(f"n_obs={n_obs} exceeds available context timestamps={len(cutoff_candidates)}")
    masks = np.zeros((n, len(t)), dtype=bool)
    for k in range(n):
        remaining = rng.choice(cutoff_candidates[1:], size=n_obs - 1, replace=False)
        masks[k, np.concatenate([[0], remaining])] = True
    perm = rng.permutation(n)
    n_train, n_val = int(.70 * n), int(.15 * n)
    selections = {"train": perm[:n_train], "val": perm[n_train:n_train + n_val], "test": perm[n_train + n_val:]}
    ids = np.asarray([f"{family}_{i:05d}" for i in range(n)])
    all_fields = {"xy": xy, "noisy_xy": noisy_xy, "family": np.full(n, FAMILY_TO_ID[family], dtype=np.int64),
                  "ids": ids, "params": np.stack(pars), "default_obs_mask": masks}
    return {split: {key: val[idx] for key, val in all_fields.items()} for split, idx in selections.items()}


def generate_project_data(out_root, n_per_family=1000, n_points=200, seed=42,
                          noise_std=.05, n_obs=30, forecast_start=7.0, t_end=10.):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if n_points < 10 or n_per_family < 10 or n_obs < 2 or not 0 < forecast_start < t_end:
        raise ValueError('Invalid generation parameters')
    t = np.linspace(0, t_end, n_points, dtype=np.float32)
    families = {f: _family_splits(f, n_per_family, t, noise_std, n_obs, forecast_start, seed)
                for f in FAMILY_TO_ID}
    summaries = {}
    for name, selected in (("two_shapes", ["circle", "spiral"]),
                           ("three_shapes", ["circle", "spiral", "ellipse"])):
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        assembled = {}
        for split in ("train", "val", "test"):
            fields = {key: np.concatenate([families[f][split][key] for f in selected], axis=0)
                      for key in ("xy", "noisy_xy", "family", "ids", "params", "default_obs_mask")}
            idx = np.random.default_rng(seed + {"train": 11, "val": 17, "test": 23}[split]).permutation(len(fields['family']))
            assembled[split] = {key: value[idx] for key, value in fields.items()}
            np.savez_compressed(out / f"{split}.npz", time=t, **assembled[split])
        train_xy = assembled['train']['xy'].astype(np.float64)
        mean = train_xy.mean(axis=(0, 1))
        scale = float(np.sqrt(np.mean((train_xy - mean) ** 2)))
        if not scale > 0:
            raise RuntimeError('Invalid training normalization scale')
        meta = {
            "dataset_name": name, "families": selected,
            "family_to_id": FAMILY_TO_ID, "parameter_names": PARAM_NAMES,
            "n_per_family": int(n_per_family), "n_points": int(n_points), "t_end": float(t_end),
            "forecast_start": float(forecast_start), "default_n_obs": int(n_obs),
            "base_noise_std": float(noise_std), "seed": int(seed),
            "normalization_mean": mean.tolist(), "normalization_scale": scale,
            "split_counts": {split: int(len(d['family'])) for split, d in assembled.items()},
            "data_note": "All full ground-truth trajectories are clean. Observation masks are restricted to t <= cutoff."
        }
        (out / 'metadata.json').write_text(json.dumps(meta, indent=2))
        summaries[name] = meta
    return summaries


def load_split(out_root, experiment='two_shapes', split='train'):
    folder = Path(out_root) / experiment
    with np.load(folder / f'{split}.npz', allow_pickle=False) as file:
        arrays = {key: file[key].copy() for key in file.files}
    metadata = json.loads((folder / 'metadata.json').read_text())
    return arrays, metadata


def validate_dataset(out_root, experiment):
    seen = set()
    counts = {}
    train_ids = None
    for split in ('train', 'val', 'test'):
        arrays, meta = load_split(out_root, experiment, split)
        x, t, ids, masks = arrays['xy'], arrays['time'], arrays['ids'], arrays['default_obs_mask']
        assert x.shape[1:] == (meta['n_points'], 2)
        assert arrays['noisy_xy'].shape == x.shape
        assert np.isfinite(x).all() and np.isfinite(arrays['params']).all()
        assert np.all(np.diff(t) > 0)
        assert np.all(masks.sum(axis=1) == meta['default_n_obs'])
        assert np.all(masks[:, 0])
        assert not masks[:, t > meta['forecast_start']].any(), 'Future observation leak'
        assert not seen.intersection(ids.tolist()), 'Duplicate trajectory across splits'
        seen.update(ids.tolist())
        counts[split] = len(ids)
    assert counts == meta['split_counts']
    return counts
