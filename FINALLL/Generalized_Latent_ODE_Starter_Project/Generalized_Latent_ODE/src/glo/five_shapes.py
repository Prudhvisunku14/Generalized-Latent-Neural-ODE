"""Five-family synthetic dataset: the original circle / spiral / ellipse families plus figure-eight and sinusoid.

The three original families are produced by the UNCHANGED `data._family_splits` (same seeds, same split
assignment), so their trajectories, noisy observations, masks and train/val/test membership are identical to
`data/processed/three_shapes`. The two new families use the same protocol (200 points on [0, 10], i.i.d. Gaussian
observation noise of std 0.05, 30 observation timestamps in [0, 7] including t = 0, 70/15/15 split per family),
each with its own seeded random stream.

Parameter ranges reuse the original ones wherever a parameter has the same role:
angular frequency |omega| ~ U(0.75, 1.45), phase ~ U(-pi, pi), centre offsets cx, cy ~ U(-0.35, 0.35).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from . import data as D

FAMILY_TO_ID = {"circle": 0, "spiral": 1, "ellipse": 2, "figure_eight": 3, "sinusoid": 4}
NEW_FAMILIES = ("figure_eight", "sinusoid")
PARAM_NAMES = D.PARAM_NAMES + ["velocity", "x0"]       # original 9 columns unchanged, two appended (0 for old families)

RANGES = {
    "figure_eight": {"A_horizontal": (0.85, 1.5), "B_vertical": (0.45, 1.0), "abs_omega": (0.75, 1.45),
                     "phase": (-np.pi, np.pi), "centre": (-0.35, 0.35), "direction": "sign of omega, +-1 with p = 1/2"},
    "sinusoid": {"amplitude": (0.4, 1.0), "omega": (0.75, 1.45), "abs_velocity": (0.2, 0.3), "phase": (-np.pi, np.pi),
                 "centre": (-0.35, 0.35), "direction": "sign of velocity, +-1 with p = 1/2",
                 "x0": "cx - 5 v, so the path is centred on cx over t in [0, 10]"},
}


def sample_new_trajectory(family: str, t: np.ndarray, rng: np.random.Generator):
    """Clean [T, 2] trajectory and an 11-column parameter vector (PARAM_NAMES)."""
    if family == "figure_eight":
        r = RANGES[family]
        A = float(rng.uniform(*r["A_horizontal"]))
        B = float(rng.uniform(*r["B_vertical"]))
        omega = float(rng.choice([-1, 1]) * rng.uniform(*r["abs_omega"]))
        phase = float(rng.uniform(*r["phase"]))
        cx, cy = rng.uniform(*r["centre"], size=2).tolist()
        theta = omega * t + phase
        x, y = cx + A * np.sin(theta), cy + B * np.sin(2 * theta)
        params = [A, omega, phase, cx, cy, 0.0, A, B, 0.0, 0.0, 0.0]
    elif family == "sinusoid":
        r = RANGES[family]
        amp = float(rng.uniform(*r["amplitude"]))
        omega = float(rng.uniform(*r["omega"]))
        phase = float(rng.uniform(*r["phase"]))
        v = float(rng.choice([-1, 1]) * rng.uniform(*r["abs_velocity"]))
        cx, cy = rng.uniform(*r["centre"], size=2).tolist()
        x0 = cx - 5.0 * v
        x, y = x0 + v * t, cy + amp * np.sin(omega * t + phase)
        params = [amp, omega, phase, cx, cy, 0.0, amp, amp, 0.0, v, x0]
    else:
        raise ValueError(f"not a new family: {family}")
    return np.column_stack([x, y]).astype(np.float32), np.asarray(params, dtype=np.float32)


def _new_family_splits(family, n, t, noise_std, n_obs, forecast_start, seed):
    """Same steps, in the same order, as data._family_splits, with the new-family sampler."""
    rng = np.random.default_rng(seed + 1009 * (FAMILY_TO_ID[family] + 1))
    xs, pars = zip(*(sample_new_trajectory(family, t, rng) for _ in range(n)))
    xy = np.stack(xs)
    noisy_xy = (xy + rng.normal(0, noise_std, size=xy.shape)).astype(np.float32)
    cand = np.flatnonzero(t <= forecast_start)
    masks = np.zeros((n, len(t)), dtype=bool)
    for k in range(n):
        masks[k, np.concatenate([[0], rng.choice(cand[1:], size=n_obs - 1, replace=False)])] = True
    perm = rng.permutation(n)
    n_train, n_val = int(.70 * n), int(.15 * n)
    sel = {"train": perm[:n_train], "val": perm[n_train:n_train + n_val], "test": perm[n_train + n_val:]}
    fields = {"xy": xy, "noisy_xy": noisy_xy, "family": np.full(n, FAMILY_TO_ID[family], dtype=np.int64),
              "ids": np.asarray([f"{family}_{i:05d}" for i in range(n)]), "params": np.stack(pars), "default_obs_mask": masks}
    return {s: {k: v[idx] for k, v in fields.items()} for s, idx in sel.items()}


def generate_five_shapes(out_root, n_per_family=1000, n_points=200, seed=42, noise_std=.05, n_obs=30,
                         forecast_start=7.0, t_end=10.0, name="five_shapes", three_shapes_dir=None):
    out = Path(out_root) / name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already exists; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, t_end, n_points, dtype=np.float32)
    fams = {}
    for f in ("circle", "spiral", "ellipse"):
        fs = D._family_splits(f, n_per_family, t, noise_std, n_obs, forecast_start, seed)
        for split, s in fs.items():
            # Regeneration reproduces parameters, masks and split membership exactly, but ellipse coordinates can differ
            # from the stored three_shapes arrays by one float32 rounding step (library version). Use the stored arrays.
            if three_shapes_dir is not None:
                with np.load(Path(three_shapes_dir) / f"{split}.npz") as z:
                    pos = {k: i for i, k in enumerate(z["ids"].tolist())}
                    j = np.array([pos[k] for k in s["ids"].tolist()])
                    assert np.array_equal(z["params"][j], s["params"]) and np.array_equal(z["default_obs_mask"][j], s["default_obs_mask"])
                    assert np.abs(z["xy"][j] - s["xy"]).max() < 1e-5 and np.abs(z["noisy_xy"][j] - s["noisy_xy"]).max() < 1e-5
                    s["xy"], s["noisy_xy"] = z["xy"][j], z["noisy_xy"][j]
            s["params"] = np.concatenate([s["params"], np.zeros((len(s["params"]), 2), np.float32)], 1)
        fams[f] = fs
    for f in NEW_FAMILIES:
        fams[f] = _new_family_splits(f, n_per_family, t, noise_std, n_obs, forecast_start, seed)
    assembled = {}
    for split in ("train", "val", "test"):
        fields = {k: np.concatenate([fams[f][split][k] for f in FAMILY_TO_ID], 0)
                  for k in ("xy", "noisy_xy", "family", "ids", "params", "default_obs_mask")}
        idx = np.random.default_rng(seed + {"train": 11, "val": 17, "test": 23}[split]).permutation(len(fields["family"]))
        assembled[split] = {k: v[idx] for k, v in fields.items()}
        np.savez_compressed(out / f"{split}.npz", time=t, **assembled[split])
    train_xy = assembled["train"]["xy"].astype(np.float64)
    mean = train_xy.mean(axis=(0, 1))
    scale = float(np.sqrt(np.mean((train_xy - mean) ** 2)))
    meta = {"dataset_name": name, "families": list(FAMILY_TO_ID), "family_to_id": FAMILY_TO_ID, "parameter_names": PARAM_NAMES,
            "n_per_family": int(n_per_family), "n_points": int(n_points), "t_end": float(t_end),
            "forecast_start": float(forecast_start), "default_n_obs": int(n_obs), "base_noise_std": float(noise_std),
            "seed": int(seed), "normalization_mean": mean.tolist(), "normalization_scale": scale,
            "split_counts": {s: int(len(d["family"])) for s, d in assembled.items()},
            "new_family_parameter_ranges": RANGES,
            "parameter_mapping_new_families": {
                "figure_eight": "radius_or_r0 = a = A (horizontal amplitude), b = B (vertical amplitude), omega, phase, cx, cy",
                "sinusoid": "radius_or_r0 = a = b = amplitude, omega, phase, cx, cy (y offset y0 = cy), velocity = v, x0"},
            "data_note": "All full ground-truth trajectories are clean. Observation masks are restricted to t <= cutoff. "
                         "circle/spiral/ellipse are identical to three_shapes (same generator, seeds and split membership)."}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


# ------------------------------------------------------------------------------------------------ shape checks
def figure_eight_checks(xy, params, t):
    """Measured properties of one figure-eight (clean trajectory only): y oscillates twice as fast as x; every time x
    crosses its centre line, y is also at its centre (the curve passes through its centre point), and successive
    passes cross with opposite slopes (the two lobes, i.e. a genuine self-crossing rather than a turn-back)."""
    cx, cy, B = params[3], params[4], params[7]
    sx = np.signbit(xy[:, 0] - cx).astype(int); sy = np.signbit(xy[:, 1] - cy).astype(int)
    zx, zy = int(np.abs(np.diff(sx)).sum()), int(np.abs(np.diff(sy)).sum())
    ks = np.flatnonzero(np.diff(sx))
    y_at, slope = [], []
    for k in ks:
        f = (cx - xy[k, 0]) / (xy[k + 1, 0] - xy[k, 0])
        y_at.append(xy[k, 1] + f * (xy[k + 1, 1] - xy[k, 1]) - cy)
        slope.append(np.sign((xy[k + 1, 1] - xy[k, 1]) * (xy[k + 1, 0] - xy[k, 0])))
    alternating = len(slope) >= 2 and all(a != b for a, b in zip(slope, slope[1:]))
    return {"x_zero_crossings": zx, "y_zero_crossings": zy, "centre_passes": len(ks),
            "max_abs_y_offset_at_centre_over_B": float(np.max(np.abs(y_at)) / B) if len(ks) else np.inf,
            "crossing_directions_alternate": bool(alternating),
            "covers_full_eight": bool(abs(params[1]) * (t[-1] - t[0]) >= 2 * np.pi)}


def sinusoid_checks(xy, params, t):
    v, amp, cy = params[9], params[0], params[4]
    dx = np.diff(xy[:, 0])
    resid_x = xy[:, 0] - (params[10] + v * t)
    return {"x_monotone": bool(np.all(np.sign(dx) == np.sign(v))), "x_linear_max_residual": float(np.abs(resid_x).max()),
            "y_half_range_over_amplitude": float((xy[:, 1].max() - xy[:, 1].min()) / 2 / amp),
            "y_mean_offset_from_cy": float(abs(xy[:, 1].mean() - cy))}
