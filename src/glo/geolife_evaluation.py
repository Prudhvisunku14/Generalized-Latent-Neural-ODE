"""Methods, metrics and trajectory descriptors for the GeoLife transfer experiment.

The Latent ODE is called exclusively through :func:`glo.inference.predict` (the verified inference path used by
scripts/predict_trajectory.py); this module only maps GPS metres into the model's raw coordinate units and back.
Every method receives the same observations and is scored at the same withheld GPS timestamps against the recorded
GPS positions (never against an interpolated reference).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from . import inference as inf
from .geolife import Task, NormStrategy, fit_transform

INTERP_METHODS = ('latent_ode', 'linear', 'cubic', 'pchip')
FORECAST_METHODS = ('latent_ode', 'hold_last', 'const_velocity')


# ------------------------------------------------------------------------------------------------------- metrics
def error_metrics(pred_m, true_m) -> dict:
    """Errors in metres. rmse_xy = sqrt(sum over points and both coordinates / (2N)), the definition used by the
    project's evaluation (pipeline.infer_test_metrics); ADE = mean Euclidean error."""
    pred_m = np.asarray(pred_m, np.float64); true_m = np.asarray(true_m, np.float64)
    if pred_m.shape != true_m.shape or pred_m.ndim != 2 or pred_m.shape[1] != 2:
        raise ValueError(f'shape mismatch {pred_m.shape} vs {true_m.shape}')
    n = len(true_m)
    if n == 0:
        return {'n': 0}
    d = pred_m - true_m
    e = np.hypot(d[:, 0], d[:, 1])
    return {'n': int(n), 'rmse_xy': float(np.sqrt((d ** 2).sum() / (2 * n))), 'mae_xy': float(np.abs(d).mean()),
            'ade': float(e.mean()), 'median_error': float(np.median(e)), 'p90_error': float(np.percentile(e, 90)),
            'max_error': float(e.max()), 'fde': float(e[-1])}


# ------------------------------------------------------------------------------------------------- baselines
def linear_interp(t_obs, xy_obs, t_q):
    return np.stack([np.interp(t_q, t_obs, xy_obs[:, k]) for k in range(2)], 1)


def cubic_interp(t_obs, xy_obs, t_q):
    """Not-a-knot cubic spline fitted to the observations only; evaluation outside the observed range is refused."""
    t_q = np.asarray(t_q, np.float64)
    if t_q.size and (t_q.min() < t_obs[0] - 1e-9 or t_q.max() > t_obs[-1] + 1e-9):
        raise ValueError('cubic interpolation queried outside the observed time range (that would be extrapolation)')
    return CubicSpline(t_obs, xy_obs, axis=0, bc_type='not-a-knot', extrapolate=False)(t_q)


def pchip_interp(t_obs, xy_obs, t_q):
    """Shape-preserving piecewise cubic Hermite interpolation (Fritsch-Carlson), observations only."""
    t_q = np.asarray(t_q, np.float64)
    if t_q.size and (t_q.min() < t_obs[0] - 1e-9 or t_q.max() > t_obs[-1] + 1e-9):
        raise ValueError('PCHIP queried outside the observed time range')
    return PchipInterpolator(t_obs, xy_obs, axis=0, extrapolate=False)(t_q)


def hold_last(t_obs, xy_obs, t_q):
    return np.repeat(xy_obs[-1:], len(t_q), 0)


def const_velocity(t_obs, xy_obs, t_q):
    v = (xy_obs[-1] - xy_obs[-2]) / (t_obs[-1] - t_obs[-2])
    return xy_obs[-1] + (np.asarray(t_q)[:, None] - t_obs[-1]) * v


BASELINES = {'linear': linear_interp, 'cubic': cubic_interp, 'pchip': pchip_interp,
             'hold_last': hold_last, 'const_velocity': const_velocity}


# ----------------------------------------------------------------------------------------------- latent ODE
def latent_ode_predict(lm: inf.LoadedModel, t_obs_model, xy_obs_m, t_query_model, strategy: NormStrategy):
    """Frozen model prediction in metres. The coordinate transform is fitted on the observed metres only; the encoder
    sees only the observations; query times are only integration end points."""
    tr = fit_transform(xy_obs_m, strategy)
    t0 = time.perf_counter()
    out = inf.predict(lm, np.asarray(t_obs_model, np.float64), tr.to_model(xy_obs_m), np.asarray(t_query_model, np.float64))
    dt = time.perf_counter() - t0
    return tr.to_metres(out['pred']), {'transform': tr, 'inference_s': dt, 'mu': out['mu'], 'weights_digest': out['weights_digest']}


def protocol_b_time(task: Task):
    """Protocol B (no knowledge of the segment end): elapsed seconds scaled so that the LAST OBSERVATION is at t = 7."""
    return 7.0 * task.t_sec / task.t_sec[task.obs_idx[-1]]


# ----------------------------------------------------------------------------------------- per-segment evaluation
def evaluate_interpolation(task: Task, lm, strategy: NormStrategy, methods=INTERP_METHODS, return_preds=False):
    """Primary experiment: withheld context records between the first and last observation."""
    o, q = task.obs_idx, task.interp_idx
    xy_o, xy_q = task.xy_m[o], task.xy_m[q]
    res, preds = {}, {}
    for m in methods:
        t0 = time.perf_counter()
        if m == 'latent_ode':
            p, info = latent_ode_predict(lm, task.t_model[o], xy_o, task.t_model[q], strategy)
            el = info['inference_s']
            res['scale_m_per_model_unit'] = info['transform'].metres_per_model_unit
            res['obs_rms_spread_m'] = info['transform'].s
        else:
            p = BASELINES[m](task.t_sec[o], xy_o, task.t_sec[q]); el = time.perf_counter() - t0
        r = error_metrics(p, xy_q); r['inference_s'] = el
        res[m] = r; preds[m] = p
    return (res, preds) if return_preds else res


def evaluate_forecast(task: Task, lm, strategy: NormStrategy, protocol='A', methods=FORECAST_METHODS, return_preds=False):
    """Secondary experiment. Protocol A: targets 7 < t <= 10 with the segment duration mapped to [0, 10] (known end).
    Protocol B: last observation mapped to t = 7, targets = every later record with t <= 10 (no end-time knowledge)."""
    o = task.obs_idx
    if protocol == 'A':
        tm = task.t_model; q = task.future_idx
    else:
        tm = protocol_b_time(task)
        q = np.flatnonzero((task.t_sec > task.t_sec[o[-1]]) & (tm <= 10.0 + 1e-12))
    xy_o, xy_q = task.xy_m[o], task.xy_m[q]
    res, preds = {'n_targets': int(len(q)), 'horizon_s': float(task.t_sec[q[-1]] - task.t_sec[o[-1]]) if len(q) else 0.0}, {}
    for m in methods:
        if m == 'latent_ode':
            p, _ = latent_ode_predict(lm, tm[o], xy_o, tm[q], strategy)
        else:
            p = BASELINES[m](task.t_sec[o], xy_o, task.t_sec[q])
        res[m] = error_metrics(p, xy_q); preds[m] = p
    return (res, preds, q) if return_preds else res


def interpolation_target_gaps(task: Task):
    """For each interpolation target: length (s) of the observation gap that contains it, and its relative position."""
    o = task.obs_idx; ts = task.t_sec
    k = np.searchsorted(ts[o], ts[task.interp_idx]) - 1
    gap = ts[o][k + 1] - ts[o][k]
    pos = (ts[task.interp_idx] - ts[o][k]) / gap
    return gap, pos


# --------------------------------------------------------------------------------------- trajectory descriptors
def _decimate(xy, min_step_m):
    keep = [0]
    for i in range(1, len(xy)):
        if np.linalg.norm(xy[i] - xy[keep[-1]]) >= min_step_m:
            keep.append(i)
    return xy[keep]


def descriptors(task: Task, turn_deg=45.0, decimate_m=25.0) -> dict:
    """Movement descriptors of the CONTEXT records (t <= 7), computed from the full GPS record for post-hoc
    stratification only (never used by any method)."""
    ctx = np.flatnonzero(task.t_model <= 7.0 + 1e-12)
    xy = task.xy_m[ctx]; ts = task.t_sec[ctx]
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path = float(steps.sum()); dur = float(ts[-1] - ts[0])
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    dec = _decimate(xy, decimate_m)
    n_turns = 0; total_turn = 0.0
    if len(dec) >= 3:
        h = np.arctan2(np.diff(dec[:, 1]), np.diff(dec[:, 0]))
        dh = np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi)
        n_turns = int((np.degrees(dh) > turn_deg).sum()); total_turn = float(np.degrees(dh).sum())
    straight = net / path if path > 0 else 0.0
    o = task.obs_idx
    return {'context_path_m': path, 'context_duration_s': dur, 'straightness': straight, 'n_sharp_turns': n_turns,
            'total_turning_deg': total_turn, 'mean_speed_mps': path / dur if dur > 0 else 0.0,
            'mean_obs_gap_s': float(np.mean(np.diff(task.t_sec[o]))), 'max_obs_gap_s': float(np.max(np.diff(task.t_sec[o])))}


def shape_category(d: dict) -> str:
    if d['straightness'] >= 0.9:
        return 'nearly_straight'
    return 'frequent_turning' if d['n_sharp_turns'] >= 3 else 'curved'


def speed_category(d: dict) -> str:
    v = d['mean_speed_mps']
    return 'slow_<2.5mps' if v < 2.5 else ('medium_2.5-10mps' if v < 10 else 'fast_>10mps')


# --------------------------------------------------------------------------------------------- statistics
def cluster_bootstrap(values, clusters, stat=np.mean, n_boot=2000, seed=0):
    """Percentile CI resampling whole clusters (users), since segments of one person are not independent."""
    values = np.asarray(values, np.float64); clusters = np.asarray(clusters)
    uniq = np.unique(clusters); groups = [np.flatnonzero(clusters == u) for u in uniq]
    rng = np.random.default_rng(seed); bs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        bs[b] = stat(values[np.concatenate([groups[i] for i in pick])])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
