"""Frozen-model inference for the Generalized Latent ODE (no training, no shape labels).

Pipeline (identical to the training/evaluation code path, see ``LatentODE.forward``):

    CSV (t, x, y)  ->  validation  ->  (xy - mean) / scale        [training normalisation]
      -> RecognitionGRU on the REVERSED sequence, features [x, y, t/T, dt/T], T = 10 (fixed)
      -> z0 = mu (posterior mean)           [+ dynamics code c for conditioned models]
      -> ODE  dz/dt = f(z[, c])  integrated from t = 0 with the checkpoint's own solver
      -> MLP decoder  ->  x_hat(t) * scale + mean

Design rules enforced here:
* The encoder only ever receives observations (never future coordinates, never a family label).
* T is the TRAINING value (10), not the last requested output time, so changing ``--forecast-end``
  cannot silently change the encoder inputs (``LatentODE.forward`` would derive T from the query grid).
* The ODE is always integrated from t = 0 (the time of z0 in training) on a grid anchored the same way
  as in training; output times are read from that solution.
* Weights are frozen (``eval``, ``requires_grad_(False)``, ``torch.inference_mode``); a digest of the
  state dict is compared before/after every prediction.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .model import LatentODE
from .pipeline import infer_test_metrics
from .solver import integrate

# CLI model name -> directory below saved_models/
MODEL_REGISTRY = {
    'three_shapes_unified': 'three_shapes_unified',
    'two_shapes_unified': 'two_shapes_unified',
    'circle_specialist': 'specialists/circle',
    'spiral_specialist': 'specialists/spiral',
    'dynamics_conditioned': 'experimental/dynamics_conditioned',
    'variable_nobs_experimental': 'experimental/variable_nobs',
}
DEFAULT_MODEL = 'three_shapes_unified'


class InputError(ValueError):
    """The supplied CSV (or an option) cannot be used; the message says why and how to fix it."""


class ModelUnavailableError(RuntimeError):
    """The requested pretrained model has not been packaged or failed verification."""


# --------------------------------------------------------------------------- integrity helpers
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def state_dict_digest(model: torch.nn.Module) -> str:
    """SHA-256 over every parameter/buffer tensor (name, dtype, shape, bytes)."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- model loading
@dataclass
class LoadedModel:
    name: str
    model: LatentODE
    model_config: dict
    preprocessing: dict
    metadata: dict
    checkpoint_path: Path
    sha256: str
    n_parameters: int
    load_report: dict = field(default_factory=dict)


def available_models(saved_root) -> dict:
    root = Path(saved_root)
    return {name: (root / rel / 'best_model.pt').exists() for name, rel in MODEL_REGISTRY.items()}


def load_packaged_model(name: str, saved_root, device='cpu') -> LoadedModel:
    """Rebuild a packaged model from its JSON configuration and verify it end to end."""
    if name not in MODEL_REGISTRY:
        raise ModelUnavailableError(f"Unknown model '{name}'. Choose one of: {', '.join(MODEL_REGISTRY)}")
    folder = Path(saved_root) / MODEL_REGISTRY[name]
    ckpt_path = folder / 'best_model.pt'
    if not ckpt_path.exists():
        have = [m for m, ok in available_models(saved_root).items() if ok]
        raise ModelUnavailableError(
            f"Model '{name}' is not available: {ckpt_path} does not exist. It was NOT substituted by another "
            f"model. Packaged models: {', '.join(have) if have else 'none'}.")
    missing = [f for f in ('model_config.json', 'preprocessing.json', 'metadata.json') if not (folder / f).exists()]
    if missing:
        raise ModelUnavailableError(f"{folder} is incomplete; missing {missing}")
    model_config = json.loads((folder / 'model_config.json').read_text())
    preprocessing = json.loads((folder / 'preprocessing.json').read_text())
    metadata = json.loads((folder / 'metadata.json').read_text())

    digest = sha256_file(ckpt_path)
    if metadata.get('sha256') and metadata['sha256'] != digest:
        raise ModelUnavailableError(f"Checksum mismatch for {ckpt_path}: file {digest} != recorded {metadata['sha256']}")
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if ck['model'] != model_config['architecture']:
        raise ModelUnavailableError(f"{ckpt_path}: architecture stored in the checkpoint differs from model_config.json")
    model = LatentODE(**model_config['architecture']).to(device)
    result = model.load_state_dict(ck['state_dict'], strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ModelUnavailableError(f"{ckpt_path}: state-dict mismatch, missing={result.missing_keys}, "
                                    f"unexpected={result.unexpected_keys}")
    n_params = sum(p.numel() for p in model.parameters())
    if metadata.get('n_parameters') is not None and metadata['n_parameters'] != n_params:
        raise ModelUnavailableError(f"{ckpt_path}: {n_params} parameters, metadata says {metadata['n_parameters']}")
    model.eval()
    model.requires_grad_(False)
    return LoadedModel(name, model, model_config, preprocessing, metadata, ckpt_path, digest, n_params,
                       {'missing_keys': [], 'unexpected_keys': [], 'epoch': int(ck['epoch'])})


# --------------------------------------------------------------------------- CSV input
def _read_csv(path, require_cols=('t', 'x', 'y'), what='input'):
    path = Path(path)
    if not path.exists():
        raise InputError(f"{what} file not found: {path}")
    with open(path, newline='', encoding='utf-8-sig') as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise InputError(f"{path}: file is empty") from None
        cols = [c.strip().lower() for c in header]
        missing = [c for c in require_cols if c not in cols]
        if missing:
            raise InputError(f"{path}: missing required column(s) {missing}; found {header}. "
                             f"Expected a header row with columns: {', '.join(require_cols)}")
        if len(set(cols)) != len(cols):
            raise InputError(f"{path}: duplicate column names in header {header}")
        idx = [cols.index(c) for c in require_cols]
        rows = []
        for line_no, row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in row):
                continue                                   # blank line
            if len(row) < len(cols):
                raise InputError(f"{path}: line {line_no} has {len(row)} fields, expected {len(cols)}")
            vals = []
            for c, i in zip(require_cols, idx):
                try:
                    v = float(row[i])
                except ValueError:
                    raise InputError(f"{path}: line {line_no}, column '{c}': '{row[i]}' is not a number") from None
                if not math.isfinite(v):
                    raise InputError(f"{path}: line {line_no}, column '{c}': value {row[i]!r} is not finite")
                vals.append(v)
            rows.append(vals)
    extra = [h for h, c in zip(header, cols) if c not in require_cols]
    if not rows:
        raise InputError(f"{path}: no data rows")
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1:3], extra


def read_observations(path, preprocessing: dict, trained_n_obs: int | None = None):
    """Read and validate sparse observations. Returns (t, xy, warnings), sorted and unique in t."""
    warnings: list[str] = []
    t, xy, extra = _read_csv(path, what='observation')
    if extra:
        warnings.append(f"Ignoring extra column(s) {extra}: only t, x, y are used. The model never receives "
                        f"shape labels or any other side information.")
    n = len(t)
    if n < 2:
        raise InputError(f"{path}: {n} observation(s); at least 2 are required (the encoder uses time gaps).")
    if n < 10:
        warnings.append(f"Only {n} observations: far below the 30 used in training; predictions are unreliable.")
    if np.any(np.diff(t) < 0):
        order = np.argsort(t, kind='stable')
        t, xy = t[order], xy[order]
        warnings.append("Timestamps were not sorted; rows were sorted by time (coordinates stay paired with their time).")
    dup = np.flatnonzero(np.diff(t.astype(np.float32)) <= 0)
    if dup.size:
        raise InputError(f"{path}: duplicate (or float32-indistinguishable) timestamps, e.g. t={t[dup[0]]:.9g}. "
                         f"Timestamps must be unique.")
    ctx_end = preprocessing['context_end']
    if t[0] < 0:
        raise InputError(f"{path}: negative timestamp t={t[0]:g}. Time starts at 0.")
    late = t > ctx_end + 1e-9
    if late.any():
        raise InputError(f"{path}: {int(late.sum())} observation(s) after the context end t={ctx_end:g} "
                         f"(first: t={t[late][0]:g}). Observations after t={ctx_end:g} are the forecast interval and "
                         f"must not be supplied to the model; remove them (use --ground-truth to score a forecast).")
    if t[0] > 1e-6:
        warnings.append(f"First observation is at t={t[0]:g}, not t=0. Training always included an observation at t=0 and "
                        f"the latent state is defined at t=0; accuracy near t=0 may be poorer.")
    if trained_n_obs and n != trained_n_obs:
        warnings.append(f"{n} observations supplied but this model was trained with {trained_n_obs}. Variable-length input is "
                        f"technically supported, yet accuracy away from {trained_n_obs} was only validated for the counts in "
                        f"docs/PRETRAINED_INFERENCE_GUIDE.md and can be much worse.")
    lim = preprocessing.get('coordinate_abs_max_train')
    if lim and np.abs(xy).max() > 1.5 * lim:
        warnings.append(f"Coordinates reach |{np.abs(xy).max():.3g}|, beyond 1.5x the training range ({lim:.3g}). "
                        f"Check units: the model assumes the training coordinate scale.")
    return t, xy, warnings


def read_ground_truth(path):
    t, xy, extra = _read_csv(path, what='ground-truth')
    if len(t) < 2:
        raise InputError(f"{path}: ground truth needs at least 2 rows")
    order = np.argsort(t, kind='stable')
    t, xy = t[order], xy[order]
    if np.any(np.diff(t.astype(np.float32)) <= 0):
        raise InputError(f"{path}: duplicate timestamps in ground truth")
    if t[0] < 0:
        raise InputError(f"{path}: negative timestamp in ground truth")
    return t, xy


# --------------------------------------------------------------------------- prediction
def make_query_times(n_query: int, forecast_end: float, preprocessing: dict):
    warnings: list[str] = []
    if n_query < 2:
        raise InputError("--n-query must be at least 2")
    if not math.isfinite(forecast_end) or forecast_end <= 0:
        raise InputError("--forecast-end must be a positive number")
    if forecast_end <= preprocessing['context_end']:
        warnings.append(f"--forecast-end={forecast_end:g} does not exceed the context end "
                        f"{preprocessing['context_end']:g}: the output contains no forecast region.")
    if forecast_end > preprocessing['time_horizon'] + 1e-9:
        warnings.append(f"--forecast-end={forecast_end:g} exceeds the trained horizon {preprocessing['time_horizon']:g}; "
                        f"the model was never trained or validated beyond t={preprocessing['time_horizon']:g}.")
    return np.linspace(0.0, forecast_end, n_query), warnings


def predict(lm: LoadedModel, t_obs, xy_obs, out_times, device='cpu') -> dict:
    """Reconstruct / forecast at ``out_times`` from sparse observations. Deterministic (z0 = mu)."""
    pre, model = lm.preprocessing, lm.model
    out_times = np.asarray(out_times, dtype=np.float64)
    if out_times.ndim != 1 or out_times.size == 0 or np.any(out_times < 0) or not np.all(np.isfinite(out_times)):
        raise InputError("output times must be a finite 1-D array of non-negative numbers")
    horizon = float(pre['time_horizon'])
    # Integration grid: always starts at 0 (time of z0); anchored at the training horizon when possible so the
    # fixed-step solver grid is identical to training for any subset of output times in [0, 10].
    extra = [0.0] + ([horizon] if out_times.max() <= horizon else [])
    grid32 = np.unique(np.concatenate([out_times, extra]).astype(np.float32))
    pos = np.searchsorted(grid32, out_times.astype(np.float32))
    before = state_dict_digest(model)

    mean = torch.tensor(pre['normalization_mean'], dtype=torch.float32, device=device)
    scale = float(pre['normalization_scale'])
    with torch.inference_mode():
        obs_xy = ((torch.as_tensor(xy_obs, dtype=torch.float32, device=device) - mean) / scale)[None]
        obs_t = torch.as_tensor(t_obs, dtype=torch.float32, device=device)[None]
        mu, logvar, cond = model.encoder(obs_xy, obs_t, float(pre['time_scale_T']))
        z0 = mu                                              # posterior mean: deterministic
        state = torch.cat([z0, cond], -1) if cond is not None else z0
        zt = integrate(model.dynamics, state, torch.from_numpy(grid32).to(device), method=model.solver, adjoint=False,
                       rtol=model.rtol, atol=model.atol, step_size=model.step_size)
        pred = model.decoder(zt[..., :model.latent_dim]).transpose(0, 1)[0]          # [grid, 2] normalised
        pred = pred * scale + mean
    after = state_dict_digest(model)
    if before != after:                                      # cannot happen; asserted for the frozen-weights guarantee
        raise RuntimeError('model weights changed during inference')
    pred = pred.cpu().numpy().astype(np.float64)[pos]
    out = {'pred': pred, 'mu': mu[0].cpu().numpy(), 'logvar': logvar[0].cpu().numpy(),
           'cond': cond[0].cpu().numpy() if cond is not None else None, 'weights_digest': after}
    if not np.all(np.isfinite(pred)):
        raise RuntimeError('prediction contains non-finite values')
    return out


def region_labels(times, context_end: float):
    return np.where(np.asarray(times, dtype=np.float32) <= np.float32(context_end), 'reconstruction', 'forecast')


# --------------------------------------------------------------------------- metrics (same definitions as training eval)
def evaluate_against_truth(lm: LoadedModel, pred_at_truth, gt_t, gt_xy, obs_t) -> dict:
    """Reuse ``pipeline.infer_test_metrics``. Observed context points (time-matched) are excluded from the
    reconstruction RMSE exactly as in the existing evaluation; if no ground-truth time coincides with an observation
    time, every context point is scored."""
    pre = lm.preprocessing
    gt_t32 = gt_t.astype(np.float32)
    obs_mask = np.isin(gt_t32, np.asarray(obs_t, dtype=np.float32))
    mean = torch.tensor(pre['normalization_mean'], dtype=torch.float32)
    scale = float(pre['normalization_scale'])
    prepared = {'raw_truth': torch.as_tensor(gt_xy, dtype=torch.float32)[None], 'mean': mean, 'scale': scale,
                'query_t': torch.as_tensor(gt_t32), 'obs_mask': torch.as_tensor(obs_mask)[None]}
    pred_norm = (torch.as_tensor(pred_at_truth, dtype=torch.float32)[None] - mean) / scale
    cutoff = pre['context_end']
    res: dict = {'ground_truth_points': int(len(gt_t)), 'context_end': cutoff,
                 'ground_truth_times_matching_observations': int(obs_mask.sum()),
                 'metric_units': 'original coordinate units', 'notes': []}
    if not (gt_t32 > cutoff).any():
        res['notes'].append('ground truth has no point after the context end; forecast metrics skipped')
        return res
    if not (gt_t32 <= cutoff).any():
        res['notes'].append('ground truth has no point inside the context interval')
    m = infer_test_metrics(pred_norm, prepared, cutoff)
    names = {'recon_rmse': 'reconstruction_rmse', 'forecast_rmse': 'forecast_rmse', 'forecast_mae': 'forecast_mae',
             'forecast_ade': 'forecast_ade', 'forecast_fde': 'forecast_fde', 'chamfer_full': 'chamfer_distance'}
    res.update({names[k]: float(v[0]) for k, v in m.items()})
    if res['ground_truth_times_matching_observations'] == 0:
        res['notes'].append('no ground-truth time coincides with an observation time, so reconstruction RMSE scores '
                            'all context points, including those that were observed')
    return res


# --------------------------------------------------------------------------- figures
def save_figures(out_dir, t_obs, xy_obs, times, pred, context_end, title, gt=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    GREEN, BLUE, RED = '#2a9469', '#2578bc', '#e0533a'
    ctx = times <= context_end + 1e-9
    fut = times > context_end
    paths = {}

    fig, ax = plt.subplots(figsize=(7, 6.4))
    if gt is not None:
        ax.plot(gt[1][:, 0], gt[1][:, 1], color=GREEN, lw=2.6, label='Ground truth', zorder=1)
    ax.plot(pred[ctx, 0], pred[ctx, 1], color=BLUE, lw=1.9, label=f'Reconstruction (t ≤ {context_end:g})', zorder=2)
    if fut.any():
        # connect the two segments visually without drawing the connector in the legend
        j = np.flatnonzero(ctx)[-1]
        ax.plot(pred[j:j + 2, 0], pred[j:j + 2, 1], color=RED, lw=1.9, zorder=2)
        ax.plot(pred[fut, 0], pred[fut, 1], color=RED, lw=1.9, label=f'Forecast (t > {context_end:g})', zorder=2)
    ax.scatter(xy_obs[:, 0], xy_obs[:, 1], s=20, color='black', label=f'Observations (n={len(t_obs)})', zorder=3)
    ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.2)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title(title); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); p = out_dir / 'reconstruction_forecast.png'; fig.savefig(p, dpi=200); plt.close(fig); paths['xy'] = p

    for k, name in ((0, 'x'), (1, 'y')):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        if gt is not None:
            ax.plot(gt[0], gt[1][:, k], color=GREEN, lw=2.4, label='Ground truth', zorder=1)
        ax.plot(times[ctx], pred[ctx, k], color=BLUE, lw=1.8, label='Reconstruction', zorder=2)
        if fut.any():
            j = np.flatnonzero(ctx)[-1]
            ax.plot(times[j:j + 2], pred[j:j + 2, k], color=RED, lw=1.8, zorder=2)
            ax.plot(times[fut], pred[fut, k], color=RED, lw=1.8, label='Forecast', zorder=2)
        ax.scatter(t_obs, xy_obs[:, k], s=18, color='black', label='Observations', zorder=3)
        ax.axvline(context_end, color='gray', ls=':', lw=1.2); ax.text(context_end, ax.get_ylim()[1], ' forecast →', va='top', fontsize=8, color='gray')
        ax.set_xlabel('t'); ax.set_ylabel(name); ax.set_title(f'{name}(t): {title}'); ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); p = out_dir / f'{name}_vs_time.png'; fig.savefig(p, dpi=200); plt.close(fig); paths[name] = p
    return paths
