"""Shared evaluation helpers for the validation study.

Key rule: a model's inputs are always normalised with the normalisation constants of the
dataset the model was TRAINED on (its checkpoint's `experiment`), whatever dataset it is
evaluated on. This matters when a two-shape model is evaluated on three-shape data
(out-of-distribution ellipses) or when models trained on different datasets are compared.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import torch
from .model import LatentODE
from .pipeline import TrajectoryDataset, create_loader, make_query_indices, prepare_batch, infer_test_metrics

METRICS = ['recon_rmse', 'forecast_rmse', 'forecast_mae', 'forecast_ade', 'forecast_fde', 'chamfer_full']


def load_model(path, device, tight_dopri=True):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = LatentODE(**ck['model']).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    if tight_dopri and model.solver == 'dopri5':
        model.rtol, model.atol = 1e-5, 1e-6
    return model, ck


def training_meta(data_root, experiment):
    return json.loads((Path(data_root) / experiment / 'metadata.json').read_text())


def collect(model, ck, data_root, data_experiment, split='test', families=None, n_obs=30,
            n_query=200, noise_std=None, seed=123, batch_size=50, device='cpu', sample=False):
    """Run `model` on a split; return per-trajectory arrays in ORIGINAL coordinates.

    noise_std=None uses the stored noisy observations; a number draws fresh noise of that std
    (seeded), so different models evaluated with the same seed see identical observations.
    """
    data = TrajectoryDataset(data_root, data_experiment, split, families)
    meta = dict(data.meta)
    train_meta = training_meta(data_root, ck['experiment'])
    meta['normalization_mean'] = train_meta['normalization_mean']
    meta['normalization_scale'] = train_meta['normalization_scale']
    qidx = make_query_indices(meta['n_points'], n_query)
    time = torch.tensor(data.arrays['time'], dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    out = {k: [] for k in ['pred', 'truth', 'obs', 'obs_t', 'obs_mask', 'family'] + METRICS}
    with torch.no_grad():
        for batch in create_loader(data, batch_size=batch_size, shuffle=False):
            P = prepare_batch(batch, time, meta, qidx, n_obs, device, generator=gen, noise_std=noise_std,
                              use_stored_noise=noise_std is None)
            pred_n = model(P['obs_xy'], P['obs_t'], P['query_t'], sample=sample)['prediction']
            mets = infer_test_metrics(pred_n, P, meta['forecast_start'])
            out['pred'].append((pred_n * P['scale'] + P['mean']).cpu())
            out['truth'].append(P['raw_truth'].cpu())
            out['obs'].append((P['obs_xy'] * P['scale'] + P['mean']).cpu())
            out['obs_t'].append(P['obs_t'].cpu())
            out['obs_mask'].append(P['obs_mask'].cpu())
            out['family'].append(P['family'].cpu())
            for k in METRICS:
                out[k].append(mets[k].cpu())
    res = {k: torch.cat(v).numpy() for k, v in out.items()}
    res['query_t'] = time[qidx].numpy()
    res['params'] = data.arrays['params']
    res['ids'] = data.arrays['ids']
    res['meta'] = meta
    return res


def summarize(res, mask=None):
    m = np.ones(len(res['family']), bool) if mask is None else mask
    return {'n': int(m.sum()), **{k: {'mean': float(res[k][m].mean()), 'std': float(res[k][m].std()),
                                      'median': float(np.median(res[k][m]))} for k in METRICS}}


def by_family(res):
    fam_names = {v: k for k, v in res['meta']['family_to_id'].items()}
    out = {'overall': summarize(res)}
    for fid in np.unique(res['family']):
        out[fam_names[int(fid)]] = summarize(res, res['family'] == fid)
    return out


def t_ci95(values):
    """Mean, sample std (ddof=1) and two-sided 95% Student-t interval over independent runs."""
    v = np.asarray(values, float)
    n = len(v)
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}[n]
    sd = float(v.std(ddof=1))
    half = tcrit * sd / math.sqrt(n)
    return {'n_runs': n, 'mean': float(v.mean()), 'std': sd, 'ci95_low': float(v.mean() - half),
            'ci95_high': float(v.mean() + half), 'values': [float(x) for x in v]}


def geometry(res):
    """Per-trajectory geometric errors relative to the KNOWN generating centre/parameters.

    Valid for circles and spirals (polar motion about (cx, cy)). For ellipses the polar
    radius is not constant, so only angular velocity and direction are meaningful.
    """
    pred, qt, prm = res['pred'], res['query_t'], res['params']
    c = prm[:, 3:5][:, None, :]
    R0, omega, growth = prm[:, 0], prm[:, 1], prm[:, 5]
    fut = qt > res['meta']['forecast_start']
    rel = pred - c
    r = np.linalg.norm(rel, axis=-1)
    ang = np.unwrap(np.arctan2(rel[..., 1], rel[..., 0]), axis=1)
    true_r = R0[:, None] * np.exp(growth[:, None] * qt[None])

    def slope(y, sel):
        T = qt[sel] - qt[sel].mean()
        return ((y[:, sel] * T).sum(-1)) / (T ** 2).sum()

    allp = np.ones_like(fut)
    return {'omega_hat': slope(ang, allp), 'omega_hat_future': slope(ang, fut),
            'growth_hat': slope(np.log(np.clip(r, 1e-4, None)), allp),
            'growth_hat_context': slope(np.log(np.clip(r, 1e-4, None)), ~fut),
            'radius_rel_err_context': (np.abs(r[:, ~fut] - true_r[:, ~fut]) / true_r[:, ~fut]).mean(-1),
            'radius_rel_err_future': (np.abs(r[:, fut] - true_r[:, fut]) / true_r[:, fut]).mean(-1),
            'radius_rel_drift_end': (r[:, -1] - true_r[:, -1]) / true_r[:, -1],
            'omega': omega, 'growth': growth}


def horizon_error(res):
    """Euclidean error per query time, [n_traj, n_query]."""
    return np.linalg.norm(res['pred'] - res['truth'], axis=-1)


class BaselineModel(torch.nn.Module):
    """Observation-only baselines with the LatentODE call signature (normalised coordinates).

    Context interval: linear interpolation of the observations. Forecast: hold the last
    observation ('const_pos') or extrapolate the last two observations' velocity ('const_vel').
    Both are affine-equivariant, so normalisation does not change their predictions.
    """
    def __init__(self, kind):
        super().__init__()
        if kind not in ('const_pos', 'const_vel'):
            raise ValueError(kind)
        self.kind = kind

    def forward(self, obs_xy, obs_t, query_t, sample=False):
        B, Q = obs_xy.shape[0], len(query_t)
        qt = query_t.cpu().numpy()
        out = torch.empty(B, Q, 2)
        for i in range(B):
            for c in range(2):
                out[i, :, c] = torch.from_numpy(np.interp(qt, obs_t[i].cpu().numpy(), obs_xy[i, :, c].cpu().numpy()))
        after = torch.from_numpy(qt > obs_t[:, -1:].cpu().numpy())                     # [B, Q]
        last = obs_xy[:, -1].cpu()
        if self.kind == 'const_vel':
            vel = (obs_xy[:, -1] - obs_xy[:, -2]).cpu() / (obs_t[:, -1] - obs_t[:, -2]).clamp_min(1e-3).cpu()[:, None]
            dt = (query_t.cpu()[None, :] - obs_t[:, -1:].cpu()).clamp_min(0)
            extrap = last[:, None, :] + vel[:, None, :] * dt[..., None]
        else:
            extrap = last[:, None, :].expand(B, Q, 2)
        out = torch.where(after[..., None], extrap, out)
        return {'prediction': out.to(obs_xy.device)}


def baseline(kind, data_experiment):
    return BaselineModel(kind), {'experiment': data_experiment}


def encode(model, ck, data_root, data_experiment, split='test', families=None, n_obs=30, n_query=200,
           seed=123, batch_size=50, device='cpu'):
    """Posterior mean/log-variance and (if conditioned) dynamics code c for every trajectory, using the
    SAME observation subsets as `collect` with the same arguments. Also returns the normalised
    query grid and normalisation constants needed by `decode`."""
    data = TrajectoryDataset(data_root, data_experiment, split, families)
    meta = dict(data.meta)
    tm = training_meta(data_root, ck['experiment'])
    meta['normalization_mean'], meta['normalization_scale'] = tm['normalization_mean'], tm['normalization_scale']
    qidx = make_query_indices(meta['n_points'], n_query)
    time = torch.tensor(data.arrays['time'], dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    mus, lvs, cs = [], [], []
    with torch.no_grad():
        for batch in create_loader(data, batch_size=batch_size, shuffle=False):
            P = prepare_batch(batch, time, meta, qidx, n_obs, device, generator=gen)
            mu, lv, c = model.encoder(P['obs_xy'], P['obs_t'], P['query_t'][-1].clamp_min(1e-6))
            mus.append(mu.cpu()); lvs.append(lv.cpu()); cs.append(c.cpu() if c is not None else None)
    return {'mu': torch.cat(mus), 'logvar': torch.cat(lvs),
            'cond': torch.cat(cs) if cs[0] is not None else None,
            'query_t': time[qidx], 'family': data.arrays['family'], 'params': data.arrays['params'],
            'mean': torch.tensor(meta['normalization_mean'], dtype=torch.float32),
            'scale': float(meta['normalization_scale']), 'meta': meta}


def decode(model, z0, cond, query_t, mean, scale):
    """Integrate from chosen (z0, c) and decode; returns predictions in ORIGINAL coordinates [B, Q, 2]."""
    from .solver import integrate
    with torch.no_grad():
        state = torch.cat([z0, cond], -1) if cond is not None else z0
        zt = integrate(model.dynamics, state, query_t, method=model.solver, adjoint=False,
                       rtol=model.rtol, atol=model.atol, step_size=model.step_size)
        pred = model.decoder(zt[..., :model.latent_dim]).transpose(0, 1)
    return (pred * scale + mean).numpy()
