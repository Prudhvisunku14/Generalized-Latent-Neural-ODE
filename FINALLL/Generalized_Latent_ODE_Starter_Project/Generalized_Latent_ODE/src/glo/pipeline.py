"""Dataloading, observation construction, training and evaluation utilities."""
from __future__ import annotations
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .data import load_split
from .model import kl_standard_normal


class TrajectoryDataset(Dataset):
    """`families` (optional list of names) keeps only those families; normalisation
    constants are left untouched (always those of `experiment`'s training split)."""
    def __init__(self, root, experiment, split, families=None):
        self.arrays, self.meta = load_split(root, experiment, split)
        if families:
            unknown = set(families) - set(self.meta['families'])
            if unknown:
                raise ValueError(f'families {sorted(unknown)} not in {experiment}')
            ids = [self.meta['family_to_id'][f] for f in families]
            keep = np.isin(self.arrays['family'], ids)
            self.arrays = {k: (v[keep] if k != 'time' else v) for k, v in self.arrays.items()}

    def __len__(self):
        return len(self.arrays['family'])

    def __getitem__(self, i):
        return {'xy': torch.from_numpy(self.arrays['xy'][i]),
                'noisy_xy': torch.from_numpy(self.arrays['noisy_xy'][i]),
                'family': int(self.arrays['family'][i])}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_query_indices(n_total, n_query):
    if not 2 <= n_query <= n_total:
        raise ValueError('n_query must be in [2, n_total]')
    return torch.linspace(0, n_total - 1, n_query).long().unique()


def prepare_batch(batch, time, meta, query_indices, n_obs, device, generator=None,
                  noise_std=None, use_stored_noise=True):
    """Construct time-sorted context points from t<=cutoff, without future leaks.

    The encoder receives noisy sparse context; clean full trajectories stay as
    supervision/evaluation targets. Every batch uses one shared ODE query grid.
    """
    query_indices = query_indices.cpu()
    t_query = time[query_indices]
    allowed = int((t_query <= meta['forecast_start']).sum().item())
    if not 2 <= n_obs <= allowed:
        raise ValueError(f'n_obs={n_obs}, but only {allowed} query times are <= cutoff')
    xy = batch['xy'][:, query_indices, :].float()
    if use_stored_noise and noise_std is None:
        noisy = batch['noisy_xy'][:, query_indices, :].float()
    else:
        stdev = meta['base_noise_std'] if noise_std is None else noise_std
        rand = torch.randn(xy.shape, generator=generator, dtype=xy.dtype)
        noisy = xy + stdev * rand
    bsize = xy.shape[0]
    ranks = torch.rand((bsize, allowed - 1), generator=generator)
    selected = torch.topk(ranks, k=n_obs-1, dim=-1).indices + 1
    selected = torch.cat([torch.zeros((bsize, 1), dtype=torch.long), selected], 1)
    selected = torch.sort(selected, dim=1).values
    obs_mask = torch.zeros((bsize, len(t_query)), dtype=torch.bool)
    obs_mask.scatter_(1, selected, True)
    row = torch.arange(bsize)[:, None]
    obs_xy = noisy[row, selected]
    obs_t = t_query[selected]
    mean = torch.tensor(meta['normalization_mean'], dtype=torch.float32)
    scale = float(meta['normalization_scale'])
    return {'truth': ((xy - mean) / scale).to(device),
            'obs_xy': ((obs_xy - mean) / scale).to(device),
            'obs_t': obs_t.to(device), 'query_t': t_query.to(device),
            'obs_indices': selected.to(device), 'obs_mask': obs_mask.to(device),
            'family': batch['family'].to(device), 'raw_truth': xy.to(device),
            'mean': mean.to(device), 'scale': scale}


def batch_objective(model_output, prepared, obs_sigma=.05, kl_weight=1., supervision='observed'):
    pred = model_output['prediction']
    if supervision == 'observed':
        row = torch.arange(pred.shape[0], device=pred.device)[:, None]
        loc = pred[row, prepared['obs_indices']]
        target = prepared['obs_xy']
    elif supervision == 'full_clean':
        loc, target = pred, prepared['truth']
    else:
        raise ValueError('supervision must be observed or full_clean')
    # Fixed-isotropic-Gaussian decoder. Report NLL as sum over scalar observations.
    sigma = max(float(obs_sigma) / prepared['scale'], 1e-4)
    nll = (.5 * ((loc-target)/sigma).square() + math.log(sigma) + .5*math.log(2*math.pi))
    nll = nll.sum(dim=(1,2)).mean()
    kl = kl_standard_normal(model_output['mu'], model_output['logvar'])
    loss = nll + kl_weight * kl
    return loss, {'loss': float(loss.detach()), 'nll': float(nll.detach()), 'kl': float(kl.detach())}


def save_checkpoint(path, model, meta, config, epoch, validation_loss):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'model': config['model'],
                'config': config, 'experiment': meta['dataset_name'],
                'epoch': epoch, 'val_loss': validation_loss}, path)


def create_loader(dataset, batch_size=32, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)


def infer_test_metrics(pred_norm, prepared, cutoff):
    """Per-trajectory reconstruction/forecast metrics, in ORIGINAL coordinate units."""
    pred = pred_norm * prepared['scale'] + prepared['mean']
    truth = prepared['raw_truth']
    query_t = prepared['query_t']
    future = query_t > cutoff
    context = query_t <= cutoff
    if not future.any():
        raise ValueError('No forecast timestamps in selected query grid')
    missing = context[None, :] & ~prepared['obs_mask']
    diffs = pred - truth
    sq = diffs.square().sum(-1)
    dist = sq.sqrt()
    b = pred.shape[0]
    mask_float = missing.float()
    nm = mask_float.sum(-1).clamp_min(1.)
    recon_rmse = ((diffs.square().sum(-1) * mask_float).sum(-1) / (2 * nm)).sqrt()
    future_sq = diffs[:, future, :].square()
    # Compute every trajectory independently to avoid batch-wide RMSE.
    forecast_rmse = future_sq.mean(dim=(1,2)).sqrt()
    ade = dist[:, future].mean(-1)
    fde = dist[:, future][:,-1]
    forecast_mae = diffs[:, future, :].abs().mean(dim=(1,2))
    # Both sets are ordered in time when computing RMSE/ADE/FDE; this geometric
    # supplemental metric intentionally ignores ordering.
    dmat = torch.cdist(pred, truth)
    chamfer = dmat.square().min(-1).values.mean(-1) + dmat.square().min(-2).values.mean(-1)
    return {'recon_rmse': recon_rmse, 'forecast_rmse': forecast_rmse, 'forecast_mae': forecast_mae,
            'forecast_ade': ade, 'forecast_fde': fde, 'chamfer_full': chamfer}
