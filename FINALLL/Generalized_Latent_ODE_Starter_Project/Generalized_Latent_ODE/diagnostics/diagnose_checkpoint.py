#!/usr/bin/env python3
"""Phase-1 audit: inspect a trained checkpoint for posterior collapse / dynamics collapse."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch, numpy as np
from glo.model import LatentODE, kl_standard_normal
from glo.pipeline import TrajectoryDataset, create_loader, make_query_indices, prepare_batch, batch_objective

p = argparse.ArgumentParser(); p.add_argument('--checkpoint', default='checkpoints/two_shapes_cond0_best.pt')
p.add_argument('--sigma', type=float, default=.05)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
ck = torch.load(root / a.checkpoint, map_location='cpu', weights_only=False)
cfg = dict(ck['model']); 
model = LatentODE(**cfg); model.load_state_dict(ck['state_dict']); model.eval()
data = TrajectoryDataset(root / 'data/processed', ck['experiment'], 'test')
meta = data.meta; q = make_query_indices(200, 200); t = torch.tensor(data.arrays['time'])
batch = next(iter(create_loader(data, 300)))
g = torch.Generator().manual_seed(0)
P = prepare_batch(batch, t, meta, q, 30, torch.device('cpu'), generator=g)
torch.manual_seed(0)
with torch.no_grad():
    out_mu = model(P['obs_xy'], P['obs_t'], P['query_t'], sample=False)
    out_s = model(P['obs_xy'], P['obs_t'], P['query_t'], sample=True)
    loss, parts = batch_objective(out_mu, P, a.sigma, 1.0, 'observed')
    mu, lv = out_mu['mu'], out_mu['logvar']
    pred = out_mu['prediction']; truth = P['truth']
    # dynamics magnitude along trajectories
    z0 = mu
    from glo.solver import integrate
    zt = integrate(model.dynamics, z0, P['query_t'])
    dz = model.dynamics(0, zt).norm(dim=-1)
res = {
  'epoch': ck['epoch'], 'val_loss': ck['val_loss'],
  'mu_mean': mu.mean(0).tolist(), 'mu_std_across_traj': mu.std(0).tolist(),
  'posterior_sigma_mean': lv.mul(.5).exp().mean(0).tolist(),
  'kl_per_traj': float(kl_standard_normal(mu, lv)), 'nll_per_traj': parts['nll'],
  'pred_norm_coord_std': pred.std(dim=(0,1)).tolist(), 'truth_norm_coord_std': truth.std(dim=(0,1)).tolist(),
  'pred_temporal_std_mean': pred.std(1).mean(0).tolist(), 'truth_temporal_std_mean': truth.std(1).mean(0).tolist(),
  'rmse_mean_post': float((pred-truth).pow(2).mean().sqrt()*P['scale']),
  'rmse_sampled_post': float((out_s['prediction']-truth).pow(2).mean().sqrt()*P['scale']),
  'rmse_predict_zero(dataset mean)': float(truth.pow(2).mean().sqrt()*P['scale']),
  'dynamics_norm_t0_t5_t10': [float(dz[0].mean()), float(dz[100].mean()), float(dz[-1].mean())],
  'latent_norm_t0_t10': [float(zt[0].norm(dim=-1).mean()), float(zt[-1].norm(dim=-1).mean())],
}
print(json.dumps(res, indent=1))
