#!/usr/bin/env python3
"""Phase-2 controlled overfitting experiments (debugging only, NOT the research protocol).

Experiment 2A: one clean circle; 2B: one clean spiral; 2C: 10 circles + 10 spirals.
The encoder sees 30 CLEAN points from t<=7 (fixed indices); the loss is MSE on the
complete clean trajectory over t in [0, 10] (full-trajectory supervision is allowed
for this diagnostic). The posterior mean is used (no sampling) so that the test
isolates representational/optimisation capacity of encoder+ODE+decoder.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from glo.data import load_split
from glo.model import LatentODE
from glo.pipeline import seed_everything

ARCHS = {
    'original': dict(latent_dim=4, encoder_hidden=25, dynamics_hidden=20, decoder_hidden=20),
    'original_latent8': dict(latent_dim=8, encoder_hidden=25, dynamics_hidden=20, decoder_hidden=20),
    'candidate': dict(latent_dim=8, encoder_hidden=64, dynamics_hidden=64, decoder_hidden=64,
                      dynamics_layers=2, decoder_layers=2),
}

p = argparse.ArgumentParser()
p.add_argument('--experiment', choices=['circle', 'spiral', 'mixed20'], required=True)
p.add_argument('--arch', choices=list(ARCHS), default='original')
p.add_argument('--iters', type=int, default=3000)
p.add_argument('--lr', type=float, default=3e-3)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--solver', default='dopri5', choices=['dopri5', 'rk4'])
p.add_argument('--step-size', type=float, default=0.1)
p.add_argument('--out', default='results/debug_overfit')
p.add_argument('--device', default='auto')
p.add_argument('--threads', type=int, default=4)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
seed_everything(a.seed)
dev = torch.device(a_dev if (a_dev := a.device) != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
if dev.type == 'cpu': torch.set_num_threads(a.threads)
arr, meta = load_split(root / 'data/processed', 'two_shapes', 'train')
fam = arr['family']
if a.experiment == 'circle':
    idx = np.flatnonzero(fam == 0)[:1]
elif a.experiment == 'spiral':
    idx = np.flatnonzero(fam == 1)[:1]
else:
    idx = np.concatenate([np.flatnonzero(fam == 0)[:10], np.flatnonzero(fam == 1)[:10]])
mean = torch.tensor(meta['normalization_mean'], dtype=torch.float32)
scale = float(meta['normalization_scale'])
q = torch.arange(0, 200, 2)                                   # 100 query points on [0,10]
t = torch.tensor(arr['time'], dtype=torch.float32)[q]
xy = (torch.tensor(arr['xy'][idx])[:, q] - mean) / scale       # clean, normalised
ctx = torch.nonzero(t <= meta['forecast_start']).flatten()
obs_i = ctx[torch.linspace(0, len(ctx) - 1, 30).long()]        # 30 fixed clean context points
obs_xy, obs_t = xy[:, obs_i].to(dev), t[obs_i][None].repeat(len(idx), 1).to(dev)
xy, t = xy.to(dev), t.to(dev)
model = LatentODE(**ARCHS[a.arch], solver=a.solver,
                  step_size=a.step_size if a.solver == 'rk4' else None).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters, eta_min=a.lr / 50)
hist, t0 = [], time.time()
for it in range(a.iters):
    opt.zero_grad()
    out = model(obs_xy, obs_t, t, sample=False)
    loss = (out['prediction'] - xy).square().mean()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    hist.append(float(loss))
    if it % 500 == 0 or it == a.iters - 1:
        print(f'{a.experiment}/{a.arch} it={it} mse={float(loss):.6f} rmse_orig={float(loss)**.5*scale:.4f} grad={float(gn):.3f}', flush=True)
model.eval()
with torch.no_grad():
    pred = model(obs_xy, obs_t, t, sample=False)['prediction']
err = (pred - xy) * scale
cutoff = t <= meta['forecast_start']
res = {'experiment': a.experiment, 'arch': a.arch, 'solver': a.solver, 'model': ARCHS[a.arch], 'iters': a.iters,
       'n_traj': len(idx), 'seconds': round(time.time() - t0, 1),
       'rmse_full': float(err.square().mean().sqrt()),
       'rmse_context_t<=7': float(err[:, cutoff].square().mean().sqrt()),
       'rmse_future_t>7': float(err[:, ~cutoff].square().mean().sqrt()),
       'rmse_per_traj_max': float(err.square().mean(dim=(1, 2)).sqrt().max())}
print(json.dumps(res, indent=1))
out = root / a.out; out.mkdir(parents=True, exist_ok=True)
(out / f'{a.experiment}_{a.arch}.json').write_text(json.dumps(res, indent=1))
P = (pred.cpu() * scale + mean).numpy(); X = (xy.cpu() * scale + mean).numpy(); T = t.cpu().numpy()
show = [0] if len(idx) == 1 else [0, 1, 10, 11]
fig, ax = plt.subplots(len(show), 4, figsize=(17, 3.6 * len(show)), squeeze=False)
for r, k in enumerate(show):
    ax[r, 0].plot(X[k, :, 0], X[k, :, 1], color='#2a9469', lw=2.4, label='Ground truth')
    ax[r, 0].plot(P[k, :, 0], P[k, :, 1], '--', color='#2578bc', lw=1.6, label='Prediction')
    ax[r, 0].set_aspect('equal', adjustable='datalim'); ax[r, 0].legend(fontsize=7, frameon=False)
    ax[r, 0].set_title(f'{"circle" if fam[idx[k]]==0 else "spiral"} #{k}')
    for c, name in ((1, 'x'), (2, 'y')):
        ax[r, c].plot(T, X[k, :, c - 1], color='#2a9469', lw=2.2, label=f'{name}(t)')
        ax[r, c].plot(T, P[k, :, c - 1], '--', color='#2578bc', lw=1.6, label=f'{name}̂(t)')
        ax[r, c].axvline(meta['forecast_start'], color='gray', ls=':'); ax[r, c].set_xlabel('t')
        ax[r, c].legend(fontsize=7, frameon=False)
    ax[r, 3].semilogy(hist, color='#444'); ax[r, 3].set_title('training MSE (normalised)'); ax[r, 3].set_xlabel('iteration')
fig.suptitle(f'Overfit test {a.experiment} | arch={a.arch} | RMSE full={res["rmse_full"]:.4f}')
fig.tight_layout(); fig.savefig(out / f'{a.experiment}_{a.arch}.png', dpi=110); plt.close(fig)
