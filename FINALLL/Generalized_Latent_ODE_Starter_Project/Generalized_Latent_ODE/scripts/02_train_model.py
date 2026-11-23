#!/usr/bin/env python3
"""Train the variational Latent ODE on the 2- or 3-shape dataset.

Defaults reproduce the original configuration. Additions (2026-11-15, see
reports/implementation_changes.md): configurable depth, KL annealing, LR schedule,
early stopping, per-epoch CSV history (NLL / KL / validation forecast RMSE / LR /
gradient norms) and truthful solver metadata in the checkpoint.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from glo.model import LatentODE
from glo.pipeline import (TrajectoryDataset, create_loader, make_query_indices, prepare_batch,
                          batch_objective, save_checkpoint, seed_everything)
from glo.solver import solver_status, HAS_TORCHDIFFEQ

p = argparse.ArgumentParser(description='Train variational Latent ODE on 2 or 3 shape families')
p.add_argument('--config', default='', help='YAML file of argument defaults (keys use underscores); CLI flags override it')
p.add_argument('--families', nargs='+', default=None, help='Train on a subset of the dataset families, e.g. circle')
p.add_argument('--experiment', choices=['two_shapes', 'three_shapes', 'five_shapes', 'geolife_regular'], default='two_shapes')
p.add_argument('--data-root', default='data/processed', help='Folder containing <experiment>/{train,val}.npz (relative to the project root)')
p.add_argument('--init-checkpoint', default='', help='Initialise all weights from this checkpoint (fine-tuning); architecture must match')
p.add_argument('--epochs', type=int, default=100)
p.add_argument('--batch-size', type=int, default=32)
p.add_argument('--n-query', type=int, default=100)
p.add_argument('--n-obs', type=int, default=30)
p.add_argument('--n-obs-min', type=int, default=0, help='If >0 (with --n-obs-max), each TRAINING batch uses a random n_obs in [min, max]; validation keeps --n-obs')
p.add_argument('--n-obs-max', type=int, default=0)
p.add_argument('--latent-dim', type=int, default=4)
p.add_argument('--encoder-hidden', type=int, default=25)
p.add_argument('--dynamics-hidden', type=int, default=20)
p.add_argument('--decoder-hidden', type=int, default=20)
p.add_argument('--dynamics-layers', type=int, default=1)
p.add_argument('--decoder-layers', type=int, default=1)
p.add_argument('--cond-dim', type=int, default=0, help='0=unconditioned baseline, 4=conditional extension')
p.add_argument('--supervision', choices=['observed', 'full_clean'], default='observed')
p.add_argument('--sigma', type=float, default=.05)
p.add_argument('--kl-weight', type=float, default=1.)
p.add_argument('--kl-anneal-epochs', type=int, default=0,
               help='Linearly ramp beta from 0 to --kl-weight over this many epochs (0 = off)')
p.add_argument('--lr', type=float, default=.001)
p.add_argument('--lr-schedule', choices=['none', 'cosine'], default='none')
p.add_argument('--min-lr', type=float, default=1e-5)
p.add_argument('--grad-clip', type=float, default=1.0)
p.add_argument('--patience', type=int, default=0, help='Early-stopping patience in epochs (0 = off)')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--data-seed', type=int, default=None,
               help='If set, batch order and training observation subsets use dedicated generators seeded with this '
                    'value, so models of different sizes see identical training data at the same seed')
p.add_argument('--solver', choices=['dopri5', 'rk4'], default='dopri5')
p.add_argument('--step-size', type=float, default=None, help='Internal step for rk4')
p.add_argument('--rtol', type=float, default=1e-4)
p.add_argument('--atol', type=float, default=1e-5)
p.add_argument('--adjoint', action='store_true')
p.add_argument('--max-train-batches', type=int, default=0, help='For a quick integration test only')
p.add_argument('--max-val-batches', type=int, default=0, help='For a quick integration test only')
p.add_argument('--checkpoint', default='', help='Path relative to project root')
p.add_argument('--history', default='', help='CSV path for per-epoch history (default: next to checkpoint)')
p.add_argument('--device', default='auto', help='auto | cpu | cuda')
p.add_argument('--threads', type=int, default=4, help='CPU threads when --device cpu')
p.add_argument('--overwrite', action='store_true', help='Allow replacing an existing checkpoint file')
p.add_argument('--require-torchdiffeq', action='store_true',
               help='Abort instead of silently using the RK4 fallback')
pre, _ = p.parse_known_args()
if pre.config:
    import yaml
    cfg_path = Path(pre.config)
    cfg_path = cfg_path if cfg_path.is_absolute() else Path(__file__).resolve().parents[1] / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    known = {a_.dest for a_ in p._actions}
    bad = set(cfg) - known
    if bad:
        p.error(f'unknown keys in {pre.config}: {sorted(bad)}')
    p.set_defaults(**cfg)
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
if args.require_torchdiffeq and not HAS_TORCHDIFFEQ:
    p.error('torchdiffeq is not installed; refusing to fall back to offline RK4')
seed_everything(args.seed)
device = torch.device(a_dev if (a_dev := args.device) != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
if device.type == 'cpu': torch.set_num_threads(args.threads)
train_data = TrajectoryDataset(root / args.data_root, args.experiment, 'train', args.families)
val_data = TrajectoryDataset(root / args.data_root, args.experiment, 'val', args.families)
meta = train_data.meta
qidx = make_query_indices(meta['n_points'], args.n_query)
t = torch.tensor(train_data.arrays['time'], dtype=torch.float32)
if (t[qidx] <= meta['forecast_start']).sum().item() < max(args.n_obs, args.n_obs_max):
    p.error('n-query too small to sample n-obs before forecast cutoff')
model_config = dict(latent_dim=args.latent_dim, encoder_hidden=args.encoder_hidden,
                    dynamics_hidden=args.dynamics_hidden, decoder_hidden=args.decoder_hidden,
                    cond_dim=args.cond_dim, solver=args.solver, adjoint=args.adjoint,
                    dynamics_layers=args.dynamics_layers, decoder_layers=args.decoder_layers,
                    rtol=args.rtol, atol=args.atol, step_size=args.step_size)
model = LatentODE(**model_config).to(device)
if args.init_checkpoint:
    init = torch.load(root / args.init_checkpoint, map_location='cpu', weights_only=True)
    if init['model'] != model_config:
        p.error(f'--init-checkpoint architecture {init["model"]} differs from {model_config}')
    model.load_state_dict(init['state_dict'])
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.min_lr)
         if args.lr_schedule == 'cosine' else None)
loader_gen = torch.Generator().manual_seed(args.data_seed) if args.data_seed is not None else None
obs_gen = torch.Generator().manual_seed(args.data_seed + 1) if args.data_seed is not None else None
train_loader = create_loader(train_data, args.batch_size, shuffle=True, generator=loader_gen)
val_loader = create_loader(val_data, args.batch_size, shuffle=False)
config = {'model': model_config, 'training': vars(args), 'solver_backend': solver_status(),
          'n_parameters': sum(p_.numel() for p_ in model.parameters())}
ckpt = root / (args.checkpoint if args.checkpoint else f'checkpoints/{args.experiment}_cond{args.cond_dim}_best.pt')
if ckpt.exists() and not args.overwrite:
    p.error(f'refusing to overwrite existing checkpoint {ckpt} (pass --overwrite)')
hist_path = root / args.history if args.history else ckpt.with_name(ckpt.stem + '_history.csv')
hist_path.parent.mkdir(parents=True, exist_ok=True)
fields = ['epoch', 'beta', 'lr', 'train_loss', 'train_nll', 'train_kl', 'val_elbo_loss', 'val_nll',
          'val_kl', 'val_forecast_rmse', 'val_context_rmse', 'grad_norm_mean', 'grad_norm_max',
          'nfe_per_batch', 'seconds']
hist_file = open(hist_path, 'w', newline='')
writer = csv.DictWriter(hist_file, fieldnames=fields); writer.writeheader()
print('Solver:', solver_status(), '| device:', device, '| train:', len(train_data), 'val:', len(val_data),
      '| params:', config['n_parameters'], flush=True)
best, best_epoch, stale = float('inf'), 0, 0
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    beta = (args.kl_weight * min(1., epoch / args.kl_anneal_epochs) if args.kl_anneal_epochs
            else args.kl_weight)
    sums = {'loss': 0., 'nll': 0., 'kl': 0.}; grads = []; n = 0
    model.train(); model.dynamics.nfe = 0
    for j, batch in enumerate(train_loader):
        if args.max_train_batches and j >= args.max_train_batches: break
        n_obs_b = (int(torch.randint(args.n_obs_min, args.n_obs_max + 1, (1,))) if args.n_obs_min > 0
                   else args.n_obs)
        prepared = prepare_batch(batch, t, meta, qidx, n_obs_b, device, generator=obs_gen)
        opt.zero_grad(set_to_none=True)
        output = model(prepared['obs_xy'], prepared['obs_t'], prepared['query_t'], sample=True)
        loss, parts = batch_objective(output, prepared, args.sigma, beta, args.supervision)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        loss.backward()
        grads.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)))
        opt.step()
        for k in sums: sums[k] += parts[k]
        n += 1
    nfe = model.dynamics.nfe / max(n, 1)
    # Validation: posterior mean, fixed observation subsets, FULL ELBO (beta = kl_weight) so that
    # checkpoint selection is comparable across annealing epochs. Forecast RMSE uses clean val
    # targets (validation split only; never the test split).
    model.eval(); vs = {'loss': 0., 'nll': 0., 'kl': 0.}; vn = 0; fsq = csq = fcount = ccount = 0.
    val_generator = torch.Generator().manual_seed(args.seed + 999)
    with torch.no_grad():
        for j, batch in enumerate(val_loader):
            if args.max_val_batches and j >= args.max_val_batches: break
            prepared = prepare_batch(batch, t, meta, qidx, args.n_obs, device, generator=val_generator)
            output = model(prepared['obs_xy'], prepared['obs_t'], prepared['query_t'], sample=False)
            _, parts = batch_objective(output, prepared, args.sigma, args.kl_weight, args.supervision)
            for k in vs: vs[k] += parts[k]
            vn += 1
            d2 = ((output['prediction'] - prepared['truth']) * prepared['scale']).square()
            fut = prepared['query_t'] > meta['forecast_start']
            fsq += float(d2[:, fut].sum()); fcount += d2[:, fut].numel()
            csq += float(d2[:, ~fut].sum()); ccount += d2[:, ~fut].numel()
    row = {'epoch': epoch, 'beta': beta, 'lr': opt.param_groups[0]['lr'],
           'train_loss': sums['loss'] / n, 'train_nll': sums['nll'] / n, 'train_kl': sums['kl'] / n,
           'val_elbo_loss': vs['loss'] / vn, 'val_nll': vs['nll'] / vn, 'val_kl': vs['kl'] / vn,
           'val_forecast_rmse': math.sqrt(fsq / fcount), 'val_context_rmse': math.sqrt(csq / ccount),
           'grad_norm_mean': sum(grads) / len(grads), 'grad_norm_max': max(grads),
           'nfe_per_batch': nfe, 'seconds': time.time() - t0}
    writer.writerow(row); hist_file.flush()
    if sched: sched.step()
    val_loss = row['val_elbo_loss']
    print(f"epoch={epoch:03d} beta={beta:.3f} lr={row['lr']:.2e} train={row['train_loss']:.2f} "
          f"(nll={row['train_nll']:.2f} kl={row['train_kl']:.2f}) val={val_loss:.2f} "
          f"val_ctx_rmse={row['val_context_rmse']:.4f} val_fc_rmse={row['val_forecast_rmse']:.4f} "
          f"grad={row['grad_norm_mean']:.1f} nfe/b={nfe:.0f} {row['seconds']:.0f}s", flush=True)
    if val_loss < best:
        best, best_epoch, stale = val_loss, epoch, 0
        save_checkpoint(ckpt, model, meta, config, epoch, val_loss)
    else:
        stale += 1
        if args.patience and stale >= args.patience:
            print(f'Early stopping: no validation improvement for {args.patience} epochs'); break
hist_file.close()
print(f'Checkpoint: {ckpt} | best validation loss: {best:.4f} at epoch {best_epoch} | history: {hist_path}')
if args.max_train_batches or args.max_val_batches:
    print('WARNING: partial-batch smoke run. Results are not publication-ready.')
