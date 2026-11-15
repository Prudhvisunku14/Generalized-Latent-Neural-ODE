#!/usr/bin/env python3
"""Section 5 reproduction: bi-directional spiral latent ODE and RNN baseline.

Produces the paper-facing outputs only:
  * table2.csv: predictive RMSE for n_obs = 30, 50, 100 (Table 2)
  * latent_ode_figure.png: Figure 8 (RNN, latent ODE reconstruction/extrapolation, latent trajectories)
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import add_common_args, apply_mode, default_dirs, get_device, load_ckpt, save_ckpt, set_seed  # noqa: E402

# Paper Sec. 6 "Setting tolerances": default 1.5e-8 was used for sequence modelling.
PAPER = dict(iterations=2000, rtol=1.5e-8, atol=1.5e-8, checkpoint_every=200, n_spirals=1000)
# ASSUMPTION: the iteration count is not stated in the paper (2000 = default of the authors' latent-ODE demo).
SMOKE = dict(iterations=12, rtol=1e-3, atol=1e-3, checkpoint_every=6, n_spirals=60)


def make_spiral_dataset(n=1000, train_points=100, extrap_points=100, global_points=500, noise_std=0.3, seed=0):
    """Generate the paper's synthetic clockwise/counter-clockwise spiral family.

    The analytic spiral follows the authors' historical public latent_ode example.
    Each trajectory uses a different contiguous starting point; the first 100
    points form the observation interval and the next 100 form extrapolation.
    """
    rng = np.random.RandomState(seed)
    theta = np.linspace(0.0, 6.0 * np.pi, global_points)
    a, b = 0.0, 0.3

    zcw = 6.0 * np.pi + 1.0 - theta
    rcw = a + b * 50.0 / zcw
    cw = np.stack([rcw * np.cos(zcw) - 5.0, rcw * np.sin(zcw)], axis=1)

    rcc = a + b * theta
    cc = np.stack([rcc * np.cos(theta) + 5.0, rcc * np.sin(theta)], axis=1)

    total = train_points + extrap_points
    lo = train_points
    hi = global_points - total
    if hi <= lo:
        lo = 0
        hi = global_points - total

    clean, noisy, directions = [], [], []
    for i in range(n):
        start = rng.randint(lo, hi + 1)
        is_cc = rng.rand() > 0.5
        base = cc if is_cc else cw
        traj = base[start : start + total].copy()
        obs = traj.copy()
        obs[:train_points] += rng.randn(train_points, 2) * noise_std
        clean.append(traj)
        noisy.append(obs)
        directions.append(1 if is_cc else 0)

    clean = torch.tensor(np.stack(clean), dtype=torch.float32)
    noisy = torch.tensor(np.stack(noisy), dtype=torch.float32)
    directions = torch.tensor(directions, dtype=torch.long)
    # The model sees a normalized local timeline; equal spacing is the key paper property.
    times = torch.linspace(0.0, 2.0, total)
    return clean, noisy, times, directions


class RecognitionRNN(nn.Module):
    """25-unit RNN consuming the observations backwards in time (paper Sec. 5 / 5.1)."""

    def __init__(self, latent_dim=4, obs_dim=2, hidden=25):
        super().__init__()
        self.hidden = hidden
        # ASSUMPTION: for irregularly sampled observations the paper does not say how the encoder sees the
        # time stamps; we feed the gap to the previously consumed (later) observation as one extra input.
        self.i2h = nn.Linear(obs_dim + 1 + hidden, hidden)
        self.h2o = nn.Linear(hidden, latent_dim * 2)

    def forward(self, x_obs, t_obs):
        B, N, _ = x_obs.shape
        h = torch.zeros(B, self.hidden, device=x_obs.device, dtype=x_obs.dtype)
        out = None
        step = t_obs[1] - t_obs[0] if N > 1 else torch.tensor(1.0, device=x_obs.device)
        for i in reversed(range(N)):
            gap = (t_obs[i + 1] - t_obs[i]) if i + 1 < N else torch.zeros((), device=x_obs.device)
            gap = (gap / (10.0 * 0.01)).expand(B, 1)   # gap in units of 10 base time steps (dt_base = 0.01)
            h = torch.tanh(self.i2h(torch.cat([x_obs[:, i], gap, h], dim=1)))
            out = self.h2o(h)
        return out


class LatentDynamics(nn.Module):
    def __init__(self, latent_dim=4, hidden=20):
        super().__init__()
        # One hidden layer with 20 units, matching Section 5.1 text.
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.Tanh(), nn.Linear(hidden, latent_dim))

    def forward(self, t, z):
        return self.net(z)


class Decoder(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, hidden=20):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, obs_dim))

    def forward(self, z):
        return self.net(z)


class LatentODE(nn.Module):
    def __init__(self, latent_dim=4, rec_hidden=25, dyn_hidden=20, dec_hidden=20, rtol=1e-5, atol=1e-5):
        super().__init__()
        self.latent_dim = latent_dim
        self.rec = RecognitionRNN(latent_dim, 2, rec_hidden)
        self.func = LatentDynamics(latent_dim, dyn_hidden)
        self.dec = Decoder(latent_dim, 2, dec_hidden)
        self.rtol = rtol
        self.atol = atol

    def infer(self, x_obs, t_obs):
        out = self.rec(x_obs, t_obs)
        return out[:, : self.latent_dim], out[:, self.latent_dim :]

    def solve(self, z0, query_times):
        zt = odeint_adjoint(self.func, z0, query_times, rtol=self.rtol, atol=self.atol, method="dopri5")
        return zt.permute(1, 0, 2)

    def decode_at(self, z0, query_times):
        return self.dec(self.solve(z0, query_times))


class RNNBaseline(nn.Module):
    """25-hidden-unit recurrent baseline for extrapolation."""
    def __init__(self, hidden=25):
        super().__init__()
        self.cell = nn.GRUCell(3, hidden)
        self.out = nn.Linear(hidden, 2)
        self.hidden = hidden

    def transition_loss(self, x_obs, t_obs):
        h = torch.zeros(x_obs.shape[0], self.hidden, device=x_obs.device, dtype=x_obs.dtype)
        losses = []
        for i in range(x_obs.shape[1] - 1):
            dt = (t_obs[i + 1] - t_obs[i]).expand(x_obs.shape[0], 1)
            h = self.cell(torch.cat([x_obs[:, i], dt], dim=1), h)
            pred = self.out(h)
            losses.append((pred - x_obs[:, i + 1]).square().sum(-1))
        return torch.stack(losses, dim=1).mean()

    @torch.no_grad()
    def extrapolate(self, x_obs, t_obs, t_future):
        h = torch.zeros(x_obs.shape[0], self.hidden, device=x_obs.device, dtype=x_obs.dtype)
        for i in range(x_obs.shape[1] - 1):
            dt = (t_obs[i + 1] - t_obs[i]).expand(x_obs.shape[0], 1)
            h = self.cell(torch.cat([x_obs[:, i], dt], dim=1), h)
        current_x = x_obs[:, -1]
        current_t = t_obs[-1]
        preds = []
        for tq in t_future:
            dt = (tq - current_t).expand(x_obs.shape[0], 1)
            h = self.cell(torch.cat([current_x, dt], dim=1), h)
            current_x = self.out(h)
            preds.append(current_x)
            current_t = tq
        return torch.stack(preds, dim=1)


def gaussian_logprob(x, mean, std):
    return (-0.5 * (((x - mean) / std) ** 2 + 2.0 * math.log(std) + math.log(2.0 * math.pi))).sum(-1)


def kl_standard_normal(mu, logvar):
    return 0.5 * (torch.exp(logvar) + mu.square() - 1.0 - logvar).sum(-1)


def choose_obs_indices(n_obs, train_points, rng, device):
    if n_obs >= train_points:
        idx = np.arange(train_points)
    else:
        idx = np.sort(rng.choice(train_points, size=n_obs, replace=False))
    return torch.tensor(idx, dtype=torch.long, device=device)


def _solve_at_obs(model, z0, times, t_obs):
    """Decode at the observation times; z0 lives at times[0]."""
    if bool(t_obs[0] == times[0]):
        return model.decode_at(z0, t_obs)
    return model.decode_at(z0, torch.cat([times[:1], t_obs]))[:, 1:]


def train_latent(model, noisy, times, n_obs, args, device, ckpt):
    opt = torch.optim.Adam(model.parameters(), lr=args.latent_lr)
    start, complete, _ = load_ckpt(ckpt, model, opt, device)
    if complete:
        return model
    rng = np.random.RandomState(args.seed + n_obs + start)
    N = noisy.shape[0]
    report = max(1, args.iterations // 10)
    for step in range(start, args.iterations):
        batch_idx = torch.tensor(rng.choice(N, size=min(args.batch_size, N), replace=False), device=device)
        obs_idx = choose_obs_indices(n_obs, args.train_points, rng, device)
        x_obs = noisy[batch_idx][:, obs_idx]
        t_obs = times[obs_idx]
        mu, logvar = model.infer(x_obs, t_obs)
        z0 = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        pred_obs = _solve_at_obs(model, z0, times, t_obs)
        logpx = gaussian_logprob(x_obs, pred_obs, args.noise_std).sum(1)
        kl = kl_standard_normal(mu, logvar)
        loss = (-logpx + kl).mean()                     # negative ELBO
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % report == 0 or step == start:
            print(f"LatentODE n={n_obs} step={step+1}/{args.iterations} negELBO={loss.item():.4f}", flush=True)
        if (step + 1) % args.checkpoint_every == 0 and step + 1 < args.iterations:
            save_ckpt(ckpt, model, opt, step + 1, False)
    save_ckpt(ckpt, model, opt, args.iterations, True)
    return model


def train_rnn(model, noisy, times, n_obs, args, device, ckpt):
    opt = torch.optim.Adam(model.parameters(), lr=args.rnn_lr)
    start, complete, _ = load_ckpt(ckpt, model, opt, device)
    if complete:
        return model
    rng = np.random.RandomState(args.seed + 1000 + n_obs + start)
    N = noisy.shape[0]
    report = max(1, args.iterations // 10)
    for step in range(start, args.iterations):
        batch_idx = torch.tensor(rng.choice(N, size=min(args.batch_size, N), replace=False), device=device)
        obs_idx = choose_obs_indices(n_obs, args.train_points, rng, device)
        loss = model.transition_loss(noisy[batch_idx][:, obs_idx], times[obs_idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % report == 0 or step == start:
            print(f"RNN n={n_obs} step={step+1}/{args.iterations} loss={loss.item():.4f}", flush=True)
        if (step + 1) % args.checkpoint_every == 0 and step + 1 < args.iterations:
            save_ckpt(ckpt, model, opt, step + 1, False)
    save_ckpt(ckpt, model, opt, args.iterations, True)
    return model


@torch.no_grad()
def evaluate_latent(model, noisy, clean, times, n_obs, args, device):
    """RMSE of the 100 time points beyond the observation window against the noise-free spiral.

    ASSUMPTION: the paper does not say whether RMSE is measured against noisy or noise-free data;
    noise-free ground truth is used.  The posterior mean of z(t0) is decoded (no sampling).
    """
    rng = np.random.RandomState(args.seed + 2000 + n_obs)
    obs_idx = choose_obs_indices(n_obs, args.train_points, rng, device)
    mu, _ = model.infer(noisy[:, obs_idx], times[obs_idx])
    pred_all = model.decode_at(mu, times)
    rmse = torch.sqrt(torch.mean((pred_all[:, args.train_points:] - clean[:, args.train_points:]) ** 2)).item()
    return rmse, obs_idx, pred_all, mu


@torch.no_grad()
def evaluate_rnn(model, noisy, clean, times, n_obs, args, device):
    rng = np.random.RandomState(args.seed + 3000 + n_obs)
    obs_idx = choose_obs_indices(n_obs, args.train_points, rng, device)
    pred_future = model.extrapolate(noisy[:, obs_idx], times[obs_idx], times[args.train_points:])
    rmse = torch.sqrt(torch.mean((pred_future - clean[:, args.train_points:]) ** 2)).item()
    return rmse, obs_idx, pred_future


@torch.no_grad()
def make_figure(model_latent, model_rnn, noisy, clean, times, directions, obs_idx, pred_latent, mu, args, out_path):
    """Figure 8: (a) RNN, (b) latent ODE (two spirals each), (c) latent trajectories."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    chosen = [int(torch.where(directions == 0)[0][0]), int(torch.where(directions == 1)[0][0])]
    t_obs = times[obs_idx]
    GREEN, BLUE, RED = "#2ca02c", "#1f77b4", "#d62728"
    fig = plt.figure(figsize=(5.2, 6.4), dpi=200)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.28, wspace=0.05)
    titles = ["(a) Recurrent Neural Network", "(b) Latent Neural Ordinary Differential Equation"]
    for r in range(2):
        for j, idx in enumerate(chosen):
            ax = fig.add_subplot(gs[r, j])
            gt = clean[idx].cpu().numpy()
            ob = noisy[idx, obs_idx].cpu().numpy()
            ax.plot(gt[:, 0], gt[:, 1], color=GREEN, lw=0.8)
            ax.scatter(ob[:, 0], ob[:, 1], s=3, color=GREEN)
            if r == 0:
                rf = model_rnn.extrapolate(noisy[idx : idx + 1, obs_idx], t_obs, times[args.train_points:])[0].cpu().numpy()
                ax.plot(rf[:, 0], rf[:, 1], color=RED, lw=0.9)
            else:
                lp = pred_latent[idx].cpu().numpy()
                ax.plot(lp[: args.train_points + 1, 0], lp[: args.train_points + 1, 1], color=BLUE, lw=0.9)
                ax.plot(lp[args.train_points:, 0], lp[args.train_points:, 1], color=RED, lw=0.9)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_title(titles[r], fontsize=6.5, loc="left")
    ax = fig.add_subplot(gs[2, :])
    zt = model_latent.solve(mu, times[: args.train_points]).cpu().numpy()
    d = directions.cpu().numpy()
    for i in range(min(200, zt.shape[0])):
        ax.plot(zt[i, :, 0], zt[i, :, 1], lw=0.4, alpha=0.6, color=GREEN if d[i] == 1 else "#17becf")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(c) Latent Trajectories (first two latent dims)", fontsize=6.5, loc="left")
    fig.legend([Line2D([], [], color=GREEN), Line2D([], [], color=GREEN, marker="o", ls="", ms=3),
                Line2D([], [], color=BLUE), Line2D([], [], color=RED)],
               ["Ground Truth", "Observation", "Prediction", "Extrapolation"],
               loc="lower center", ncol=4, fontsize=6, frameon=False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--rtol", type=float, default=None)
    p.add_argument("--atol", type=float, default=None)
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--n-spirals", type=int, default=None)
    # ASSUMPTION: optimiser settings are not stated in the paper (values follow the authors' public demo).
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latent-lr", type=float, default=1e-2)
    p.add_argument("--rnn-lr", type=float, default=1e-2)
    p.add_argument("--noise-std", type=float, default=0.3)
    p.add_argument("--train-points", type=int, default=100)
    p.add_argument("--extrap-points", type=int, default=100)
    args = p.parse_args()
    apply_mode(args, PAPER, SMOKE)
    default_dirs(args, "latent_ode", "latent_ode")
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir, ckpt_dir = Path(args.output_dir), Path(args.checkpoint_dir)
    print(f"[latent_ode] mode={args.mode} device={device} iterations={args.iterations} tol={args.rtol}")

    clean, noisy, times, directions = make_spiral_dataset(
        n=args.n_spirals, train_points=args.train_points, extrap_points=args.extrap_points,
        noise_std=args.noise_std, seed=args.seed)
    clean, noisy, times, directions = clean.to(device), noisy.to(device), times.to(device), directions.to(device)
    split = int(0.8 * clean.shape[0])           # ASSUMPTION: 80/20 train/test split (not stated)
    train_noisy = noisy[:split]
    test_clean, test_noisy, test_dir = clean[split:], noisy[split:], directions[split:]

    table, fig_pack = [], None
    for n_obs in (30, 50, 100):
        latent = LatentODE(rtol=args.rtol, atol=args.atol).to(device)
        rnn = RNNBaseline(hidden=25).to(device)
        latent = train_latent(latent, train_noisy, times, n_obs, args, device, ckpt_dir / f"latent_ode_obs{n_obs}.pt")
        rnn = train_rnn(rnn, train_noisy, times, n_obs, args, device, ckpt_dir / f"rnn_obs{n_obs}.pt")
        lat_rmse, obs_idx, pred_lat, mu = evaluate_latent(latent, test_noisy, test_clean, times, n_obs, args, device)
        rnn_rmse, _, _ = evaluate_rnn(rnn, test_noisy, test_clean, times, n_obs, args, device)
        table.append({"# Observations": f"{n_obs}/100", "RNN": round(rnn_rmse, 4), "Latent ODE": round(lat_rmse, 4)})
        print(f"[latent_ode] n={n_obs}: RNN RMSE={rnn_rmse:.4f}  Latent ODE RMSE={lat_rmse:.4f}", flush=True)
        if n_obs == 30:
            fig_pack = (latent, rnn, obs_idx, pred_lat, mu)

    with open(out_dir / "table2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["# Observations", "RNN", "Latent ODE"])
        w.writeheader()
        w.writerows(table)
    latent, rnn, obs_idx, pred_lat, mu = fig_pack
    make_figure(latent, rnn, test_noisy, test_clean, times, test_dir, obs_idx, pred_lat, mu, args,
                out_dir / "latent_ode_figure.png")
    print(f"[latent_ode] wrote {out_dir / 'table2.csv'} and latent_ode_figure.png")


if __name__ == "__main__":
    main()
