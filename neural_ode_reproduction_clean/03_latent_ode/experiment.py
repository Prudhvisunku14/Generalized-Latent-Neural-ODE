#!/usr/bin/env python3
"""Experiment 3: RNN versus Latent ODE on irregularly observed spiral trajectories.

Outputs:
  * Table-2-style RMSE comparison for 30/50/100 observations.
  * High-resolution PNG/PDF trajectory comparisons for all three observation counts.
  * Latent-space PNG/PDF and GIF visualizations.
  * A trajectory GIF for the 30-observation case.

The implementation keeps the architecture stated in Section 5.1 of the paper:
25-unit recognition RNN, 4-D latent state, one 20-unit hidden layer for latent
dynamics, and one 20-unit hidden layer in the decoder.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    ProgressTimer,
    add_common_args,
    append_csv,
    apply_mode,
    default_dirs,
    device_summary,
    format_duration,
    get_device,
    gpu_peak_memory_mb,
    load_ckpt,
    maybe_remove_checkpoint,
    reset_gpu_peak_memory,
    save_ckpt,
    save_system_info,
    set_seed,
    utc_timestamp,
    write_csv,
)

PAPER = dict(
    iterations=2000,
    rtol=1.5e-8,
    atol=1.5e-8,
    checkpoint_every=200,
    n_spirals=1000,
    steps_per_epoch=20,
)
SMOKE = dict(
    iterations=12,
    rtol=1e-3,
    atol=1e-3,
    checkpoint_every=6,
    n_spirals=60,
    steps_per_epoch=4,
)

PAPER_RMSE = {
    30: {"RNN": 0.3937, "Latent ODE": 0.1642},
    50: {"RNN": 0.3202, "Latent ODE": 0.1502},
    100: {"RNN": 0.1813, "Latent ODE": 0.1346},
}


def make_spiral_dataset(n: int = 1000, train_points: int = 100, extrap_points: int = 100,
                        global_points: int = 500, noise_std: float = 0.3,
                        seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate clockwise/counter-clockwise Archimedean spiral trajectories.

    Each example is a different contiguous segment of a longer spiral.  Observation
    noise is added only to the first ``train_points``; the future interval remains
    clean so extrapolation RMSE has an unambiguous ground truth.
    """

    rng = np.random.RandomState(seed)
    theta = np.linspace(0.0, 6.0 * np.pi, global_points)
    a, b = 0.0, 0.3

    z_cw = 6.0 * np.pi + 1.0 - theta
    r_cw = a + b * 50.0 / z_cw
    cw = np.stack([r_cw * np.cos(z_cw) - 5.0, r_cw * np.sin(z_cw)], axis=1)

    r_ccw = a + b * theta
    ccw = np.stack([r_ccw * np.cos(theta) + 5.0, r_ccw * np.sin(theta)], axis=1)

    total = train_points + extrap_points
    if total > global_points:
        raise ValueError("train_points + extrap_points must be <= global_points")
    low = min(train_points, global_points - total)
    high = global_points - total

    clean, noisy, directions = [], [], []
    for _ in range(n):
        start = rng.randint(low, high + 1) if high > low else low
        is_ccw = bool(rng.rand() > 0.5)
        base = ccw if is_ccw else cw
        trajectory = base[start:start + total].copy()
        observed = trajectory.copy()
        observed[:train_points] += rng.randn(train_points, 2) * noise_std
        clean.append(trajectory)
        noisy.append(observed)
        directions.append(1 if is_ccw else 0)

    clean_t = torch.tensor(np.stack(clean), dtype=torch.float32)
    noisy_t = torch.tensor(np.stack(noisy), dtype=torch.float32)
    directions_t = torch.tensor(directions, dtype=torch.long)
    # A normalized local clock is sufficient because all generated trajectories share
    # the same physical spacing.  Irregularity comes from subsampling this grid.
    times = torch.linspace(0.0, 2.0, total)
    return clean_t, noisy_t, times, directions_t


class RecognitionRNN(nn.Module):
    """25-unit RNN that consumes observed values backwards in time."""

    def __init__(self, latent_dim: int = 4, obs_dim: int = 2, hidden: int = 25):
        super().__init__()
        self.hidden = hidden
        self.i2h = nn.Linear(obs_dim + hidden, hidden)
        self.h2o = nn.Linear(hidden, latent_dim * 2)

    def forward(self, x_obs: torch.Tensor) -> torch.Tensor:
        batch, n_obs, _ = x_obs.shape
        h = torch.zeros(batch, self.hidden, device=x_obs.device, dtype=x_obs.dtype)
        out = None
        for i in reversed(range(n_obs)):
            h = torch.tanh(self.i2h(torch.cat([x_obs[:, i], h], dim=1)))
            out = self.h2o(h)
        if out is None:
            raise ValueError("RecognitionRNN received zero observations")
        return out


class LatentDynamics(nn.Module):
    def __init__(self, latent_dim: int = 4, hidden: int = 20):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.Tanh(), nn.Linear(hidden, latent_dim))
        self.nfe = 0

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        self.nfe += 1
        return self.net(z)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 4, obs_dim: int = 2, hidden: int = 20):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, obs_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LatentODE(nn.Module):
    def __init__(self, latent_dim: int = 4, rec_hidden: int = 25, dyn_hidden: int = 20,
                 dec_hidden: int = 20, rtol: float = 1e-5, atol: float = 1e-5):
        super().__init__()
        self.latent_dim = latent_dim
        self.rec = RecognitionRNN(latent_dim, 2, rec_hidden)
        self.func = LatentDynamics(latent_dim, dyn_hidden)
        self.dec = Decoder(latent_dim, 2, dec_hidden)
        self.rtol = float(rtol)
        self.atol = float(atol)

    def infer(self, x_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.rec(x_obs)
        return out[:, :self.latent_dim], out[:, self.latent_dim:]

    def solve(self, z0: torch.Tensor, query_times: torch.Tensor) -> torch.Tensor:
        zt = odeint_adjoint(self.func, z0, query_times, rtol=self.rtol, atol=self.atol, method="dopri5")
        return zt.permute(1, 0, 2)

    def decode_at(self, z0: torch.Tensor, query_times: torch.Tensor) -> torch.Tensor:
        return self.dec(self.solve(z0, query_times))


class RNNBaseline(nn.Module):
    """25-hidden-unit recurrent baseline; optionally receives time gaps."""

    def __init__(self, hidden: int = 25, use_dt: bool = True):
        super().__init__()
        self.use_dt = use_dt
        self.cell = nn.GRUCell(3 if use_dt else 2, hidden)
        self.out = nn.Linear(hidden, 2)
        self.hidden = hidden

    def _cell_input(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, dt], dim=1) if self.use_dt else x

    def transition_nll(self, x_obs: torch.Tensor, t_obs: torch.Tensor, noise_std: float) -> torch.Tensor:
        h = torch.zeros(x_obs.shape[0], self.hidden, device=x_obs.device, dtype=x_obs.dtype)
        logps = []
        for i in range(x_obs.shape[1] - 1):
            dt = (t_obs[:, i + 1] - t_obs[:, i]).unsqueeze(1)
            h = self.cell(self._cell_input(x_obs[:, i], dt), h)
            pred = self.out(h)
            logps.append(gaussian_logprob(x_obs[:, i + 1], pred, noise_std))
        return -torch.stack(logps, dim=1).mean()

    @torch.no_grad()
    def extrapolate(self, x_obs: torch.Tensor, t_obs: torch.Tensor, t_future: torch.Tensor) -> torch.Tensor:
        batch = x_obs.shape[0]
        h = torch.zeros(batch, self.hidden, device=x_obs.device, dtype=x_obs.dtype)
        for i in range(x_obs.shape[1] - 1):
            dt = (t_obs[:, i + 1] - t_obs[:, i]).unsqueeze(1)
            h = self.cell(self._cell_input(x_obs[:, i], dt), h)

        current_x = x_obs[:, -1]
        current_t = t_obs[:, -1]
        preds = []
        for tq in t_future:
            dt = (tq - current_t).unsqueeze(1)
            h = self.cell(self._cell_input(current_x, dt), h)
            current_x = self.out(h)
            preds.append(current_x)
            current_t = torch.ones_like(current_t) * tq
        return torch.stack(preds, dim=1)


def gaussian_logprob(x: torch.Tensor, mean: torch.Tensor, std: float) -> torch.Tensor:
    return (-0.5 * (((x - mean) / std) ** 2 + 2.0 * math.log(std) + math.log(2.0 * math.pi))).sum(-1)


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.exp(logvar) + mu.square() - 1.0 - logvar).sum(-1)


def choose_obs_indices_batch(batch: int, n_obs: int, train_points: int,
                             rng: np.random.RandomState, device: torch.device) -> torch.Tensor:
    if n_obs >= train_points:
        idx = np.tile(np.arange(train_points, dtype=np.int64), (batch, 1))
    else:
        idx = np.empty((batch, n_obs), dtype=np.int64)
        for i in range(batch):
            idx[i] = np.sort(rng.choice(train_points, size=n_obs, replace=False))
    return torch.tensor(idx, dtype=torch.long, device=device)


def gather_by_indices(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_ids = torch.arange(data.shape[0], device=data.device).unsqueeze(1)
    return data[batch_ids, indices]


def times_by_indices(times: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return times[indices]


def time_now(device: torch.device) -> float:
    import time
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def train_latent(model: LatentODE, noisy: torch.Tensor, times: torch.Tensor, n_obs: int,
                 args: argparse.Namespace, device: torch.device, ckpt: Path,
                 metrics_path: Path) -> LatentODE:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.latent_lr)
    maybe_remove_checkpoint(ckpt, args.fresh)
    if args.fresh and metrics_path.exists():
        metrics_path.unlink()
    start, _, extra = load_ckpt(ckpt, model, optimizer, device)
    if start >= args.iterations:
        return model

    prior_elapsed = float(extra.get("total_train_seconds", 0.0))
    rng = np.random.RandomState(args.seed + 100 + n_obs)
    if extra.get("local_rng_state") is not None:
        rng.set_state(extra["local_rng_state"])
    total_epochs = max(1, math.ceil(args.iterations / args.steps_per_epoch))
    epoch_index = start // args.steps_per_epoch
    timer = ProgressTimer(total_epochs, device, start_unit=epoch_index)
    epoch_loss_sum = 0.0
    epoch_steps = 0
    f_nfes: list[int] = []
    b_nfes: list[int] = []
    timer.start()
    reset_gpu_peak_memory(device)

    for step in range(start, args.iterations):
        batch_size = min(args.batch_size, noisy.shape[0])
        batch_idx = torch.tensor(rng.choice(noisy.shape[0], size=batch_size, replace=False), device=device)
        batch_data = noisy[batch_idx]
        obs_idx = choose_obs_indices_batch(batch_size, n_obs, args.train_points, rng, device)
        x_obs = gather_by_indices(batch_data[:, :args.train_points], obs_idx)

        mu, logvar = model.infer(x_obs)
        z0 = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        model.func.nfe = 0
        pred_train = model.decode_at(z0, times[:args.train_points])
        forward_nfe = int(model.func.nfe)
        model.func.nfe = 0
        pred_obs = gather_by_indices(pred_train, obs_idx)

        logpx = gaussian_logprob(x_obs, pred_obs, args.noise_std).sum(1)
        kl = kl_standard_normal(mu, logvar)
        loss = (-logpx + kl).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        backward_nfe = int(model.func.nfe)
        model.func.nfe = 0
        optimizer.step()

        epoch_loss_sum += float(loss.detach())
        epoch_steps += 1
        f_nfes.append(forward_nfe)
        b_nfes.append(backward_nfe)
        completed = step + 1

        if completed % args.steps_per_epoch == 0 or completed == args.iterations:
            epoch_index += 1
            timing = timer.finish(epoch_index)
            avg_loss = epoch_loss_sum / max(1, epoch_steps)
            f_nfe = float(np.mean(f_nfes))
            b_nfe = float(np.mean(b_nfes))
            total_elapsed = prior_elapsed + timing["total_seconds"]
            peak_mb = gpu_peak_memory_mb(device)
            print(
                f"[LatentODE n={n_obs}] Epoch {epoch_index:03d}/{total_epochs:03d} | "
                f"step {completed}/{args.iterations} | negELBO {avg_loss:.4f} | "
                f"NFE-F {f_nfe:.1f} | NFE-B {b_nfe:.1f} | "
                f"epoch {format_duration(timing['unit_seconds'])} | total {format_duration(total_elapsed)} | "
                f"ETA {format_duration(timing['eta_seconds'])} | peak GPU {peak_mb:.0f} MB",
                flush=True,
            )
            append_csv(metrics_path, {
                "timestamp_utc": utc_timestamp(),
                "logging_epoch": epoch_index,
                "step": completed,
                "observations": n_obs,
                "model": "Latent ODE",
                "negative_elbo": avg_loss,
                "nfe_forward_mean": f_nfe,
                "nfe_backward_mean": b_nfe,
                "epoch_seconds": timing["unit_seconds"],
                "total_elapsed_seconds": total_elapsed,
                "eta_seconds": timing["eta_seconds"],
                "peak_gpu_memory_mb": peak_mb,
            })
            epoch_loss_sum = 0.0
            epoch_steps = 0
            f_nfes.clear()
            b_nfes.clear()
            if completed < args.iterations:
                timer.start()
                reset_gpu_peak_memory(device)

        if completed % args.checkpoint_every == 0 and completed < args.iterations:
            save_ckpt(
                ckpt, model, optimizer, completed, False,
                extra={
                    "total_train_seconds": prior_elapsed + (time_now(device) - timer.run_start),
                    "local_rng_state": rng.get_state(),
                },
            )

    final_elapsed = prior_elapsed + (time_now(device) - timer.run_start)
    save_ckpt(
        ckpt, model, optimizer, args.iterations, True,
        extra={"total_train_seconds": final_elapsed, "local_rng_state": rng.get_state()},
    )
    return model


def train_rnn(model: RNNBaseline, noisy: torch.Tensor, times: torch.Tensor, n_obs: int,
              args: argparse.Namespace, device: torch.device, ckpt: Path,
              metrics_path: Path) -> RNNBaseline:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.rnn_lr)
    maybe_remove_checkpoint(ckpt, args.fresh)
    if args.fresh and metrics_path.exists():
        metrics_path.unlink()
    start, _, extra = load_ckpt(ckpt, model, optimizer, device)
    if start >= args.iterations:
        return model

    prior_elapsed = float(extra.get("total_train_seconds", 0.0))
    rng = np.random.RandomState(args.seed + 1000 + n_obs)
    if extra.get("local_rng_state") is not None:
        rng.set_state(extra["local_rng_state"])
    total_epochs = max(1, math.ceil(args.iterations / args.steps_per_epoch))
    epoch_index = start // args.steps_per_epoch
    timer = ProgressTimer(total_epochs, device, start_unit=epoch_index)
    epoch_loss_sum = 0.0
    epoch_steps = 0
    timer.start()
    reset_gpu_peak_memory(device)

    for step in range(start, args.iterations):
        batch_size = min(args.batch_size, noisy.shape[0])
        batch_idx = torch.tensor(rng.choice(noisy.shape[0], size=batch_size, replace=False), device=device)
        batch_data = noisy[batch_idx, :args.train_points]
        obs_idx = choose_obs_indices_batch(batch_size, n_obs, args.train_points, rng, device)
        x_obs = gather_by_indices(batch_data, obs_idx)
        t_obs = times_by_indices(times[:args.train_points], obs_idx)
        loss = model.transition_nll(x_obs, t_obs, args.noise_std)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        epoch_loss_sum += float(loss.detach())
        epoch_steps += 1
        completed = step + 1
        if completed % args.steps_per_epoch == 0 or completed == args.iterations:
            epoch_index += 1
            timing = timer.finish(epoch_index)
            avg_loss = epoch_loss_sum / max(1, epoch_steps)
            total_elapsed = prior_elapsed + timing["total_seconds"]
            peak_mb = gpu_peak_memory_mb(device)
            print(
                f"[RNN n={n_obs}]       Epoch {epoch_index:03d}/{total_epochs:03d} | "
                f"step {completed}/{args.iterations} | NLL {avg_loss:.4f} | "
                f"epoch {format_duration(timing['unit_seconds'])} | total {format_duration(total_elapsed)} | "
                f"ETA {format_duration(timing['eta_seconds'])} | peak GPU {peak_mb:.0f} MB",
                flush=True,
            )
            append_csv(metrics_path, {
                "timestamp_utc": utc_timestamp(),
                "logging_epoch": epoch_index,
                "step": completed,
                "observations": n_obs,
                "model": "RNN",
                "nll": avg_loss,
                "epoch_seconds": timing["unit_seconds"],
                "total_elapsed_seconds": total_elapsed,
                "eta_seconds": timing["eta_seconds"],
                "peak_gpu_memory_mb": peak_mb,
            })
            epoch_loss_sum = 0.0
            epoch_steps = 0
            if completed < args.iterations:
                timer.start()
                reset_gpu_peak_memory(device)

        if completed % args.checkpoint_every == 0 and completed < args.iterations:
            save_ckpt(
                ckpt, model, optimizer, completed, False,
                extra={
                    "total_train_seconds": prior_elapsed + (time_now(device) - timer.run_start),
                    "local_rng_state": rng.get_state(),
                },
            )

    final_elapsed = prior_elapsed + (time_now(device) - timer.run_start)
    save_ckpt(
        ckpt, model, optimizer, args.iterations, True,
        extra={"total_train_seconds": final_elapsed, "local_rng_state": rng.get_state()},
    )
    return model


@torch.no_grad()
def evaluate_latent(model: LatentODE, noisy: torch.Tensor, clean: torch.Tensor, times: torch.Tensor,
                    obs_idx: torch.Tensor, args: argparse.Namespace) -> tuple[float, torch.Tensor, torch.Tensor]:
    x_obs = gather_by_indices(noisy[:, :args.train_points], obs_idx)
    mu, _ = model.infer(x_obs)
    pred_all = model.decode_at(mu, times)
    rmse = torch.sqrt(torch.mean((pred_all[:, args.train_points:] - clean[:, args.train_points:]) ** 2)).item()
    return rmse, pred_all, mu


@torch.no_grad()
def evaluate_rnn(model: RNNBaseline, noisy: torch.Tensor, clean: torch.Tensor, times: torch.Tensor,
                 obs_idx: torch.Tensor, args: argparse.Namespace) -> tuple[float, torch.Tensor]:
    x_obs = gather_by_indices(noisy[:, :args.train_points], obs_idx)
    t_obs = times_by_indices(times[:args.train_points], obs_idx)
    pred_future = model.extrapolate(x_obs, t_obs, times[args.train_points:])
    rmse = torch.sqrt(torch.mean((pred_future - clean[:, args.train_points:]) ** 2)).item()
    return rmse, pred_future


def representative_indices(directions: torch.Tensor) -> list[int]:
    result = []
    for direction in (0, 1):
        matches = torch.where(directions == direction)[0]
        if len(matches):
            result.append(int(matches[0]))
    if not result:
        result = [0]
    return result


def make_comparison_figure(n_obs: int, rnn: RNNBaseline, noisy: torch.Tensor, clean: torch.Tensor,
                           times: torch.Tensor, directions: torch.Tensor, obs_idx: torch.Tensor,
                           pred_latent: torch.Tensor, pred_rnn: torch.Tensor,
                           args: argparse.Namespace, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = representative_indices(directions)
    fig, axes = plt.subplots(2, len(chosen), figsize=(5.4 * len(chosen), 7.2), squeeze=False, constrained_layout=True)
    for col, idx in enumerate(chosen):
        gt = clean[idx].cpu().numpy()
        obs = gather_by_indices(noisy[idx:idx + 1, :args.train_points], obs_idx[idx:idx + 1])[0].cpu().numpy()
        lat = pred_latent[idx].cpu().numpy()
        rnn_future = pred_rnn[idx].cpu().numpy()

        ax = axes[0, col]
        ax.plot(gt[:, 0], gt[:, 1], linewidth=1.2, label="Ground truth")
        ax.scatter(obs[:, 0], obs[:, 1], s=14, label=f"Observed ({n_obs})")
        ax.plot(rnn_future[:, 0], rnn_future[:, 1], linestyle="--", linewidth=1.4, label="RNN extrapolation")
        ax.set_title("RNN")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

        ax = axes[1, col]
        ax.plot(gt[:, 0], gt[:, 1], linewidth=1.2, label="Ground truth")
        ax.scatter(obs[:, 0], obs[:, 1], s=14, label=f"Observed ({n_obs})")
        ax.plot(lat[:args.train_points, 0], lat[:args.train_points, 1], linewidth=1.4, label="Latent ODE reconstruction")
        ax.plot(lat[args.train_points:, 0], lat[args.train_points:, 1], linestyle="--", linewidth=1.4,
                label="Latent ODE extrapolation")
        ax.set_title("Latent ODE")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(f"Irregular spiral reconstruction and extrapolation: {n_obs}/100 observations", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"rnn_vs_latent_ode_obs{n_obs}.{suffix}", dpi=600 if suffix == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


def make_latent_space_figure(model: LatentODE, mu: torch.Tensor, directions: torch.Tensor,
                             times: torch.Tensor, args: argparse.Namespace, out_dir: Path) -> torch.Tensor:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = min(200, mu.shape[0])
    zt = model.solve(mu[:count], times[:args.train_points]).detach().cpu()
    direction_np = directions[:count].cpu().numpy()
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    for i in range(count):
        ax.plot(zt[i, :, 0], zt[i, :, 1], linewidth=0.6, alpha=0.55,
                linestyle="-" if direction_np[i] else "--")
    ax.set_xlabel("Latent dimension 1")
    ax.set_ylabel("Latent dimension 2")
    ax.set_title("Latent ODE trajectories (first two latent dimensions)")
    ax.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"latent_space.{suffix}", dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    return zt


def make_trajectory_gif(n_obs: int, noisy: torch.Tensor, clean: torch.Tensor,
                        obs_idx: torch.Tensor, pred_latent: torch.Tensor, pred_rnn: torch.Tensor,
                        args: argparse.Namespace, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    idx = 0
    gt = clean[idx].cpu().numpy()
    lat = pred_latent[idx].cpu().numpy()
    rnn = pred_rnn[idx].cpu().numpy()
    obs_ids = obs_idx[idx].cpu().numpy()
    obs = noisy[idx, obs_ids].cpu().numpy()

    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    ax.plot(gt[:, 0], gt[:, 1], alpha=0.35, linewidth=1.0, label="Ground truth")
    ax.scatter(obs[:, 0], obs[:, 1], s=18, label=f"Observed ({n_obs})")
    latent_line, = ax.plot([], [], linewidth=1.6, label="Latent ODE")
    rnn_line, = ax.plot([], [], linestyle="--", linewidth=1.6, label="RNN future")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    title = ax.set_title("")

    total = gt.shape[0]
    frame_ids = np.unique(np.linspace(1, total, 50, dtype=int))

    def update(frame_no: int):
        end = int(frame_ids[frame_no])
        latent_line.set_data(lat[:end, 0], lat[:end, 1])
        if end > args.train_points:
            future_end = end - args.train_points
            rnn_line.set_data(rnn[:future_end, 0], rnn[:future_end, 1])
        else:
            rnn_line.set_data([], [])
        title.set_text(f"Trajectory progress: {100 * end / total:.0f}%")
        return latent_line, rnn_line, title

    anim = FuncAnimation(fig, update, frames=len(frame_ids), interval=100, blit=False)
    anim.save(out_dir / f"rnn_vs_latent_ode_obs{n_obs}.gif", writer=PillowWriter(fps=10), dpi=130)
    plt.close(fig)


def make_latent_space_gif(zt: torch.Tensor, directions: torch.Tensor, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    z = zt.numpy()
    count, n_times, _ = z.shape
    direction_np = directions[:count].cpu().numpy()
    cw = direction_np == 0
    ccw = direction_np == 1
    x_min, x_max = float(z[:, :, 0].min()), float(z[:, :, 0].max())
    y_min, y_max = float(z[:, :, 1].min()), float(z[:, :, 1].max())
    pad_x = max(0.1, 0.08 * (x_max - x_min))
    pad_y = max(0.1, 0.08 * (y_max - y_min))

    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    scatter_cw = ax.scatter(z[cw, 0, 0], z[cw, 0, 1], s=14, alpha=0.6, label="Clockwise")
    scatter_ccw = ax.scatter(z[ccw, 0, 0], z[ccw, 0, 1], s=14, alpha=0.6, label="Counter-clockwise")
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_xlabel("Latent dimension 1")
    ax.set_ylabel("Latent dimension 2")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    title = ax.set_title("Latent-space evolution")
    frame_ids = np.unique(np.linspace(0, n_times - 1, min(50, n_times), dtype=int))

    def update(frame_no: int):
        t = int(frame_ids[frame_no])
        scatter_cw.set_offsets(z[cw, t, :2])
        scatter_ccw.set_offsets(z[ccw, t, :2])
        title.set_text(f"Latent-space evolution: {100 * (t + 1) / n_times:.0f}%")
        return scatter_cw, scatter_ccw, title

    anim = FuncAnimation(fig, update, frames=len(frame_ids), interval=100, blit=False)
    anim.save(out_dir / "latent_space_evolution.gif", writer=PillowWriter(fps=10), dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="optimizer steps grouped into one terminal/logging epoch")
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--n-spirals", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-lr", type=float, default=1e-2)
    parser.add_argument("--rnn-lr", type=float, default=1e-2)
    parser.add_argument("--rnn-no-dt", action="store_true",
                        help="use a plain RNN without time-gap input")
    parser.add_argument("--noise-std", type=float, default=0.3)
    parser.add_argument("--train-points", type=int, default=100)
    parser.add_argument("--extrap-points", type=int, default=100)
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    apply_mode(args, PAPER, SMOKE)
    default_dirs(args, "exp3_latent_ode", "exp3_latent_ode")
    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device(args.device, args.gpu, args.require_cuda)
    out_dir, ckpt_dir = Path(args.output_dir), Path(args.checkpoint_dir)
    save_system_info(out_dir / "system_info.json", device, args)

    print(
        f"[Exp3] mode={args.mode} | {device_summary(device)} | iterations={args.iterations} | "
        f"tol={args.rtol} | batch={args.batch_size}",
        flush=True,
    )

    clean, noisy, times, directions = make_spiral_dataset(
        n=args.n_spirals, train_points=args.train_points, extrap_points=args.extrap_points,
        noise_std=args.noise_std, seed=args.seed,
    )

    # Randomized split avoids depending on generation order while keeping a fixed seed.
    split_rng = np.random.RandomState(args.seed + 77)
    perm = torch.tensor(split_rng.permutation(clean.shape[0]), dtype=torch.long)
    clean, noisy, directions = clean[perm], noisy[perm], directions[perm]
    split = int(0.8 * clean.shape[0])

    clean = clean.to(device)
    noisy = noisy.to(device)
    times = times.to(device)
    directions = directions.to(device)
    train_noisy = noisy[:split]
    test_clean = clean[split:]
    test_noisy = noisy[split:]
    test_directions = directions[split:]

    table_rows: list[dict[str, object]] = []
    figure_pack = None

    for n_obs in (30, 50, 100):
        latent = LatentODE(rtol=args.rtol, atol=args.atol).to(device)
        rnn = RNNBaseline(hidden=25, use_dt=not args.rnn_no_dt).to(device)

        latent = train_latent(
            latent, train_noisy, times, n_obs, args, device,
            ckpt_dir / f"latent_ode_obs{n_obs}.pt",
            out_dir / "training" / f"latent_ode_obs{n_obs}.csv",
        )
        rnn = train_rnn(
            rnn, train_noisy, times, n_obs, args, device,
            ckpt_dir / f"rnn_obs{n_obs}.pt",
            out_dir / "training" / f"rnn_obs{n_obs}.csv",
        )

        # IMPORTANT: both models are evaluated on exactly the same irregular observation indices.
        eval_rng = np.random.RandomState(args.seed + 5000 + n_obs)
        obs_idx = choose_obs_indices_batch(test_noisy.shape[0], n_obs, args.train_points, eval_rng, device)
        latent_rmse, pred_latent, mu = evaluate_latent(latent, test_noisy, test_clean, times, obs_idx, args)
        rnn_rmse, pred_rnn = evaluate_rnn(rnn, test_noisy, test_clean, times, obs_idx, args)

        paper = PAPER_RMSE[n_obs]
        table_rows.append({
            "Observations": f"{n_obs}/100",
            "Paper RNN RMSE": paper["RNN"],
            "Reproduced RNN RMSE": round(rnn_rmse, 6),
            "RNN Absolute Difference": round(abs(rnn_rmse - paper["RNN"]), 6),
            "Paper Latent ODE RMSE": paper["Latent ODE"],
            "Reproduced Latent ODE RMSE": round(latent_rmse, 6),
            "Latent ODE Absolute Difference": round(abs(latent_rmse - paper["Latent ODE"]), 6),
        })
        print(
            f"[Exp3] {n_obs}/100 | RNN RMSE={rnn_rmse:.4f} (paper {paper['RNN']:.4f}) | "
            f"Latent ODE RMSE={latent_rmse:.4f} (paper {paper['Latent ODE']:.4f})",
            flush=True,
        )

        make_comparison_figure(
            n_obs, rnn, test_noisy, test_clean, times, test_directions, obs_idx,
            pred_latent, pred_rnn, args, out_dir,
        )
        np.savez_compressed(
            out_dir / f"obs{n_obs}_predictions.npz",
            obs_indices=obs_idx.cpu().numpy(),
            clean=test_clean.cpu().numpy(),
            noisy=test_noisy.cpu().numpy(),
            latent_prediction=pred_latent.cpu().numpy(),
            rnn_future_prediction=pred_rnn.cpu().numpy(),
        )

        if n_obs == 30:
            figure_pack = (latent, obs_idx, pred_latent, pred_rnn, mu)
            if not args.no_gif:
                make_trajectory_gif(n_obs, test_noisy, test_clean, obs_idx, pred_latent, pred_rnn, args, out_dir)

    write_csv(out_dir / "table2_comparison.csv", table_rows)

    if figure_pack is not None:
        latent, obs_idx, pred_latent, pred_rnn, mu = figure_pack
        zt = make_latent_space_figure(latent, mu, test_directions, times, args, out_dir)
        if not args.no_gif:
            make_latent_space_gif(zt, test_directions, out_dir)

    print(f"[Exp3] outputs: {out_dir}")


if __name__ == "__main__":
    main()
