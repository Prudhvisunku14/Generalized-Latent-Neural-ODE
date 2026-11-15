#!/usr/bin/env python3
"""Experiment 2B / Figure 5: maximum-likelihood CNF versus planar NF.

Creates publication-quality PNG/PDF panels plus smooth GIFs for Two Circles and
Two Moons.  The GIFs are generated from trained models, not from hand-drawn data.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

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
from flows import ContinuousNormalizingFlow, PlanarMLEFlow  # noqa: E402

FRACTIONS = [0.05, 0.20, 0.40, 0.60, 0.80, 1.00]
PAPER = dict(
    cnf_iters=10_000,
    nf_iters=500_000,
    sample_points=8_000,
    gif_points=2_500,
    gif_frames=40,
    checkpoint_every=5_000,
    nf_steps_per_epoch=50_000,
    cnf_steps_per_epoch=1_000,
)
SMOKE = dict(
    cnf_iters=15,
    nf_iters=100,
    sample_points=1_000,
    gif_points=400,
    gif_frames=8,
    checkpoint_every=50,
    nf_steps_per_epoch=50,
    cnf_steps_per_epoch=5,
)
GRID = 128


def sample_two_circles(n: int, device: torch.device, noise: float = 0.06) -> torch.Tensor:
    """Two concentric circles, matching the public CNF demo convention."""
    n1 = n // 2
    n2 = n - n1
    a1 = 2.0 * np.pi * torch.rand(n1, device=device)
    a2 = 2.0 * np.pi * torch.rand(n2, device=device)
    outer = torch.stack([torch.cos(a1), torch.sin(a1)], dim=1)
    inner = 0.5 * torch.stack([torch.cos(a2), torch.sin(a2)], dim=1)
    x = torch.cat([outer, inner], dim=0)
    x = x + noise * torch.randn_like(x)
    return x[torch.randperm(n, device=device)]


def sample_two_moons(n: int, device: torch.device, noise: float = 0.06) -> torch.Tensor:
    """Two-moons geometry; exact noise/scale are not specified in the 2018 paper."""
    n1 = n // 2
    n2 = n - n1
    t1 = np.pi * torch.rand(n1, device=device)
    t2 = np.pi * torch.rand(n2, device=device)
    moon1 = torch.stack([torch.cos(t1), torch.sin(t1)], dim=1)
    moon2 = torch.stack([1.0 - torch.cos(t2), 1.0 - torch.sin(t2) - 0.5], dim=1)
    x = torch.cat([moon1, moon2], dim=0)
    x = x + noise * torch.randn_like(x)
    x = (x - torch.tensor([0.5, 0.25], device=device)) * 0.75
    return x[torch.randperm(n, device=device)]


DATASETS = {
    "circles": ("Two Circles", sample_two_circles, (-1.5, 1.5, -1.5, 1.5)),
    "moons": ("Two Moons", sample_two_moons, (-1.4, 1.4, -1.4, 1.4)),
}


def time_now(device: torch.device) -> float:
    import time
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def train_generic(model: torch.nn.Module, kind: str, loss_fn, dataset_fn, optimizer,
                  iters: int, steps_per_epoch: int, args: argparse.Namespace,
                  path: Path, tag: str, metrics_path: Path) -> torch.nn.Module:
    maybe_remove_checkpoint(path, args.fresh)
    if args.fresh and metrics_path.exists():
        metrics_path.unlink()

    start, _, extra = load_ckpt(path, model, optimizer, args.device_obj)
    if start >= iters:
        return model

    prior_elapsed = float(extra.get("total_train_seconds", 0.0))
    total_epochs = max(1, math.ceil(iters / steps_per_epoch))
    timer = ProgressTimer(total_epochs, args.device_obj, start_unit=start // steps_per_epoch)
    epoch_index = start // steps_per_epoch
    epoch_loss_sum = 0.0
    epoch_steps = 0
    f_nfes: list[int] = []
    b_nfes: list[int] = []
    timer.start()
    reset_gpu_peak_memory(args.device_obj)

    for step in range(start, iters):
        x = dataset_fn(args.batch_size, args.device_obj)
        if kind == "cnf":
            model.dynamics.nfe = 0
        logp = loss_fn(model, x)
        if kind == "cnf":
            f_nfes.append(int(model.dynamics.nfe))
            model.dynamics.nfe = 0

        loss = -logp.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if kind == "cnf":
            b_nfes.append(int(model.dynamics.nfe))
            model.dynamics.nfe = 0
        optimizer.step()

        epoch_loss_sum += float(loss.detach())
        epoch_steps += 1
        completed = step + 1
        at_epoch_end = completed % steps_per_epoch == 0 or completed == iters

        if at_epoch_end:
            epoch_index += 1
            timing = timer.finish(epoch_index)
            avg_loss = epoch_loss_sum / max(1, epoch_steps)
            forward_nfe = float(np.mean(f_nfes)) if f_nfes else 0.0
            backward_nfe = float(np.mean(b_nfes)) if b_nfes else 0.0
            total_elapsed = prior_elapsed + timing["total_seconds"]
            peak_mb = gpu_peak_memory_mb(args.device_obj)
            print(
                f"{tag} | Epoch {epoch_index:02d}/{total_epochs:02d} | step {completed}/{iters} | "
                f"NLL {avg_loss:.5f} | NFE-F {forward_nfe:.1f} | NFE-B {backward_nfe:.1f} | "
                f"epoch {format_duration(timing['unit_seconds'])} | total {format_duration(total_elapsed)} | "
                f"ETA {format_duration(timing['eta_seconds'])} | peak GPU {peak_mb:.0f} MB",
                flush=True,
            )
            append_csv(metrics_path, {
                "timestamp_utc": utc_timestamp(),
                "logging_epoch": epoch_index,
                "step": completed,
                "model": kind,
                "nll": avg_loss,
                "nfe_forward_mean": forward_nfe,
                "nfe_backward_mean": backward_nfe,
                "epoch_seconds": timing["unit_seconds"],
                "total_elapsed_seconds": total_elapsed,
                "eta_seconds": timing["eta_seconds"],
                "peak_gpu_memory_mb": peak_mb,
            })
            epoch_loss_sum = 0.0
            epoch_steps = 0
            f_nfes.clear()
            b_nfes.clear()
            if completed < iters:
                timer.start()
                reset_gpu_peak_memory(args.device_obj)

        if completed % args.checkpoint_every == 0 and completed < iters:
            save_ckpt(
                path, model, optimizer, completed, False,
                extra={"total_train_seconds": prior_elapsed + (time_now(args.device_obj) - timer.run_start)},
            )

    final_elapsed = prior_elapsed + (time_now(args.device_obj) - timer.run_start)
    save_ckpt(path, model, optimizer, iters, True, extra={"total_train_seconds": final_elapsed})
    return model


def hist_image(x: torch.Tensor, extent: tuple[float, float, float, float], bins: int = 64) -> np.ndarray:
    array = x.detach().cpu().numpy()
    hist, _, _ = np.histogram2d(
        array[:, 0], array[:, 1], bins=bins,
        range=[[extent[0], extent[1]], [extent[2], extent[3]]],
    )
    return hist.T


def make_grid(extent: tuple[float, float, float, float], device: torch.device) -> torch.Tensor:
    xs = torch.linspace(extent[0], extent[1], GRID, device=device)
    ys = torch.linspace(extent[2], extent[3], GRID, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


@torch.no_grad()
def evaluate_nll(model: torch.nn.Module, dataset_fn, device: torch.device, n: int = 10_000) -> float:
    x = dataset_fn(n, device)
    return float(-model.log_prob(x).mean())


def build_panels(key: str, args: argparse.Namespace, device: torch.device,
                 ckpt_dir: Path, out_dir: Path) -> dict:
    title, dataset_fn, extent = DATASETS[key]

    cnf = ContinuousNormalizingFlow(
        width=64, hidden_dim=args.hyper_hidden, integration_time=args.integration_time,
        rtol=args.rtol, atol=args.atol, learn_base=False, base_variance=0.1, adjoint=True,
    ).to(device)
    cnf = train_generic(
        cnf, "cnf", lambda model, x: model.log_prob(x), dataset_fn,
        torch.optim.Adam(cnf.parameters(), lr=args.cnf_lr), args.cnf_iters,
        args.cnf_steps_per_epoch, args, ckpt_dir / f"{key}_cnf64.pt", f"[{key}] CNF64",
        out_dir / "training" / f"{key}_cnf64.csv",
    )

    nf = PlanarMLEFlow(n_layers=64, base_variance=0.1).to(device)
    nf = train_generic(
        nf, "nf", lambda model, x: model.log_prob(x), dataset_fn,
        torch.optim.RMSprop(nf.parameters(), lr=args.nf_lr, momentum=args.nf_momentum),
        args.nf_iters, args.nf_steps_per_epoch, args, ckpt_dir / f"{key}_nf64.pt", f"[{key}] NF64",
        out_dir / "training" / f"{key}_nf64.csv",
    )

    grid = make_grid(extent, device)
    density = [d.reshape(GRID, GRID).cpu().numpy() for d in cnf.density_stages(grid, FRACTIONS)]
    nf_density = [d.reshape(GRID, GRID).cpu().numpy() for d in nf.stage_density(grid, FRACTIONS)]
    cnf_sample_tensors = cnf.sample_stages(args.sample_points, FRACTIONS)
    samples = [hist_image(s, extent) for s in cnf_sample_tensors]
    target_points = dataset_fn(args.sample_points, device)
    target = hist_image(target_points, extent)

    # Raw stage samples are useful for replotting without another ODE solve.
    np.savez_compressed(
        out_dir / f"{key}_stage_samples.npz",
        fractions=np.asarray(FRACTIONS),
        **{f"cnf_{int(100*f):03d}": s.detach().cpu().numpy() for f, s in zip(FRACTIONS, cnf_sample_tensors)},
        target=target_points.detach().cpu().numpy(),
    )

    return {
        "key": key,
        "title": title,
        "extent": extent,
        "density": density,
        "samples": samples,
        "nf": nf_density,
        "target": target,
        "cnf_model": cnf,
        "nf_model": nf,
        "dataset_fn": dataset_fn,
    }


def plot(panels: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.0, 4.4), dpi=180)
    outer = fig.add_gridspec(1, len(panels), wspace=0.08, left=0.02, right=0.995, top=0.88, bottom=0.12)
    for panel_i, panel in enumerate(panels):
        inner = outer[panel_i].subgridspec(3, 7, width_ratios=[1] * 6 + [1.15], wspace=0.01, hspace=0.04)
        rows = [("CNF Density", panel["density"]), ("CNF Samples", panel["samples"]), ("NF Density", panel["nf"])]
        for row, (name, images) in enumerate(rows):
            for col in range(6):
                ax = fig.add_subplot(inner[row, col])
                ax.imshow(images[col], origin="lower", extent=panel["extent"], aspect="equal", cmap="viridis")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"{int(round(FRACTIONS[col] * 100))}%", fontsize=8)
                if col == 0:
                    ax.set_ylabel(name, fontsize=8)
        ax_target = fig.add_subplot(inner[1, 6])
        ax_target.imshow(panel["target"], origin="lower", extent=panel["extent"], aspect="equal", cmap="viridis")
        ax_target.set_xticks([])
        ax_target.set_yticks([])
        ax_target.set_title("Target", fontsize=8)
        ax_blank = fig.add_subplot(inner[0, 6])
        ax_blank.axis("off")
        ax_blank = fig.add_subplot(inner[2, 6])
        ax_blank.axis("off")
        fig.text((panel_i + 0.5) / len(panels), 0.035, panel["title"], ha="center", fontsize=10)

    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"figure5_transformations.{suffix}", dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def make_transform_gif(panel: dict, args: argparse.Namespace, out_dir: Path, device: torch.device) -> None:
    """Animate NF and CNF sample transformations side-by-side."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fractions = np.linspace(1.0 / args.gif_frames, 1.0, args.gif_frames)
    frac_list = [float(v) for v in fractions]
    cnf_stages = panel["cnf_model"].sample_stages(args.gif_points, frac_list)
    nf_stages = panel["nf_model"].sample_stages(args.gif_points, frac_list)
    target = panel["dataset_fn"](args.gif_points, device).detach().cpu().numpy()
    cnf_np = [x.detach().cpu().numpy() for x in cnf_stages]
    nf_np = [x.detach().cpu().numpy() for x in nf_stages]

    extent = panel["extent"]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2), constrained_layout=True)
    for ax, title in zip(axes, ["Continuous NF", "Normalizing Flow", "Target"]):
        ax.set_title(title)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    scat_cnf = axes[0].scatter(cnf_np[0][:, 0], cnf_np[0][:, 1], s=4, alpha=0.55)
    scat_nf = axes[1].scatter(nf_np[0][:, 0], nf_np[0][:, 1], s=4, alpha=0.55)
    axes[2].scatter(target[:, 0], target[:, 1], s=4, alpha=0.55)
    progress = fig.suptitle(f"{panel['title']} | 0%", fontsize=12)

    def update(frame: int):
        scat_cnf.set_offsets(cnf_np[frame])
        scat_nf.set_offsets(nf_np[frame])
        progress.set_text(f"{panel['title']} | transformation {100 * fractions[frame]:.0f}%")
        return scat_cnf, scat_nf, progress

    animation = FuncAnimation(fig, update, frames=args.gif_frames, interval=100, blit=False)
    animation.save(out_dir / f"{panel['key']}_nf_vs_cnf.gif", writer=PillowWriter(fps=10), dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--cnf-iters", type=int, default=None)
    parser.add_argument("--nf-iters", type=int, default=None)
    parser.add_argument("--sample-points", type=int, default=None)
    parser.add_argument("--gif-points", type=int, default=None)
    parser.add_argument("--gif-frames", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--nf-steps-per-epoch", type=int, default=None)
    parser.add_argument("--cnf-steps-per-epoch", type=int, default=None)
    parser.add_argument("--dataset", choices=["both", "circles", "moons"], default="both")
    parser.add_argument("--no-gif", action="store_true", help="skip GIF generation")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cnf-lr", type=float, default=1e-3)
    parser.add_argument("--nf-lr", type=float, default=1e-5)
    parser.add_argument("--nf-momentum", type=float, default=0.9)
    parser.add_argument("--hyper-hidden", type=int, default=32)
    parser.add_argument("--integration-time", type=float, default=10.0)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    apply_mode(args, PAPER, SMOKE)
    default_dirs(args, "exp2_cnf", "exp2_cnf/figure5")
    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device(args.device, args.gpu, args.require_cuda)
    args.device_obj = device
    out_dir, ckpt_dir = Path(args.output_dir), Path(args.checkpoint_dir)
    save_system_info(out_dir / "system_info_figure5.json", device, args)

    print(
        f"[Exp2/Fig5] mode={args.mode} | {device_summary(device)} | "
        f"CNF={args.cnf_iters} steps | NF={args.nf_iters} steps",
        flush=True,
    )

    keys = ["circles", "moons"] if args.dataset == "both" else [args.dataset]
    panels = [build_panels(key, args, device, ckpt_dir, out_dir) for key in keys]
    plot(panels, out_dir)

    metrics: list[dict[str, object]] = []
    for panel in panels:
        cnf_nll = evaluate_nll(panel["cnf_model"], panel["dataset_fn"], device)
        nf_nll = evaluate_nll(panel["nf_model"], panel["dataset_fn"], device)
        metrics.extend([
            {"Dataset": panel["title"], "Model": "CNF", "Final NLL": cnf_nll},
            {"Dataset": panel["title"], "Model": "NF", "Final NLL": nf_nll},
        ])
        if not args.no_gif:
            make_transform_gif(panel, args, out_dir, device)

    write_csv(out_dir / "figure5_nll.csv", metrics)
    print(f"[Exp2/Fig5] outputs: {out_dir}")


if __name__ == "__main__":
    main()
