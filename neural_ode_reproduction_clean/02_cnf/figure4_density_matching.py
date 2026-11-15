#!/usr/bin/env python3
"""Experiment 2A / Figure 4: planar NF versus planar CNF density matching.

Paper-scale training keeps the published budgets (NF: 500,000 RMSprop steps;
CNF: 10,000 Adam steps).  For readable terminal timing, those steps are grouped
into *logging epochs*; the grouping does not change the optimizer updates.
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
from flows import ContinuousNormalizingFlow, PlanarDensityFlow  # noqa: E402
from targets import PAPER_FIGURE4_ROWS, PAPER_IMAGE_FLIPPED, TARGETS  # noqa: E402

PAPER = dict(
    nf_iters=500_000,
    cnf_iters=10_000,
    sizes=[2, 4, 8, 16, 32],
    eval_samples=50_000,
    checkpoint_every=5_000,
    nf_steps_per_epoch=50_000,
    cnf_steps_per_epoch=1_000,
)
SMOKE = dict(
    nf_iters=150,
    cnf_iters=15,
    sizes=[2, 8, 32],
    eval_samples=2_000,
    checkpoint_every=50,
    nf_steps_per_epoch=50,
    cnf_steps_per_epoch=5,
)
EXTENT = (-4.0, 4.0, -4.0, 4.0)
GRID = 160


def make_grid(device: torch.device) -> torch.Tensor:
    xs = torch.linspace(EXTENT[0], EXTENT[1], GRID, device=device)
    ys = torch.linspace(EXTENT[2], EXTENT[3], GRID, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


def train(model: torch.nn.Module, kind: str, target_fn, iters: int, make_opt,
          args: argparse.Namespace, path: Path, tag: str, steps_per_epoch: int,
          metrics_path: Path) -> torch.nn.Module:
    optimizer = make_opt(model)
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
    epoch_forward_nfe: list[int] = []
    epoch_backward_nfe: list[int] = []
    timer.start()
    reset_gpu_peak_memory(args.device_obj)

    for step in range(start, iters):
        if kind == "nf":
            z, logq = model.sample_and_logq(args.batch_size)
            forward_nfe = backward_nfe = 0
        else:
            model.dynamics.nfe = 0
            z, logq = model.density_matching_sample(args.batch_size)
            forward_nfe = int(model.dynamics.nfe)
            model.dynamics.nfe = 0

        loss = (logq + target_fn(z)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if kind == "cnf":
            backward_nfe = int(model.dynamics.nfe)
            model.dynamics.nfe = 0
        optimizer.step()

        epoch_loss_sum += float(loss.detach())
        epoch_steps += 1
        if kind == "cnf":
            epoch_forward_nfe.append(forward_nfe)
            epoch_backward_nfe.append(backward_nfe)

        completed = step + 1
        at_epoch_end = completed % steps_per_epoch == 0 or completed == iters
        if at_epoch_end:
            epoch_index += 1
            timing = timer.finish(epoch_index)
            total_elapsed = prior_elapsed + timing["total_seconds"]
            avg_loss = epoch_loss_sum / max(1, epoch_steps)
            f_nfe = float(np.mean(epoch_forward_nfe)) if epoch_forward_nfe else 0.0
            b_nfe = float(np.mean(epoch_backward_nfe)) if epoch_backward_nfe else 0.0
            peak_mb = gpu_peak_memory_mb(args.device_obj)
            print(
                f"{tag} | Epoch {epoch_index:02d}/{total_epochs:02d} | "
                f"step {completed}/{iters} | KL* {avg_loss:.5f} | "
                f"NFE-F {f_nfe:.1f} | NFE-B {b_nfe:.1f} | "
                f"epoch {format_duration(timing['unit_seconds'])} | "
                f"total {format_duration(total_elapsed)} | ETA {format_duration(timing['eta_seconds'])} | "
                f"peak GPU {peak_mb:.0f} MB",
                flush=True,
            )
            append_csv(metrics_path, {
                "timestamp_utc": utc_timestamp(),
                "logging_epoch": epoch_index,
                "step": completed,
                "target": tag.split()[0].strip("[]"),
                "model": kind,
                "loss_kl_up_to_constant": avg_loss,
                "nfe_forward_mean": f_nfe,
                "nfe_backward_mean": b_nfe,
                "epoch_seconds": timing["unit_seconds"],
                "total_elapsed_seconds": total_elapsed,
                "eta_seconds": timing["eta_seconds"],
                "peak_gpu_memory_mb": peak_mb,
            })
            epoch_loss_sum = 0.0
            epoch_steps = 0
            epoch_forward_nfe.clear()
            epoch_backward_nfe.clear()
            if completed < iters:
                timer.start()
                reset_gpu_peak_memory(args.device_obj)

        if completed % args.checkpoint_every == 0 and completed < iters:
            save_ckpt(
                path, model, optimizer, completed, False,
                extra={"total_train_seconds": prior_elapsed + (time_now(args.device_obj) - timer.run_start)},
            )

    # Save the real elapsed total at the end.  The next resume only needs a sensible
    # accumulated value for reporting; training itself is unaffected by this metadata.
    final_elapsed = prior_elapsed + (time_now(args.device_obj) - timer.run_start)
    save_ckpt(path, model, optimizer, iters, True, extra={"total_train_seconds": final_elapsed})
    return model


def time_now(device: torch.device) -> float:
    import time
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.no_grad()
def final_loss(model: torch.nn.Module, kind: str, target_fn, n: int) -> float:
    if kind == "nf":
        z, logq = model.sample_and_logq(n)
    else:
        z, logq = model.density_matching_sample(n)
    return float((logq + target_fn(z)).mean())


@torch.no_grad()
def model_density(model: torch.nn.Module, kind: str, grid: torch.Tensor) -> np.ndarray:
    if kind == "nf":
        logp = model.log_prob(grid)
    else:
        logp = torch.cat([model.log_prob(grid[i:i + 8192]) for i in range(0, grid.shape[0], 8192)])
    return logp.exp().reshape(GRID, GRID).cpu().numpy()


@torch.no_grad()
def target_density(target_fn, grid: torch.Tensor) -> np.ndarray:
    return torch.exp(-target_fn(grid)).reshape(GRID, GRID).cpu().numpy()


def plot(images: dict, losses: dict, sizes: list[int], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["target", "nf2", "nf8", "nf32", "cnf2", "cnf8", "cnf32"]
    heads = ["Target", "K=2", "K=8", "K=32", "M=2", "M=8", "M=32"]
    fig = plt.figure(figsize=(12.0, 4.6), dpi=180)
    gs = fig.add_gridspec(
        3, 11,
        width_ratios=[1, 0.15, 1, 1, 1, 0.28, 1, 1, 1, 0.15, 1.35],
        wspace=0.06, hspace=0.06, left=0.04, right=0.99, top=0.91, bottom=0.12,
    )
    img_cols = [0, 2, 3, 4, 6, 7, 8]
    origin = "upper" if PAPER_IMAGE_FLIPPED else "lower"

    for row in range(3):
        for col, name in enumerate(cols):
            ax = fig.add_subplot(gs[row, img_cols[col]])
            img = images[(row, name)]
            ax.imshow(img, origin=origin, extent=EXTENT, aspect="equal", cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(heads[col], fontsize=9)
            if col == 0:
                ax.set_ylabel(f"Target {row + 1}", fontsize=9)

        ax = fig.add_subplot(gs[row, 10])
        ax.plot(sizes, losses[(row, "cnf")], marker="o", label="CNF")
        ax.plot(sizes, losses[(row, "nf")], marker="s", linestyle="--", label="NF")
        ax.set_xlabel("K / M")
        ax.set_ylabel("KL* loss")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    fig.text(0.29, 0.035, "Normalizing Flow", ha="center", fontsize=10)
    fig.text(0.63, 0.035, "Continuous Normalizing Flow", ha="center", fontsize=10)
    fig.text(0.92, 0.035, "Loss vs. capacity", ha="center", fontsize=10)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"figure4_nf_vs_cnf.{suffix}", dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--nf-iters", type=int, default=None)
    parser.add_argument("--cnf-iters", type=int, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--nf-steps-per-epoch", type=int, default=None)
    parser.add_argument("--cnf-steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--nf-lr", type=float, default=1e-5)
    parser.add_argument("--nf-momentum", type=float, default=0.9)
    parser.add_argument("--cnf-lr", type=float, default=1e-3)
    parser.add_argument("--hyper-hidden", type=int, default=32)
    parser.add_argument("--integration-time", type=float, default=10.0)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    apply_mode(args, PAPER, SMOKE)
    default_dirs(args, "exp2_cnf", "exp2_cnf/figure4")
    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device(args.device, args.gpu, args.require_cuda)
    args.device_obj = device
    out_dir, ckpt_dir = Path(args.output_dir), Path(args.checkpoint_dir)
    save_system_info(out_dir / "system_info_figure4.json", device, args)

    for needed in (2, 8, 32):
        if needed not in args.sizes:
            raise ValueError("--sizes must contain 2, 8 and 32 for the published Figure 4 columns")

    print(
        f"[Exp2/Fig4] mode={args.mode} | {device_summary(device)} | "
        f"NF={args.nf_iters} steps | CNF={args.cnf_iters} steps | sizes={args.sizes}",
        flush=True,
    )

    grid = make_grid(device)
    images: dict = {}
    losses: dict = {}
    loss_rows: list[dict[str, object]] = []

    for row, target_name in enumerate(PAPER_FIGURE4_ROWS):
        target_fn = TARGETS[target_name]
        images[(row, "target")] = target_density(target_fn, grid)
        losses[(row, "nf")], losses[(row, "cnf")] = [], []

        for k in args.sizes:
            model = PlanarDensityFlow(k).to(device)
            model = train(
                model, "nf", target_fn, args.nf_iters,
                lambda mod: torch.optim.RMSprop(mod.parameters(), lr=args.nf_lr, momentum=args.nf_momentum),
                args, ckpt_dir / f"{target_name}_nf_k{k}.pt", f"[{target_name}] NF K={k}",
                args.nf_steps_per_epoch, out_dir / "training" / f"{target_name}_nf_k{k}.csv",
            )
            value = final_loss(model, "nf", target_fn, args.eval_samples)
            losses[(row, "nf")].append(value)
            loss_rows.append({"Target": target_name, "Capacity": k, "Model": "NF", "Final KL*": value})
            if k in (2, 8, 32):
                images[(row, f"nf{k}")] = model_density(model, "nf", grid)

        for m in args.sizes:
            model = ContinuousNormalizingFlow(
                width=m, hidden_dim=args.hyper_hidden, integration_time=args.integration_time,
                rtol=args.rtol, atol=args.atol, learn_base=True, base_variance=1.0, adjoint=True,
            ).to(device)
            model = train(
                model, "cnf", target_fn, args.cnf_iters,
                lambda mod: torch.optim.Adam(mod.parameters(), lr=args.cnf_lr),
                args, ckpt_dir / f"{target_name}_cnf_m{m}.pt", f"[{target_name}] CNF M={m}",
                args.cnf_steps_per_epoch, out_dir / "training" / f"{target_name}_cnf_m{m}.csv",
            )
            value = final_loss(model, "cnf", target_fn, args.eval_samples)
            losses[(row, "cnf")].append(value)
            loss_rows.append({"Target": target_name, "Capacity": m, "Model": "CNF", "Final KL*": value})
            if m in (2, 8, 32):
                images[(row, f"cnf{m}")] = model_density(model, "cnf", grid)

    write_csv(out_dir / "figure4_losses.csv", loss_rows)
    plot(images, losses, args.sizes, out_dir)
    print(f"[Exp2/Fig4] outputs: {out_dir}")


if __name__ == "__main__":
    main()
