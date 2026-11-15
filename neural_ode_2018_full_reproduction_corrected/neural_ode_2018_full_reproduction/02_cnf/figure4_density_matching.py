#!/usr/bin/env python3
"""Figure 4: planar NF (K layers) vs planar CNF (M hidden units) density matching.

Loss: KL(q || p) up to the constant log Z, i.e. E_q[log q(z) + U(z)] with p ~ exp(-U).
Paper settings: CNF = 10,000 Adam iterations; NF = 500,000 RMSprop iterations.
Targets: Rezende & Mohamed (2015) U1, U3, U4 (see targets.py for the visual verification).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import add_common_args, apply_mode, default_dirs, get_device, load_ckpt, save_ckpt, set_seed  # noqa: E402
from flows import ContinuousNormalizingFlow, PlanarDensityFlow  # noqa: E402
from targets import PAPER_FIGURE4_ROWS, PAPER_IMAGE_FLIPPED, TARGETS  # noqa: E402

PAPER = dict(
    nf_iters=500_000,                 # paper Sec. 4.1
    cnf_iters=10_000,                 # paper Sec. 4.1
    sizes=[2, 4, 8, 16, 32],          # Loss-vs-K/M panel of the paper shows 5 points (2,4,8,16,32)
    eval_samples=50_000,
    checkpoint_every=5_000,
)
SMOKE = dict(nf_iters=150, cnf_iters=15, sizes=[2, 8, 32], eval_samples=2_000, checkpoint_every=50)
EXTENT = (-4.0, 4.0, -4.0, 4.0)       # ASSUMPTION: plotting window (not stated in the paper)
GRID = 160


def make_grid(device):
    xs = torch.linspace(EXTENT[0], EXTENT[1], GRID, device=device)
    ys = torch.linspace(EXTENT[2], EXTENT[3], GRID, device=device)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


def train(model, kind, target_fn, iters, make_opt, args, path, tag):
    opt = make_opt(model)
    start, complete, _ = load_ckpt(path, model, opt, args.device_obj)
    if complete:
        return model
    report = max(1, iters // 10)
    for step in range(start, iters):
        if kind == "nf":
            z, logq = model.sample_and_logq(args.batch_size)
        else:
            z, logq = model.density_matching_sample(args.batch_size)
        loss = (logq + target_fn(z)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % report == 0 or step == start:
            print(f"{tag} step={step+1}/{iters} KL*={loss.item():.4f}", flush=True)
        if (step + 1) % args.checkpoint_every == 0 and step + 1 < iters:
            save_ckpt(path, model, opt, step + 1, False)
    save_ckpt(path, model, opt, iters, True)
    return model


@torch.no_grad()
def final_loss(model, kind, target_fn, n):
    if kind == "nf":
        z, logq = model.sample_and_logq(n)
    else:
        z, logq = model.density_matching_sample(n)
    return float((logq + target_fn(z)).mean())


@torch.no_grad()
def model_density(model, kind, grid):
    """Exact q(x) of the trained model on the plotting grid (NF: bisection inverse; CNF: reverse ODE)."""
    if kind == "nf":
        lp = model.log_prob(grid)
    else:
        lp = torch.cat([model.log_prob(grid[i : i + 8192]) for i in range(0, grid.shape[0], 8192)])
    return lp.exp().reshape(GRID, GRID).cpu().numpy()


@torch.no_grad()
def target_density(target_fn, grid):
    return torch.exp(-target_fn(grid)).reshape(GRID, GRID).cpu().numpy()


def plot(images, losses, sizes, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["target", "nf2", "nf8", "nf32", "cnf2", "cnf8", "cnf32"]
    heads = ["", "K=2", "K=8", "K=32", "M=2", "M=8", "M=32"]
    fig = plt.figure(figsize=(9.2, 3.5), dpi=220)
    # columns: Target | gap | K=2,8,32 | gap | M=2,8,32 | gap | loss panel  (group gaps as in the paper)
    gs = fig.add_gridspec(3, 11, width_ratios=[1, 0.18, 1, 1, 1, 0.3, 1, 1, 1, 0.18, 1.25], wspace=0.06, hspace=0.06,
                          left=0.05, right=0.99, top=0.9, bottom=0.12)
    img_cols = [0, 2, 3, 4, 6, 7, 8]
    origin = "upper" if PAPER_IMAGE_FLIPPED else "lower"   # paper images are vertically flipped
    for r in range(3):
        for c, name in enumerate(cols):
            ax = fig.add_subplot(gs[r, img_cols[c]])
            img = images[(r, name)]
            ax.imshow(img, origin=origin, extent=EXTENT, aspect="equal", cmap="viridis", vmin=0, vmax=img.max())
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(heads[c], fontsize=7, pad=2)
            if c == 0:
                ax.set_ylabel(str(r + 1), rotation=0, fontsize=8, labelpad=6, va="center")
        ax = fig.add_subplot(gs[r, 10])
        ax.set_box_aspect(0.72)
        ax.plot(sizes, losses[(r, "cnf")], "-s", color="#1f77b4", ms=2.5, lw=0.8, label="CNF")
        ax.plot(sizes, losses[(r, "nf")], "--s", color="#ff7f0e", ms=2.5, lw=0.8, label="NF")
        ax.grid(True, lw=0.4)
        ax.set_yticklabels([])
        ax.set_xticks([10, 20, 30])
        ax.tick_params(labelsize=6, length=1.5, pad=1)
        ax.legend(fontsize=3.5, loc="upper right", frameon=True, handlelength=1.2)
    fig.text(0.095, 0.03, "(a) Target", ha="center", fontsize=8)
    fig.text(0.31, 0.03, "(b) NF", ha="center", fontsize=8)
    fig.text(0.58, 0.03, "(c) CNF", ha="center", fontsize=8)
    fig.text(0.9, 0.03, "(d) Loss vs. K/M", ha="center", fontsize=8)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--nf-iters", type=int, default=None)
    p.add_argument("--cnf-iters", type=int, default=None)
    p.add_argument("--sizes", type=int, nargs="+", default=None, help="K=M values for the loss panel")
    p.add_argument("--eval-samples", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=None)
    # ASSUMPTION: the paper does not give batch size, learning rates or integration time.
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--nf-lr", type=float, default=1e-5, help="RMSprop lr; Rezende&Mohamed-style, NOT stated in the Neural ODE paper")
    p.add_argument("--nf-momentum", type=float, default=0.9)
    p.add_argument("--cnf-lr", type=float, default=1e-3, help="Adam lr; not stated in the paper")
    p.add_argument("--hyper-hidden", type=int, default=32)
    p.add_argument("--integration-time", type=float, default=10.0)
    p.add_argument("--rtol", type=float, default=1e-5, help="paper Sec. 6: 1e-5 for density estimation")
    p.add_argument("--atol", type=float, default=1e-5)
    args = p.parse_args()
    apply_mode(args, PAPER, SMOKE)
    default_dirs(args, "cnf", "cnf/figure4")
    set_seed(args.seed)
    device = get_device(args.device)
    args.device_obj = device
    ckpt_dir, out_dir = Path(args.checkpoint_dir), Path(args.output_dir)
    for need in (2, 8, 32):
        assert need in args.sizes, "sizes must contain 2, 8 and 32 (the columns of Figure 4)"
    print(f"[fig4] mode={args.mode} device={device} NF iters={args.nf_iters} CNF iters={args.cnf_iters} sizes={args.sizes}")

    grid = make_grid(device)
    images, losses = {}, {}
    for r, tname in enumerate(PAPER_FIGURE4_ROWS):
        target_fn = TARGETS[tname]
        images[(r, "target")] = target_density(target_fn, grid)
        losses[(r, "nf")], losses[(r, "cnf")] = [], []
        for k in args.sizes:
            m = PlanarDensityFlow(k).to(device)
            m = train(m, "nf", target_fn, args.nf_iters,
                      lambda mod: torch.optim.RMSprop(mod.parameters(), lr=args.nf_lr, momentum=args.nf_momentum),
                      args, ckpt_dir / f"{tname}_nf_k{k}.pt", f"[{tname}] NF K={k}")
            losses[(r, "nf")].append(final_loss(m, "nf", target_fn, args.eval_samples))
            if k in (2, 8, 32):
                images[(r, f"nf{k}")] = model_density(m, "nf", grid)
        for k in args.sizes:
            m = ContinuousNormalizingFlow(width=k, hidden_dim=args.hyper_hidden, integration_time=args.integration_time,
                                          rtol=args.rtol, atol=args.atol, learn_base=True, base_variance=1.0,
                                          adjoint=True).to(device)
            m = train(m, "cnf", target_fn, args.cnf_iters,
                      lambda mod: torch.optim.Adam(mod.parameters(), lr=args.cnf_lr),
                      args, ckpt_dir / f"{tname}_cnf_m{k}.pt", f"[{tname}] CNF M={k}")
            losses[(r, "cnf")].append(final_loss(m, "cnf", target_fn, args.eval_samples))
            if k in (2, 8, 32):
                images[(r, f"cnf{k}")] = model_density(m, "cnf", grid)
    plot(images, losses, args.sizes, out_dir / "figure4.png")
    print("[fig4] final losses (KL up to const):")
    for r, tname in enumerate(PAPER_FIGURE4_ROWS):
        print(" ", tname, "NF", np.round(losses[(r, "nf")], 3), "CNF", np.round(losses[(r, "cnf")], 3))
    print(f"[fig4] wrote {out_dir / 'figure4.png'}")


if __name__ == "__main__":
    main()
