#!/usr/bin/env python3
"""Figure 5: maximum-likelihood CNF (64 hidden units) vs NF (64 stacked 1-unit planar layers).

Rows (as in the paper): Density (exact model density p(z(t)) of the CNF), Samples (CNF samples
pushed forward), NF (exact density of the NF's generative intermediate state), plus the data
(Target) next to the Samples row.  Columns: 5%, 20%, 40%, 60%, 80%, 100% of the flow.
Panels (a) Two Circles and (b) Two Moons are written to a single figure5.png.
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
from flows import ContinuousNormalizingFlow, PlanarMLEFlow  # noqa: E402

FRACTIONS = [0.05, 0.20, 0.40, 0.60, 0.80, 1.00]

# ASSUMPTION: the paper states 64 units / 64 layers but NOT the iteration counts or learning rates of
# Figure 5.  CNF/NF budgets below reuse the Figure 4 budgets (10k Adam / 500k RMSprop).
PAPER = dict(cnf_iters=10_000, nf_iters=500_000, viz_samples=60_000, sample_points=8_000, checkpoint_every=5_000)
SMOKE = dict(cnf_iters=15, nf_iters=100, viz_samples=3_000, sample_points=1_500, checkpoint_every=50)
GRID = 128


def sample_two_circles(n, device, noise=0.06):
    """Same distribution as sklearn.datasets.make_circles(noise=0.06, factor=0.5) (public CNF demo)."""
    n1 = n // 2
    n2 = n - n1
    a1 = 2.0 * np.pi * torch.rand(n1, device=device)
    a2 = 2.0 * np.pi * torch.rand(n2, device=device)
    outer = torch.stack([torch.cos(a1), torch.sin(a1)], dim=1)
    inner = 0.5 * torch.stack([torch.cos(a2), torch.sin(a2)], dim=1)
    x = torch.cat([outer, inner], dim=0)
    x = x + noise * torch.randn_like(x)
    return x[torch.randperm(n, device=device)]


def sample_two_moons(n, device, noise=0.06):
    """sklearn make_moons geometry, recentred and scaled by 0.75.

    ASSUMPTION: noise level, centring and scale are not given in the paper.
    """
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


def train_generic(model, loss_fn, dataset_fn, opt, iters, args, path, tag):
    start, complete, _ = load_ckpt(path, model, opt, args.device_obj)
    if complete:
        return model
    report = max(1, iters // 10)
    for step in range(start, iters):
        x = dataset_fn(args.batch_size, args.device_obj)
        loss = -loss_fn(model, x).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % report == 0 or step == start:
            print(f"{tag} step={step+1}/{iters} NLL={loss.item():.4f}", flush=True)
        if (step + 1) % args.checkpoint_every == 0 and step + 1 < iters:
            save_ckpt(path, model, opt, step + 1, False)
    save_ckpt(path, model, opt, iters, True)
    return model


def hist_image(x, extent, bins=64):
    a = x.detach().cpu().numpy()
    H, _, _ = np.histogram2d(a[:, 0], a[:, 1], bins=bins, range=[[extent[0], extent[1]], [extent[2], extent[3]]])
    return H.T


def make_grid(extent, device):
    xs = torch.linspace(extent[0], extent[1], GRID, device=device)
    ys = torch.linspace(extent[2], extent[3], GRID, device=device)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


def build_panels(key, args, device, ckpt_dir):
    title, fn, extent = DATASETS[key]
    cnf = ContinuousNormalizingFlow(width=64, hidden_dim=args.hyper_hidden, integration_time=args.integration_time,
                                    rtol=args.rtol, atol=args.atol, learn_base=False, base_variance=0.1,
                                    adjoint=True).to(device)
    cnf = train_generic(cnf, lambda m, x: m.log_prob(x), fn, torch.optim.Adam(cnf.parameters(), lr=args.cnf_lr),
                        args.cnf_iters, args, ckpt_dir / f"{key}_cnf64.pt", f"[{key}] CNF64")
    nf = PlanarMLEFlow(n_layers=64, base_variance=0.1).to(device)
    nf = train_generic(nf, lambda m, x: m.log_prob(x), fn,
                       torch.optim.RMSprop(nf.parameters(), lr=args.nf_lr, momentum=args.nf_momentum),
                       args.nf_iters, args, ckpt_dir / f"{key}_nf64.pt", f"[{key}] NF64")
    grid = make_grid(extent, device)
    dens = [d.reshape(GRID, GRID).cpu().numpy() for d in cnf.density_stages(grid, FRACTIONS)]
    nf_d = [d.reshape(GRID, GRID).cpu().numpy() for d in nf.stage_density(grid, FRACTIONS)]
    samples = [hist_image(s, extent) for s in cnf.sample_stages(args.sample_points, FRACTIONS)]
    target = hist_image(fn(args.sample_points, device), extent)
    return dict(title=title, extent=extent, density=dens, samples=samples, nf=nf_d, target=target)


def plot(panels, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10.5, 3.4), dpi=220)
    outer = fig.add_gridspec(1, 2, wspace=0.08, left=0.015, right=0.995, top=0.88, bottom=0.12)
    for pi, pan in enumerate(panels):
        inner = outer[pi].subgridspec(3, 7, width_ratios=[1] * 6 + [1.15], wspace=0.0, hspace=0.04)
        rows = [("Density", pan["density"]), ("Samples", pan["samples"]), ("NF", pan["nf"])]
        for r, (name, imgs) in enumerate(rows):
            for c in range(6):
                ax = fig.add_subplot(inner[r, c])
                img = imgs[c]
                ax.imshow(img, origin="lower", extent=pan["extent"], aspect="equal", cmap="viridis", vmin=0, vmax=img.max())
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.3)
                if r == 0:
                    ax.set_title(f"{int(round(FRACTIONS[c]*100))}%", fontsize=7, pad=2)
                if c == 0:
                    ax.set_ylabel(name, fontsize=7, labelpad=2)
        ax = fig.add_subplot(inner[1, 6])
        ax.imshow(pan["target"], origin="lower", extent=pan["extent"], aspect="equal", cmap="viridis", vmin=0, vmax=pan["target"].max())
        ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(inner[2, 6]); ax.axis("off")
        ax.text(0.5, 0.5, "Target", ha="center", va="center", fontsize=8, transform=ax.transAxes)
        fig.text(0.25 + 0.5 * pi, 0.03, f"({'ab'[pi]}) {pan['title']}", ha="center", fontsize=8)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--cnf-iters", type=int, default=None)
    p.add_argument("--nf-iters", type=int, default=None)
    p.add_argument("--viz-samples", type=int, default=None)
    p.add_argument("--sample-points", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--dataset", choices=["both", "circles", "moons"], default="both")
    # ASSUMPTION: hyper-parameters below are not stated in the paper.
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--cnf-lr", type=float, default=1e-3)
    p.add_argument("--nf-lr", type=float, default=1e-5)
    p.add_argument("--nf-momentum", type=float, default=0.9)
    p.add_argument("--hyper-hidden", type=int, default=32)
    p.add_argument("--integration-time", type=float, default=10.0)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-5)
    args = p.parse_args()
    apply_mode(args, PAPER, SMOKE)
    default_dirs(args, "cnf", "cnf/figure5")
    set_seed(args.seed)
    device = get_device(args.device)
    args.device_obj = device
    print(f"[fig5] mode={args.mode} device={device} CNF iters={args.cnf_iters} NF iters={args.nf_iters}")
    keys = ["circles", "moons"] if args.dataset == "both" else [args.dataset]
    panels = [build_panels(k, args, device, Path(args.checkpoint_dir)) for k in keys]
    out = Path(args.output_dir) / "figure5.png"
    plot(panels, out)
    print(f"[fig5] wrote {out}")


if __name__ == "__main__":
    main()
