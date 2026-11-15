#!/usr/bin/env python3
"""Experiment 1: MNIST MLP/ResNet/RK-Net/ODE-Net + Figure-3 ODE statistics.

The ResNet/ODE-Net architecture follows the authors' public ``odenet_mnist.py``
reproduction (convolutional downsampling, six residual blocks, GroupNorm).  The
1-layer MLP is included because the paper reports it in Table 1, although that
paper value is a LeCun et al. reference rather than a model trained by Chen et al.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchdiffeq import odeint, odeint_adjoint

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
    sync_device,
    utc_timestamp,
    write_csv,
)

# The epochs/lr schedule/augmentation are not stated in the paper; these follow
# the authors' public torchdiffeq MNIST reproduction.
PAPER = dict(epochs=160, train_subset=0, test_subset=0, fig3_batches=8, fig3_batch_size=128)
SMOKE = dict(epochs=2, train_subset=512, test_subset=256, fig3_batches=2, fig3_batch_size=32)

PAPER_TABLE = {
    "mlp": {"label": "1-Layer MLP", "error": 1.60, "params_m": 0.24, "memory": "-", "time": "-"},
    "resnet": {"label": "ResNet", "error": 0.41, "params_m": 0.60, "memory": "O(L)", "time": "O(L)"},
    "rknet": {"label": "RK-Net", "error": 0.47, "params_m": 0.22, "memory": "O(L~)", "time": "O(L~)"},
    "odenet": {"label": "ODE-Net", "error": 0.42, "params_m": 0.22, "memory": "O(1)", "time": "O(L~)"},
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def norm(dim: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        out = self.relu(self.norm1(x))
        if self.downsample is not None:
            shortcut = self.downsample(out)
        out = self.conv1(out)
        out = self.relu(self.norm2(out))
        out = self.conv2(out)
        return out + shortcut


class ODEFunc(nn.Module):
    """Time-independent hidden dynamics used in the public MNIST reproduction."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = norm(dim)
        self.norm2 = norm(dim)
        self.norm3 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(dim, dim)
        self.conv2 = conv3x3(dim, dim)
        self.nfe = 0

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        self.nfe += 1
        out = self.relu(self.norm1(x))
        out = self.conv1(out)
        out = self.relu(self.norm2(out))
        out = self.conv2(out)
        return self.norm3(out)


class ODEBlock(nn.Module):
    def __init__(self, func: ODEFunc, tol: float = 1e-3, adjoint: bool = True, method: str = "dopri5"):
        super().__init__()
        self.func = func
        self.tol = tol
        self.adjoint = adjoint
        self.method = method
        self.register_buffer("integration_time", torch.tensor([0.0, 1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        solver = odeint_adjoint if self.adjoint else odeint
        t = self.integration_time.to(dtype=x.dtype, device=x.device)
        return solver(self.func, x, t, rtol=self.tol, atol=self.tol, method=self.method)[-1]

    @property
    def nfe(self) -> int:
        return self.func.nfe

    @nfe.setter
    def nfe(self, value: int) -> None:
        self.func.nfe = value


class RKBlock(nn.Module):
    """Fixed-step RK4 network with direct backpropagation through solver operations.

    The paper identifies an RK-Net but does not state the fixed step size.  ``0.1``
    remains an explicit reproduction assumption and can be changed from the CLI.
    """

    def __init__(self, func: ODEFunc, step_size: float = 0.1):
        super().__init__()
        self.func = func
        self.step_size = float(step_size)
        self.register_buffer("integration_time", torch.tensor([0.0, 1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.integration_time.to(dtype=x.dtype, device=x.device)
        return odeint(self.func, x, t, method="rk4", options={"step_size": self.step_size})[-1]

    @property
    def nfe(self) -> int:
        return self.func.nfe

    @nfe.setter
    def nfe(self, value: int) -> None:
        self.func.nfe = value


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)


def build_model(kind: str, tol: float, adjoint: bool, rk_step: float) -> nn.Module:
    if kind == "mlp":
        # 784*300 + 300 + 300*10 + 10 = 238,510 ~= 0.24 M parameters.
        return nn.Sequential(Flatten(), nn.Linear(28 * 28, 300), nn.ReLU(inplace=True), nn.Linear(300, 10))

    stem = [
        nn.Conv2d(1, 64, 3, 1),
        norm(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1),
        norm(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1),
    ]
    if kind == "resnet":
        features: list[nn.Module] = [ResBlock(64, 64) for _ in range(6)]
    elif kind == "rknet":
        features = [RKBlock(ODEFunc(64), step_size=rk_step)]
    elif kind == "odenet":
        features = [ODEBlock(ODEFunc(64), tol=tol, adjoint=adjoint)]
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    head = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]
    return nn.Sequential(*stem, *features, *head)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dynamic_block(model: nn.Module) -> ODEBlock | RKBlock | None:
    for module in model.modules():
        if isinstance(module, (ODEBlock, RKBlock)):
            return module
    return None


def get_loaders(root: Path, batch_size: int, test_batch_size: int, mode: str,
                train_subset: int, test_subset: int, device: torch.device,
                workers: int) -> tuple[DataLoader, DataLoader, bool]:
    train_transform = transforms.Compose([transforms.RandomCrop(28, padding=4), transforms.ToTensor()])
    test_transform = transforms.ToTensor()
    synthetic = False
    try:
        train = datasets.MNIST(root=str(root), train=True, download=True, transform=train_transform)
        test = datasets.MNIST(root=str(root), train=False, download=True, transform=test_transform)
    except Exception as err:  # noqa: BLE001
        if mode != "smoke":
            raise RuntimeError(f"MNIST could not be downloaded/loaded: {err}") from err
        print("!" * 78)
        print("MNIST unavailable -> using synthetic random data for SMOKE mode only.")
        print("The resulting accuracy values are meaningless; this only tests the pipeline.")
        print("!" * 78)
        synthetic = True
        train = datasets.FakeData(size=2048, image_size=(1, 28, 28), num_classes=10, transform=train_transform)
        test = datasets.FakeData(size=1024, image_size=(1, 28, 28), num_classes=10, transform=test_transform)

    if train_subset:
        train = Subset(train, range(min(train_subset, len(train))))
    if test_subset:
        test = Subset(test, range(min(test_subset, len(test))))

    pin = device.type == "cuda"
    loader_kwargs: dict[str, object] = dict(num_workers=workers, pin_memory=pin)
    if workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False, **loader_kwargs),
        DataLoader(test, batch_size=test_batch_size, shuffle=False, drop_last=False, **loader_kwargs),
        synthetic,
    )


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=device.type == "cuda")
        y = y.to(device, non_blocking=device.type == "cuda")
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def train_one(kind: str, args: argparse.Namespace, device: torch.device,
              train_loader: DataLoader, test_loader: DataLoader,
              ckpt_dir: Path, out_dir: Path) -> tuple[nn.Module, dict[str, float], list[float]]:
    """Train one model and save one CSV row per epoch.

    ODE/RK models record mean forward/backward NFE separately.  The terminal shows
    epoch time, total elapsed time, ETA and peak GPU memory for every epoch.
    """

    set_seed(args.seed, deterministic=args.deterministic)
    model = build_model(kind, args.tol, not args.direct_backprop, args.rk_step).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    boundaries = {60: 0.1, 100: 0.01, 140: 0.001}

    ckpt = ckpt_dir / f"{kind}.pt"
    maybe_remove_checkpoint(ckpt, args.fresh)
    metrics_path = out_dir / "training" / f"{kind}_epochs.csv"
    if args.fresh and metrics_path.exists():
        metrics_path.unlink()

    start, _, extra = load_ckpt(ckpt, model, optimizer, device)
    history_nfe = list(extra.get("forward_nfe_history", []))
    prior_elapsed = float(extra.get("total_train_seconds", 0.0))
    peak_gpu_mb = float(extra.get("peak_gpu_mb", 0.0))
    final_acc = float(extra.get("final_acc", float("nan")))

    if start >= args.epochs:
        return model, {
            "final_acc": final_acc,
            "total_train_seconds": prior_elapsed,
            "peak_gpu_mb": peak_gpu_mb,
        }, history_nfe

    timer = ProgressTimer(args.epochs, device, start_unit=start)
    last_total_elapsed = prior_elapsed
    print(f"\n--- {PAPER_TABLE[kind]['label']} | starting at epoch {start + 1}/{args.epochs} ---", flush=True)

    for epoch in range(start, args.epochs):
        timer.start()
        reset_gpu_peak_memory(device)

        decay = 1.0
        for boundary, factor in boundaries.items():
            if epoch >= boundary:
                decay = factor
        for group in optimizer.param_groups:
            group["lr"] = args.lr * decay

        model.train()
        loss_sum = 0.0
        n_samples = 0
        forward_nfes: list[int] = []
        backward_nfes: list[int] = []
        dyn = get_dynamic_block(model)

        for x, y in train_loader:
            x = x.to(device, non_blocking=device.type == "cuda")
            y = y.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)

            if dyn is not None:
                dyn.nfe = 0
            logits = model(x)
            if dyn is not None:
                forward_nfes.append(int(dyn.nfe))
                dyn.nfe = 0

            loss = criterion(logits, y)
            loss.backward()
            if dyn is not None:
                backward_nfes.append(int(dyn.nfe))
                dyn.nfe = 0
            optimizer.step()

            bs = y.shape[0]
            loss_sum += float(loss.detach()) * bs
            n_samples += bs

        avg_loss = loss_sum / max(1, n_samples)
        if kind == "odenet":
            history_nfe.append(float(np.mean(forward_nfes)))

        final_acc = accuracy(model, test_loader, device)
        timing = timer.finish(epoch + 1)
        current_peak = gpu_peak_memory_mb(device)
        peak_gpu_mb = max(peak_gpu_mb, current_peak)
        total_elapsed = prior_elapsed + timing["total_seconds"]
        last_total_elapsed = total_elapsed
        f_nfe = float(np.mean(forward_nfes)) if forward_nfes else 0.0
        b_nfe = float(np.mean(backward_nfes)) if backward_nfes else 0.0

        print(
            f"[{kind:6s}] Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"loss {avg_loss:.5f} | test {100*final_acc:.3f}% | "
            f"NFE-F {f_nfe:.1f} | NFE-B {b_nfe:.1f} | "
            f"epoch {format_duration(timing['unit_seconds'])} | "
            f"total {format_duration(total_elapsed)} | ETA {format_duration(timing['eta_seconds'])} | "
            f"peak GPU {current_peak:.0f} MB",
            flush=True,
        )

        append_csv(metrics_path, {
            "timestamp_utc": utc_timestamp(),
            "epoch": epoch + 1,
            "model": kind,
            "device": str(device),
            "train_loss": avg_loss,
            "test_accuracy": final_acc,
            "test_error_percent": 100.0 * (1.0 - final_acc),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "nfe_forward_mean": f_nfe,
            "nfe_backward_mean": b_nfe,
            "epoch_seconds": timing["unit_seconds"],
            "total_elapsed_seconds": total_elapsed,
            "eta_seconds": timing["eta_seconds"],
            "peak_gpu_memory_mb": current_peak,
        })

        save_ckpt(
            ckpt, model, optimizer, epoch + 1, complete=(epoch + 1 >= args.epochs),
            extra={
                "forward_nfe_history": history_nfe,
                "final_acc": final_acc,
                "total_train_seconds": total_elapsed,
                "peak_gpu_mb": peak_gpu_mb,
            },
        )

    return model, {
        "final_acc": final_acc,
        "total_train_seconds": last_total_elapsed,
        "peak_gpu_mb": peak_gpu_mb,
    }, history_nfe


def figure3_stats(model: nn.Module, args: argparse.Namespace, device: torch.device,
                  test_loader: DataLoader, train_nfe_history: list[float], out_dir: Path) -> None:
    """Create Figure-3-style plots and save the underlying measurements as CSV."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    model.eval()
    dyn = get_dynamic_block(model)
    if not isinstance(dyn, ODEBlock):
        raise RuntimeError("Figure 3 requires ODE-Net")

    layers = list(model)
    pos = next(i for i, module in enumerate(layers) if module is dyn)
    stem = nn.Sequential(*layers[:pos])
    criterion = nn.CrossEntropyLoss()

    # High-accuracy float64 reference trajectory for the ODE block.
    func64 = copy.deepcopy(dyn.func).double().to(device)
    t64 = torch.tensor([0.0, 1.0], dtype=torch.float64, device=device)
    tols = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    rows: list[dict[str, float | int]] = []
    old_tol = dyn.tol

    for batch_i, (x, y) in enumerate(test_loader):
        if batch_i >= args.fig3_batches:
            break
        x = x[: args.fig3_batch_size].to(device, non_blocking=device.type == "cuda")
        y = y[: args.fig3_batch_size].to(device, non_blocking=device.type == "cuda")
        with torch.no_grad():
            h0 = stem(x)
            ref = odeint(func64, h0.double(), t64, rtol=1e-10, atol=1e-10, method="dopri5")[-1]

        for tol in tols:
            dyn.tol = tol
            dyn.nfe = 0
            sync_device(device)
            started = time.perf_counter()
            with torch.no_grad():
                h1 = dyn(h0)
            sync_device(device)
            elapsed = time.perf_counter() - started
            nfe_forward = int(dyn.nfe)
            error = float((h1.double() - ref).abs().mean())

            model.zero_grad(set_to_none=True)
            dyn.nfe = 0
            loss = criterion(model(x), y)
            nfe_forward_for_backward_run = int(dyn.nfe)
            dyn.nfe = 0
            loss.backward()
            nfe_backward = int(dyn.nfe)
            dyn.nfe = 0

            rows.append({
                "batch": batch_i,
                "tolerance": tol,
                "nfe_forward": nfe_forward,
                "numerical_error": error,
                "elapsed_seconds": elapsed,
                "nfe_forward_backward_run": nfe_forward_for_backward_run,
                "nfe_backward": nfe_backward,
            })
            print(
                f"[figure3] batch={batch_i} tol={tol:.0e} error={error:.3e} "
                f"NFE-F={nfe_forward} NFE-B={nfe_backward} time={elapsed*1e3:.1f} ms",
                flush=True,
            )

    dyn.tol = old_tol
    model.zero_grad(set_to_none=True)
    if not rows:
        raise RuntimeError("No Figure 3 measurements were produced.")

    max_time = max(float(r["elapsed_seconds"]) for r in rows)
    for row in rows:
        row["relative_time"] = float(row["elapsed_seconds"]) / max_time if max_time > 0 else math.nan
    write_csv(out_dir / "figure3_stats.csv", rows)

    tol_a = np.asarray([float(r["tolerance"]) for r in rows])
    nf_a = np.asarray([float(r["nfe_forward"]) for r in rows])
    err_a = np.asarray([float(r["numerical_error"]) for r in rows])
    rel_t = np.asarray([float(r["relative_time"]) for r in rows])
    nb_a = np.asarray([float(r["nfe_backward"]) for r in rows])

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 2.8), dpi=180, constrained_layout=True)
    norm_color = LogNorm(vmin=1e-5, vmax=1.0)
    specs = [
        (nf_a, err_a, "NFE Forward", "Numerical Error", True),
        (nf_a, rel_t, "NFE Forward", "Relative Time", False),
        (nf_a, nb_a, "NFE Forward", "NFE Backward", False),
    ]
    for ax, (xs, ys, xlabel, ylabel, log_y) in zip(axes[:3], specs):
        sc = ax.scatter(xs, ys, c=tol_a, cmap="viridis", norm=norm_color, s=20)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if log_y:
            ax.set_yscale("log")
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("Solver tolerance")

    lim = max(float(nf_a.max()), float(nb_a.max())) * 1.05
    axes[2].plot([0, lim], [0, lim], linestyle="--", linewidth=0.8)
    axes[2].set_xlim(0, lim)
    axes[2].set_ylim(0, lim)

    epochs = np.arange(1, len(train_nfe_history) + 1)
    axes[3].plot(epochs, train_nfe_history, linewidth=1.3)
    axes[3].set_xlabel("Training Epoch")
    axes[3].set_ylabel("NFE Forward")
    axes[3].grid(alpha=0.25)

    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"figure3.{suffix}", dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--only", choices=["all", "table1", "figure3"], default="all")
    parser.add_argument("--models", nargs="+", choices=list(PAPER_TABLE), default=list(PAPER_TABLE))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers; default=min(8, CPU_count)")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--tol", type=float, default=1e-3, help="paper Sec. 6 classification tolerance")
    parser.add_argument("--rk-step", type=float, default=0.1, help="reproduction assumption; not stated in paper")
    parser.add_argument("--direct-backprop", action="store_true", help="disable adjoint for ODE-Net")
    parser.add_argument("--fig3-batches", type=int, default=None)
    parser.add_argument("--fig3-batch-size", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    apply_mode(args, PAPER, SMOKE)
    default_dirs(args, "exp1_mnist", "exp1_mnist")
    args.data_dir = args.data_dir or str(Path(__file__).resolve().parent.parent / "data")
    args.workers = args.workers if args.workers is not None else min(8, os_cpu_count())

    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device(args.device, args.gpu, args.require_cuda)
    out_dir, ckpt_dir = Path(args.output_dir), Path(args.checkpoint_dir)
    save_system_info(out_dir / "system_info.json", device, args)

    print(f"[Exp1] mode={args.mode} | {device_summary(device)} | epochs={args.epochs} | seed={args.seed}")
    train_loader, test_loader, synthetic = get_loaders(
        Path(args.data_dir), args.batch_size, args.test_batch_size, args.mode,
        args.train_subset, args.test_subset, device, args.workers,
    )

    trained: dict[str, nn.Module] = {}
    nfe_history: list[float] = []
    table_rows: list[dict[str, object]] = []

    if args.only in ("all", "table1"):
        for kind in args.models:
            model, stats, hist = train_one(kind, args, device, train_loader, test_loader, ckpt_dir, out_dir)
            trained[kind] = model
            if kind == "odenet":
                nfe_history = hist
            paper = PAPER_TABLE[kind]
            reproduced_error = 100.0 * (1.0 - stats["final_acc"])
            table_rows.append({
                "Model": paper["label"],
                "Paper Test Error (%)": paper["error"],
                "Reproduced Test Error (%)": round(reproduced_error, 4),
                "Absolute Error Difference (pp)": round(abs(reproduced_error - float(paper["error"])), 4),
                "Paper # Params (M)": paper["params_m"],
                "Reproduced # Params (M)": round(count_parameters(model) / 1e6, 4),
                "Paper Memory Complexity": paper["memory"],
                "Measured Peak GPU Memory (MB)": round(stats["peak_gpu_mb"], 2),
                "Training Time (s)": round(stats["total_train_seconds"], 2),
            })

        write_csv(out_dir / "table1_comparison.csv", table_rows)
        for row in table_rows:
            print(
                f"[Exp1] {row['Model']}: reproduced error={row['Reproduced Test Error (%)']}% | "
                f"paper={row['Paper Test Error (%)']}% | params={row['Reproduced # Params (M)']} M",
                flush=True,
            )

    if args.only in ("all", "figure3"):
        if "odenet" in trained:
            odenet_model = trained["odenet"]
        else:
            odenet_model, _, nfe_history = train_one("odenet", args, device, train_loader, test_loader, ckpt_dir, out_dir)
        figure3_stats(odenet_model, args, device, test_loader, nfe_history, out_dir)

    if synthetic:
        (out_dir / "SYNTHETIC_DATA_SMOKE_ONLY.txt").write_text(
            "These SMOKE outputs used random fake data because MNIST was unavailable.\n",
            encoding="utf-8",
        )
    print(f"[Exp1] outputs: {out_dir}")


def os_cpu_count() -> int:
    import os
    return max(1, os.cpu_count() or 1)


if __name__ == "__main__":
    main()
