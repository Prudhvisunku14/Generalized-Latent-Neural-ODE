#!/usr/bin/env python3
"""Section 3 reproduction: MNIST ResNet, RK-Net, ODE-Net and Figure 3.

The architecture follows the authors' public torchdiffeq MNIST reproduction,
while the reported paper targets are kept in the repository README.
"""

import argparse
import copy
import csv
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
from common import add_common_args, apply_mode, append_training_metrics, default_dirs, get_device, load_ckpt, save_ckpt, set_seed  # noqa: E402

# ASSUMPTION: epochs / lr schedule / augmentation are not stated in the paper; they follow the authors'
# public torchdiffeq MNIST example (160 epochs, lr 0.1 decayed x0.1 at epochs 60/100/140, random-crop aug).
PAPER = dict(epochs=160, train_subset=0, test_subset=0)
SMOKE = dict(epochs=2, train_subset=512, test_subset=256)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x
        out = self.relu(self.norm1(x))
        if self.downsample is not None:
            shortcut = self.downsample(out)
        out = self.conv1(out)
        out = self.relu(self.norm2(out))
        out = self.conv2(out)
        return out + shortcut


class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, bias=True):
        super().__init__()
        self.layer = nn.Conv2d(dim_in + 1, dim_out, ksize, stride, padding, bias=bias)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t
        return self.layer(torch.cat([tt, x], dim=1))


class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = norm(dim)
        self.norm2 = norm(dim)
        self.norm3 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.relu(self.norm1(x))
        out = self.conv1(t, out)
        out = self.relu(self.norm2(out))
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):
    def __init__(self, func, tol=1e-3, adjoint=True, method="dopri5"):
        super().__init__()
        self.func = func
        self.tol = tol
        self.adjoint = adjoint
        self.method = method
        self.register_buffer("integration_time", torch.tensor([0.0, 1.0]))

    def forward(self, x):
        t = self.integration_time.type_as(x)
        solver = odeint_adjoint if self.adjoint else odeint
        out = solver(self.func, x, t, rtol=self.tol, atol=self.tol, method=self.method)
        return out[-1]

    @property
    def nfe(self):
        return self.func.nfe

    @nfe.setter
    def nfe(self, value):
        self.func.nfe = value


class RKBlock(nn.Module):
    """Fixed-step RK-Net; direct backpropagation through RK4 evaluations."""
    def __init__(self, func, step_size=0.1):
        super().__init__()
        self.func = func
        self.step_size = step_size
        self.register_buffer("integration_time", torch.tensor([0.0, 1.0]))

    def forward(self, x):
        t = self.integration_time.type_as(x)
        out = odeint(
            self.func,
            x,
            t,
            method="rk4",
            options={"step_size": self.step_size},
        )
        return out[-1]

    @property
    def nfe(self):
        return self.func.nfe

    @nfe.setter
    def nfe(self, value):
        self.func.nfe = value


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


def build_model(kind: str, tol: float, adjoint: bool, rk_step: float) -> nn.Module:
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
        features = [ResBlock(64, 64) for _ in range(6)]
    elif kind == "rknet":
        features = [RKBlock(ODEFunc(64), step_size=rk_step)]
    elif kind == "odenet":
        features = [ODEBlock(ODEFunc(64), tol=tol, adjoint=adjoint)]
    else:
        raise ValueError(kind)
    head = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]
    return nn.Sequential(*stem, *features, *head)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_loaders(root: Path, batch_size: int, test_batch_size: int, mode: str, train_subset: int, test_subset: int):
    train_transform = transforms.Compose([transforms.RandomCrop(28, padding=4), transforms.ToTensor()])
    test_transform = transforms.ToTensor()
    synthetic = False
    try:
        train = datasets.MNIST(root=str(root), train=True, download=True, transform=train_transform)
        train_eval = datasets.MNIST(root=str(root), train=True, download=True, transform=test_transform)
        test = datasets.MNIST(root=str(root), train=False, download=True, transform=test_transform)
    except Exception as err:  # noqa: BLE001
        if mode != "smoke":
            raise RuntimeError(f"MNIST could not be downloaded/loaded: {err}") from err
        # SMOKE ONLY: random images, used purely to exercise the code path when offline.
        print("!" * 70, "\n!! MNIST unavailable -> SYNTHETIC RANDOM DATA (smoke test only, numbers are meaningless)\n" + "!" * 70)
        synthetic = True
        train = datasets.FakeData(size=2048, image_size=(1, 28, 28), num_classes=10, transform=train_transform)
        train_eval = datasets.FakeData(size=2048, image_size=(1, 28, 28), num_classes=10, transform=test_transform)
        test = datasets.FakeData(size=1024, image_size=(1, 28, 28), num_classes=10, transform=test_transform)
    if train_subset:
        n = min(train_subset, len(train))
        train, train_eval = Subset(train, range(n)), Subset(train_eval, range(n))
    if test_subset:
        test = Subset(test, range(min(test_subset, len(test))))
    pin = torch.cuda.is_available()
    nw = 2 if mode == "paper" else 0
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pin),
        DataLoader(train_eval, batch_size=test_batch_size, shuffle=False, num_workers=nw, pin_memory=pin),
        DataLoader(test, batch_size=test_batch_size, shuffle=False, num_workers=nw, pin_memory=pin),
        synthetic,
    )


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def get_dynamic_block(model):
    for m in model.modules():
        if isinstance(m, (ODEBlock, RKBlock)):
            return m
    return None


def train_one(kind, args, device, train_loader, test_loader, ckpt_dir, out_dir=None):
    """Train one model with checkpoint/resume every epoch.

    The reported test error is that of the FINAL epoch (no selection on the test set).
    """
    set_seed(args.seed)
    model = build_model(kind, args.tol, not args.direct_backprop, args.rk_step).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    boundaries = {60: 0.1, 100: 0.01, 140: 0.001}
    ckpt = ckpt_dir / f"{kind}.pt"
    start, complete, extra = load_ckpt(ckpt, model, optimizer, device)
    history_nfe = list(extra.get("nfe_history", []))
    final_acc = extra.get("final_acc", float("nan"))
    out_dir = Path(out_dir) if out_dir is not None else None
    metrics_path = out_dir / f"{kind}_epoch_metrics.csv" if out_dir is not None else None
    if complete:
        return model, final_acc, history_nfe

    for epoch in range(start, args.epochs):
        epoch_start = time.perf_counter()
        decay = 1.0
        for b, d in boundaries.items():
            if epoch >= b:
                decay = d
        for group in optimizer.param_groups:
            group["lr"] = args.lr * decay
        model.train()
        f_nfes = []
        dyn = get_dynamic_block(model)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if dyn is not None:
                dyn.nfe = 0
            logits = model(x)
            if dyn is not None:
                f_nfes.append(dyn.nfe)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        if kind == "odenet":
            history_nfe.append(float(np.mean(f_nfes)))
        final_acc = accuracy(model, test_loader, device)
        elapsed = time.perf_counter() - epoch_start
        print(f"[{kind}] epoch {epoch+1:03d}/{args.epochs} loss={loss.item():.4f} test_acc={final_acc:.4f} elapsed={elapsed:.2f}s", flush=True)
        if metrics_path is not None:
            append_training_metrics(
                metrics_path,
                {
                    "epoch": epoch + 1,
                    "model": kind,
                    "device": str(device),
                    "loss": float(loss.item()),
                    "accuracy": float(final_acc),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_seconds": float(elapsed),
                    "nfe_mean": float(np.mean(f_nfes)) if f_nfes else 0.0,
                },
            )
        save_ckpt(ckpt, model, optimizer, epoch + 1, complete=(epoch + 1 == args.epochs),
                  extra={"nfe_history": history_nfe, "final_acc": final_acc})
    return model, final_acc, history_nfe


def figure3_stats(model, args, device, test_loader, train_nfe_history, out_path):
    """Figure 3: (a) error vs NFE, (b) relative time vs NFE, (c) backward vs forward NFE, (d) NFE during training.

    Every point is one (test batch, tolerance) pair, coloured by the tolerance (1e-0 ... 1e-5, as in the paper).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    model.eval()
    dyn = get_dynamic_block(model)
    if not isinstance(dyn, ODEBlock):
        raise RuntimeError("Figure 3 requires ODE-Net")
    layers = list(model)
    pos = next(i for i, m in enumerate(layers) if m is dyn)
    stem, head = nn.Sequential(*layers[:pos]), nn.Sequential(*layers[pos + 1:])
    criterion = nn.CrossEntropyLoss()

    # float64 high-accuracy reference of the ODE block (tol 1e-10), so errors down to 1e-5 are meaningful
    func64 = copy.deepcopy(dyn.func).double().to(device)
    t64 = torch.tensor([0.0, 1.0], dtype=torch.float64, device=device)
    tols = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    rows = []
    old_tol = dyn.tol
    for bi, (x, y) in enumerate(test_loader):
        if bi >= args.fig3_batches:
            break
        x, y = x[: args.fig3_batch_size].to(device), y[: args.fig3_batch_size].to(device)
        with torch.no_grad():
            h0 = stem(x)
            ref = odeint(func64, h0.double(), t64, rtol=1e-10, atol=1e-10, method="dopri5")[-1]
        for tol in tols:
            dyn.tol = tol
            dyn.func.nfe = 0
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                h1 = dyn(h0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            nfe_f = dyn.func.nfe
            err = (h1.double() - ref).abs().mean().item()
            # backward NFE (adjoint) for the same tolerance
            model.zero_grad(set_to_none=True)
            dyn.func.nfe = 0
            loss = criterion(model(x), y)
            nfe_fwd_again = dyn.func.nfe
            dyn.func.nfe = 0
            loss.backward()
            nfe_b = dyn.func.nfe
            rows.append((tol, nfe_f, err, elapsed, nfe_b))
            print(f"[figure3] batch={bi} tol={tol:.0e} err={err:.2e} NFE-F={nfe_f} NFE-B={nfe_b} t={elapsed*1e3:.1f}ms", flush=True)
    dyn.tol = old_tol
    model.zero_grad(set_to_none=True)

    tol_a, nf_a, err_a, t_a, nb_a = (np.array(c) for c in zip(*rows))
    rel_t = t_a / t_a.max()
    norm = LogNorm(vmin=1e-5, vmax=1.0)
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.3), dpi=200, gridspec_kw={"wspace": 0.75})
    sc = None
    for ax, (xs, ys, xl, yl, logy) in zip(axes[:3], [
        (nf_a, err_a, "(a) NFE Forward", "Numerical Error", True),
        (nf_a, rel_t, "(b) NFE Forward", "Relative Time", False),
        (nf_a, nb_a, "(c) NFE Forward", "NFE Backward", False),
    ]):
        sc = ax.scatter(xs, ys, c=tol_a, cmap="rainbow", norm=norm, s=14, linewidths=0)
        ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7)
        ax.tick_params(labelsize=6)
        if logy:
            ax.set_yscale("log")
        cb = fig.colorbar(sc, ax=ax, pad=0.04, fraction=0.09)
        cb.ax.tick_params(labelsize=5)
        cb.ax.invert_yaxis()
    lim = max(nf_a.max(), nb_a.max()) * 1.05
    axes[2].plot([0, lim], [0, lim], "k--", lw=0.5)
    axes[2].set_xlim(0, lim); axes[2].set_ylim(0, lim)
    axes[3].plot(np.arange(1, len(train_nfe_history) + 1), train_nfe_history, lw=1.2)
    axes[3].set_xlabel("(d) Training Epoch", fontsize=7); axes[3].set_ylabel("NFE Forward", fontsize=7)
    axes[3].tick_params(labelsize=6); axes[3].grid(True, lw=0.3)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--only", choices=["all", "table1", "figure3"], default="all")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--test-subset", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-batch-size", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--tol", type=float, default=1e-3, help="paper Sec. 6: tolerance 1e-3 for classification")
    p.add_argument("--rk-step", type=float, default=0.1, help="ASSUMPTION: RK4 step size of RK-Net (not stated)")
    p.add_argument("--direct-backprop", action="store_true", help="Disable adjoint for ODE-Net")
    p.add_argument("--fig3-batches", type=int, default=None)
    p.add_argument("--fig3-batch-size", type=int, default=None)
    p.add_argument("--data-dir", default=None)
    args = p.parse_args()
    apply_mode(args, PAPER, SMOKE)
    args.fig3_batches = args.fig3_batches or (8 if args.mode == "paper" else 2)
    args.fig3_batch_size = args.fig3_batch_size or (128 if args.mode == "paper" else 32)
    default_dirs(args, "mnist", "mnist")
    args.data_dir = args.data_dir or str(Path(__file__).resolve().parent.parent / "data")
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir, ckpt_dir = Path(args.output_dir), Path(args.checkpoint_dir)
    print(f"[mnist] mode={args.mode} device={device} epochs={args.epochs}")
    train_loader, _, test_loader, synthetic = get_loaders(Path(args.data_dir), args.batch_size, args.test_batch_size,
                                                          args.mode, args.train_subset, args.test_subset)
    trained, nfe_hist = {}, []
    if args.only in ("all", "table1"):
        rows = []
        for kind, label in [("resnet", "ResNet"), ("rknet", "RK-Net"), ("odenet", "ODE-Net")]:
            model, acc, hist = train_one(kind, args, device, train_loader, test_loader, ckpt_dir, out_dir)
            trained[kind] = model
            if kind == "odenet":
                nfe_hist = hist
            rows.append({"Model": label, "Test Error (%)": round(100.0 * (1.0 - acc), 2),
                         "# Params (M)": round(count_parameters(model) / 1e6, 2)})
            print(f"[mnist] {label}: test error {rows[-1]['Test Error (%)']}%  params {rows[-1]['# Params (M)']} M", flush=True)
        with open(out_dir / "table1.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if args.only in ("all", "figure3"):
        if "odenet" in trained:
            model = trained["odenet"]
        else:
            model, _, nfe_hist = train_one("odenet", args, device, train_loader, test_loader, ckpt_dir, out_dir)
        figure3_stats(model, args, device, test_loader, nfe_hist, out_dir / "figure3.png")
    if synthetic:
        (out_dir / "SYNTHETIC_DATA_SMOKE_ONLY.txt").write_text("These outputs come from random fake data; they only prove the code runs.\n")
    print(f"[mnist] outputs in {out_dir}")


if __name__ == "__main__":
    main()
