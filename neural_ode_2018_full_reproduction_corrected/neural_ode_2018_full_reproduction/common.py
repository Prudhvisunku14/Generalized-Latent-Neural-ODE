"""Shared helpers: centralized seed, device, SMOKE/PAPER modes, resumable checkpoints."""

import csv
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent

# Single documented seed for every default run (override with --seed).
SEED = 0

MODES = ("smoke", "paper")


def set_seed(seed=SEED, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Best-effort determinism (some CUDA ops remain non-deterministic).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(name="auto"):
    if name == "auto":
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(0)
            except Exception:  # pragma: no cover - platform-specific fallback path
                print("[warning] CUDA detected but unusable on this system; falling back to CPU.", flush=True)
                return torch.device("cpu")
            return torch.device("cuda")
        return torch.device("cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[warning] --device cuda requested, but CUDA is unavailable on this system; falling back to CPU.", flush=True)
        return torch.device("cpu")
    if device.type == "cuda":
        try:
            torch.cuda.set_device(0)
        except Exception:  # pragma: no cover - platform-specific fallback path
            print("[warning] CUDA requested but the installed driver is incompatible; falling back to CPU.", flush=True)
            return torch.device("cpu")
    return device


def append_training_metrics(path, row):
    """Append a single training record to a CSV file in the results directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp_utc",
        "epoch",
        "model",
        "device",
        "loss",
        "accuracy",
        "learning_rate",
        "elapsed_seconds",
        "nfe_mean",
    ]
    write_header = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "epoch": row.get("epoch", ""),
            "model": row.get("model", ""),
            "device": row.get("device", ""),
            "loss": row.get("loss", ""),
            "accuracy": row.get("accuracy", ""),
            "learning_rate": row.get("learning_rate", ""),
            "elapsed_seconds": row.get("elapsed_seconds", ""),
            "nfe_mean": row.get("nfe_mean", ""),
        })


def add_common_args(parser):
    parser.add_argument("--mode", choices=MODES, default="paper",
                        help="smoke: short functional check; paper: full paper-scale settings")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    return parser


def apply_mode(args, paper, smoke):
    """Fill every argument left at None from the PAPER or SMOKE table (CLI overrides win)."""
    table = paper if args.mode == "paper" else smoke
    for k, v in table.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return args


def default_dirs(args, result_sub, ckpt_sub):
    """SMOKE writes to results_smoke/ so it never overwrites real PAPER results."""
    base = "results" if args.mode == "paper" else "results_smoke"
    ck = "checkpoints" if args.mode == "paper" else "checkpoints_smoke"
    if args.output_dir is None:
        args.output_dir = str(ROOT / base / result_sub)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(ROOT / ck / ckpt_sub)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    return args


def rng_state():
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng_state(st):
    if not st:
        return
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])


def save_ckpt(path, model, optimizer, step, complete=False, extra=None):
    """Atomic checkpoint: model, optimizer, iteration, RNG state."""
    path = Path(path)
    payload = {
        "state": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "complete": bool(complete),
        "rng": rng_state(),
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, optimizer, device):
    """Returns (start_step, complete, extra). Restores RNG state when resuming."""
    path = Path(path)
    if not path.exists():
        return 0, False, {}
    d = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(d["state"])
    if d.get("complete", False):
        return int(d["step"]), True, d.get("extra", {})
    if optimizer is not None and d.get("optimizer") is not None:
        optimizer.load_state_dict(d["optimizer"])
    restore_rng_state(d.get("rng"))
    return int(d.get("step", 0)), False, d.get("extra", {})
