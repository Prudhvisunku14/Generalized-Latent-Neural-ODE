"""Shared utilities for the Neural ODE reproduction project.

The helpers in this file keep the three experiments independent while giving them
consistent GPU selection, timing, CSV logging and checkpoint/resume behaviour.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SEED = 0
MODES = ("smoke", "paper")


def set_seed(seed: int = SEED, deterministic: bool = False) -> None:
    """Seed Python/NumPy/PyTorch.

    ``deterministic=False`` is the default because the user explicitly wants to use
    the GPU efficiently.  Pass ``--deterministic`` when exact repeatability is more
    important than peak speed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        # Safe performance setting for fixed-shape convolutions used in Exp. 1.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def get_device(name: str = "auto", gpu: int = 0, require_cuda: bool = False) -> torch.device:
    """Return the requested device and select the requested CUDA index correctly."""

    if name == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu}")
        elif require_cuda:
            raise RuntimeError(
                "CUDA was required but PyTorch cannot see an NVIDIA GPU. "
                "Run `python check_gpu.py` and install a CUDA-enabled PyTorch build."
            )
        else:
            return torch.device("cpu")
    else:
        device = torch.device(name)
        if device.type == "cuda":
            index = device.index if device.index is not None else gpu
            device = torch.device(f"cuda:{index}")
            if not torch.cuda.is_available():
                if require_cuda:
                    raise RuntimeError("CUDA device requested, but CUDA is unavailable in this PyTorch installation.")
                print("[warning] CUDA requested but unavailable; using CPU.", flush=True)
                return torch.device("cpu")

    if device.type == "cuda":
        index = device.index if device.index is not None else gpu
        try:
            torch.cuda.set_device(index)
            # Trigger CUDA initialization now, so failures appear before a long run starts.
            _ = torch.empty(1, device=device)
        except Exception as exc:  # pragma: no cover - hardware/driver dependent
            if require_cuda:
                raise RuntimeError(f"CUDA device cuda:{index} could not be initialized: {exc}") from exc
            print(f"[warning] CUDA initialization failed ({exc}); using CPU.", flush=True)
            return torch.device("cpu")
    return device


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--mode", choices=MODES, default="paper",
                        help="smoke: short functional check; paper: paper-scale training")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, cuda:1, ...")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index used when --device=auto/cuda")
    parser.add_argument("--require-cuda", action="store_true",
                        help="fail instead of silently using CPU when CUDA is unavailable")
    parser.add_argument("--deterministic", action="store_true",
                        help="favor reproducibility over maximum cuDNN performance")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore/delete matching checkpoints and train from the beginning")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    return parser


def apply_mode(args: argparse.Namespace, paper: dict[str, Any], smoke: dict[str, Any]) -> argparse.Namespace:
    table = paper if args.mode == "paper" else smoke
    for key, value in table.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def default_dirs(args: argparse.Namespace, result_sub: str, ckpt_sub: str) -> argparse.Namespace:
    result_root = "results" if args.mode == "paper" else "results_smoke"
    ckpt_root = "checkpoints" if args.mode == "paper" else "checkpoints_smoke"
    if args.output_dir is None:
        args.output_dir = str(ROOT / result_root / result_sub)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(ROOT / ckpt_root / ckpt_sub)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    return args


def maybe_remove_checkpoint(path: Path, fresh: bool) -> None:
    if fresh and path.exists():
        path.unlink()


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([x.detach().to(device="cpu", dtype=torch.uint8).contiguous() for x in state["cuda"]])


def save_ckpt(path: Path | str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None,
              step: int, complete: bool = False, extra: dict[str, Any] | None = None) -> None:
    """Atomically save model/optimizer/RNG state."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def load_ckpt(path: Path | str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None,
              device: torch.device) -> tuple[int, bool, dict[str, Any]]:
    """Load a checkpoint and restore optimizer/RNG even if the prior run was marked complete.

    This lets a user increase ``--epochs``/``--iterations`` later and continue from the
    completed checkpoint instead of being forced to restart.
    """

    path = Path(path)
    if not path.exists():
        return 0, False, {}
    data = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(data["state"])
    if optimizer is not None and data.get("optimizer") is not None:
        optimizer.load_state_dict(data["optimizer"])
    restore_rng_state(data.get("rng"))
    return int(data.get("step", 0)), bool(data.get("complete", False)), data.get("extra", {})


def append_csv(path: Path | str, row: dict[str, Any], fieldnames: Iterable[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or row.keys())
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path | str, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    names = list(fieldnames or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "--:--:--"
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class ProgressTimer:
    total_units: int
    device: torch.device
    start_unit: int = 0

    def __post_init__(self) -> None:
        sync_device(self.device)
        self.run_start = time.perf_counter()
        self.unit_start = self.run_start
        self.completed_this_run = 0

    def start(self) -> None:
        sync_device(self.device)
        self.unit_start = time.perf_counter()

    def finish(self, global_completed: int) -> dict[str, float]:
        sync_device(self.device)
        now = time.perf_counter()
        unit_seconds = now - self.unit_start
        total_seconds = now - self.run_start
        self.completed_this_run += 1
        mean_seconds = total_seconds / max(1, self.completed_this_run)
        remaining = max(0, self.total_units - global_completed)
        eta_seconds = mean_seconds * remaining
        return {
            "unit_seconds": unit_seconds,
            "total_seconds": total_seconds,
            "eta_seconds": eta_seconds,
        }


def gpu_peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def reset_gpu_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def device_summary(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    idx = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return f"cuda:{idx} | {props.name} | {props.total_memory / (1024**3):.1f} GB VRAM"


def save_system_info(path: Path | str, device: torch.device, args: argparse.Namespace | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {
        "timestamp_utc": utc_timestamp(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update({
            "gpu_name": props.name,
            "gpu_vram_gb": round(props.total_memory / (1024**3), 3),
            "gpu_capability": f"{props.major}.{props.minor}",
        })
    if args is not None:
        info["args"] = vars(args)
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
