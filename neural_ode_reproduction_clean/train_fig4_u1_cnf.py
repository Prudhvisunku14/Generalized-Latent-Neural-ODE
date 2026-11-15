#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
CNF_DIR = ROOT / "02_cnf"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CNF_DIR))

from common import (
    add_common_args,
    apply_mode,
    default_dirs,
    device_summary,
    get_device,
    save_system_info,
    set_seed,
    write_csv,
)

from figure4_density_matching import train, final_loss
from flows import ContinuousNormalizingFlow
from targets import TARGETS


PAPER = {
    "cnf_iters": 10000,
    "sizes": [2, 4, 8, 16, 32],
    "eval_samples": 50000,
    "checkpoint_every": 500,
    "cnf_steps_per_epoch": 100,
}

SMOKE = {
    "cnf_iters": 20,
    "sizes": [2, 8, 32],
    "eval_samples": 2000,
    "checkpoint_every": 10,
    "cnf_steps_per_epoch": 5,
}


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)

    p.add_argument("--cnf-iters", type=int, default=None)
    p.add_argument("--sizes", type=int, nargs="+", default=None)
    p.add_argument("--eval-samples", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--cnf-steps-per-epoch", type=int, default=None)

    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--cnf-lr", type=float, default=1e-3)
    p.add_argument("--hyper-hidden", type=int, default=32)
    p.add_argument("--integration-time", type=float, default=10.0)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-5)

    args = p.parse_args()

    apply_mode(args, PAPER, SMOKE)

    default_dirs(
        args,
        "exp2_cnf",
        "exp2_cnf/figure4"
    )

    set_seed(args.seed, deterministic=args.deterministic)

    device = get_device(
        args.device,
        args.gpu,
        args.require_cuda
    )

    args.device_obj = device

    out_dir = Path(args.output_dir)
    ckpt_dir = Path(args.checkpoint_dir)

    save_system_info(
        out_dir / "system_info_u1_cnf_only.json",
        device,
        args
    )

    print(
        f"[Exp2/Fig4 U1-CNF ONLY] "
        f"mode={args.mode} | "
        f"{device_summary(device)} | "
        f"CNF={args.cnf_iters} steps | "
        f"widths={args.sizes}",
        flush=True,
    )

    target_fn = TARGETS["u1"]

    rows = []

    for m in args.sizes:

        print(
            f"\n==============================\n"
            f" U1 | CNF M={m}\n"
            f"==============================",
            flush=True,
        )

        model = ContinuousNormalizingFlow(
            width=m,
            hidden_dim=args.hyper_hidden,
            integration_time=args.integration_time,
            rtol=args.rtol,
            atol=args.atol,
            learn_base=True,
            base_variance=1.0,
            adjoint=True,
        ).to(device)

        ckpt = ckpt_dir / f"u1_cnf_m{m}.pt"

        metrics = (
            out_dir /
            "training" /
            f"u1_cnf_m{m}.csv"
        )

        model = train(
            model,
            "cnf",
            target_fn,
            args.cnf_iters,
            lambda mod: torch.optim.Adam(
                mod.parameters(),
                lr=args.cnf_lr
            ),
            args,
            ckpt,
            f"[u1] CNF M={m}",
            args.cnf_steps_per_epoch,
            metrics,
        )

        value = final_loss(
            model,
            "cnf",
            target_fn,
            args.eval_samples
        )

        rows.append(
            {
                "Target": "u1",
                "Capacity": m,
                "Model": "CNF",
                "Final KL*": value,
            }
        )

        print(
            f"[u1] CNF M={m} COMPLETE | "
            f"Final KL*={value:.6f} | "
            f"checkpoint={ckpt}",
            flush=True,
        )

    write_csv(
        out_dir / "u1_cnf_only_losses.csv",
        rows
    )

    print("\nU1 CNF TRAINING FINISHED", flush=True)
    print(f"Checkpoints : {ckpt_dir}", flush=True)
    print(
        f"Results     : "
        f"{out_dir / 'u1_cnf_only_losses.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
