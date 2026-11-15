#!/usr/bin/env python3
"""Easy entry point for Experiment 2 (Figure 4 + Figure 5)."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "paper"], default="paper")
    p.add_argument("--part", choices=["all", "figure4", "figure5"], default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    args = p.parse_args()

    common = [
        "--mode", args.mode,
        "--seed", str(args.seed),
        "--device", args.device,
        "--gpu", str(args.gpu),
    ]
    if args.require_cuda:
        common.append("--require-cuda")
    if args.deterministic:
        common.append("--deterministic")
    if args.fresh:
        common.append("--fresh")

    jobs = []
    if args.part in ("all", "figure4"):
        jobs.append(ROOT / "02_cnf" / "figure4_density_matching.py")
    if args.part in ("all", "figure5"):
        jobs.append(ROOT / "02_cnf" / "figure5_maximum_likelihood.py")

    for script in jobs:
        cmd = [sys.executable, str(script), *common]
        if script.name.startswith("figure5") and args.no_gif:
            cmd.append("--no-gif")
        print("\n$ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
