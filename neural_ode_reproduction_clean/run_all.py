#!/usr/bin/env python3
"""Run Experiments 1, 2 and 3 sequentially on one computer/GPU."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "paper"], default="paper")
    p.add_argument("--only", nargs="*", choices=["exp1", "exp2", "exp3"], default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    args = p.parse_args()

    common = ["--mode", args.mode, "--seed", str(args.seed), "--device", args.device, "--gpu", str(args.gpu)]
    if args.require_cuda:
        common.append("--require-cuda")
    if args.deterministic:
        common.append("--deterministic")
    if args.fresh:
        common.append("--fresh")

    jobs = [
        ("exp1", ROOT / "run_exp1.py", []),
        ("exp2", ROOT / "run_exp2.py", ["--no-gif"] if args.no_gif else []),
        ("exp3", ROOT / "run_exp3.py", ["--no-gif"] if args.no_gif else []),
    ]
    for name, script, extra in jobs:
        if args.only and name not in args.only:
            continue
        cmd = [sys.executable, str(script), *common, *extra]
        print("\n$ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    result_root = ROOT / ("results" if args.mode == "paper" else "results_smoke")
    print(f"\nFinished. Results are under: {result_root}")


if __name__ == "__main__":
    main()
