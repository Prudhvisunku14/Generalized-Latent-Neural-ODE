#!/usr/bin/env python3
"""Run every experiment.  python run_all_paper.py --mode smoke|paper"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "paper"], default="paper")
    ap.add_argument("--only", nargs="*", choices=["mnist", "fig4", "fig5", "latent_ode"], default=None)
    ap.add_argument("--seed", type=int, default=0)
    args, extra = ap.parse_known_args()
    jobs = [("mnist", "run_mnist.py"), ("fig4", "run_cnf_fig4.py"), ("fig5", "run_cnf_fig5.py"), ("latent_ode", "run_latent_ode.py")]
    for name, script in jobs:
        if args.only and name not in args.only:
            continue
        cmd = [sys.executable, str(ROOT / script), "--mode", args.mode, "--seed", str(args.seed)] + extra
        print("\n$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    print(f"\nFinished ({args.mode}). Results: {ROOT / ('results' if args.mode == 'paper' else 'results_smoke')}")


if __name__ == "__main__":
    main()
