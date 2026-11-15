#!/usr/bin/env python3
"""Quick CUDA/GPU check before starting long training jobs."""

from __future__ import annotations

import argparse
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to test")
    args = parser.parse_args()

    print(f"PyTorch           : {torch.__version__}")
    print(f"CUDA build        : {torch.version.cuda}")
    print(f"CUDA available    : {torch.cuda.is_available()}")
    print(f"GPU count         : {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print("CUDA tensor test  : FAIL")
        print("Install a CUDA-enabled PyTorch build before running PAPER mode on your GPU.")
        sys.exit(1)

    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        print(f"Requested GPU {args.gpu} does not exist.")
        sys.exit(2)

    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        marker = "  <-- selected" if i == args.gpu else ""
        print(f"GPU {i}             : {p.name}{marker}")
        print(f"GPU {i} VRAM        : {p.total_memory / 1024**3:.2f} GB")
        print(f"GPU {i} capability  : {p.major}.{p.minor}")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)
    x = torch.randn(2048, 2048, device=device)
    y = x @ x
    torch.cuda.synchronize(device)
    print(f"CUDA tensor test  : PASS ({y.device})")


if __name__ == "__main__":
    main()
