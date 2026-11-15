#!/usr/bin/env python3
"""Easy entry point for Experiment 1 (MNIST)."""
import runpy
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.argv[0] = str(ROOT / "01_mnist" / "experiment.py")
runpy.run_path(sys.argv[0], run_name="__main__")
