#!/usr/bin/env python3
"""Easy entry point for Experiment 3 (RNN vs Latent ODE)."""
import runpy
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.argv[0] = str(ROOT / "03_latent_ode" / "experiment.py")
runpy.run_path(sys.argv[0], run_name="__main__")
