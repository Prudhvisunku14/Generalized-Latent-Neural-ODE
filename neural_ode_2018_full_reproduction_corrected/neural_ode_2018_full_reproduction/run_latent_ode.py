#!/usr/bin/env python3
"""Thin entry point:  python run_latent_ode.py --mode {smoke,paper}  [experiment options]"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.argv[0] = str(ROOT / "03_latent_ode/experiment.py")
runpy.run_path(sys.argv[0], run_name="__main__")
