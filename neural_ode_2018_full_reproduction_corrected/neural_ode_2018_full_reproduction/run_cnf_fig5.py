#!/usr/bin/env python3
"""Thin entry point:  python run_cnf_fig5.py --mode {smoke,paper}  [experiment options]"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.argv[0] = str(ROOT / "02_cnf/figure5_maximum_likelihood.py")
runpy.run_path(sys.argv[0], run_name="__main__")
