#!/usr/bin/env python3
"""Thin entry point:  python run_cnf_fig4.py --mode {smoke,paper}  [experiment options]"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.argv[0] = str(ROOT / "02_cnf/figure4_density_matching.py")
runpy.run_path(sys.argv[0], run_name="__main__")
