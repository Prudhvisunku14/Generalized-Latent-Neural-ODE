#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python run_all_paper.py --mode "${1:-paper}"
