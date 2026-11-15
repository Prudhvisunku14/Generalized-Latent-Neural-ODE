#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python run_exp1.py --mode paper --require-cuda "$@"
