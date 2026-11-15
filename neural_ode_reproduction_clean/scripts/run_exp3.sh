#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python run_exp3.py --mode paper --require-cuda "$@"
