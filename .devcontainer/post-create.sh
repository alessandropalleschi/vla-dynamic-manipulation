#!/usr/bin/env bash
# Runs once after the dev container is created.
set -euo pipefail
python -m pip install --upgrade pip
# CPU torch keeps the image small; for CUDA use e.g.  pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pytest
echo "Ready. Try:  make demo"
