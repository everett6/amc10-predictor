#!/usr/bin/env bash
# Freeze the Python prediction engine into dist-backend/amc10-backend/.
# Thin wrapper; the logic lives in build_backend.py so Windows CI can share it.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/build_backend.py "$@"
