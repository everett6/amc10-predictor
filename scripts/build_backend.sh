#!/usr/bin/env bash
# Freeze the Python prediction engine into dist-backend/amc10-backend/ so the
# desktop app can ship it without requiring Python on the user's machine.
# Build on the platform you are packaging for (PyInstaller does not cross-compile).
#
# The build runs in its own virtualenv (build/venv) holding only the runtime
# dependencies. Freezing from a general-purpose environment drags in whatever
# else is installed there (a first attempt came out at 5.4 GB, mostly PyTorch).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x build/venv/bin/python ]; then
  python3 -m venv build/venv
  build/venv/bin/pip install --quiet numpy scipy pandas scikit-learn fastapi uvicorn pyinstaller
fi
build/venv/bin/python -m PyInstaller --noconfirm --clean --onedir --name amc10-backend \
  --distpath dist-backend --workpath build/pyinstaller --specpath build/spec \
  --paths src \
  --add-data "$(pwd)/src/storage/schema.sql:storage" \
  --collect-submodules uvicorn \
  --hidden-import sklearn.ensemble \
  --exclude-module matplotlib --exclude-module tkinter --exclude-module pytest --exclude-module IPython \
  src/api/app.py
echo "Built dist-backend/amc10-backend"
