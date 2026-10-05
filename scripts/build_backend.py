"""Freeze the Python prediction engine into dist-backend/amc10-backend/.

The desktop app ships this folder so users do not need Python installed.
PyInstaller does not cross-compile: run this on the OS you are packaging for
(Linux, macOS or Windows). Works the same from a shell or from CI.

The build runs in its own virtualenv (build/venv) holding only the runtime
dependencies. Freezing from a general-purpose environment drags in whatever
else is installed there (a first attempt came out at 5.4 GB, mostly PyTorch).
Pass --no-venv to use the current interpreter instead (CI already gives each
job a clean one).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DEPS = ["numpy", "scipy", "pandas", "scikit-learn", "fastapi", "uvicorn", "pyinstaller"]


def build_python(use_venv: bool) -> str:
    if not use_venv:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *RUNTIME_DEPS])
        return sys.executable
    venv_dir = ROOT / "build" / "venv"
    py = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not py.exists():
        venv.create(venv_dir, with_pip=True)
        subprocess.check_call([str(py), "-m", "pip", "install", "--quiet", *RUNTIME_DEPS])
    return str(py)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-venv", action="store_true", help="freeze with the current Python")
    args = parser.parse_args()

    py = build_python(use_venv=not args.no_venv)
    schema = ROOT / "src" / "storage" / "schema.sql"
    subprocess.check_call(
        [
            py, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
            "--name", "amc10-backend",
            "--distpath", str(ROOT / "dist-backend"),
            "--workpath", str(ROOT / "build" / "pyinstaller"),
            "--specpath", str(ROOT / "build" / "spec"),
            "--paths", str(ROOT / "src"),
            "--add-data", f"{schema}{os.pathsep}storage",
            "--collect-submodules", "uvicorn",
            "--hidden-import", "sklearn.ensemble",
            "--exclude-module", "matplotlib",
            "--exclude-module", "tkinter",
            "--exclude-module", "pytest",
            "--exclude-module", "IPython",
            str(ROOT / "src" / "api" / "app.py"),
        ],
        cwd=ROOT,
    )
    print("Built dist-backend/amc10-backend")


if __name__ == "__main__":
    main()
