"""Start the frozen engine, wait for /api/health, then stop it.

Used by CI to check that the PyInstaller build actually boots on each OS
before it is wrapped into an installer. Exits non-zero on failure.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    exe = ROOT / "dist-backend" / "amc10-backend" / (
        "amc10-backend.exe" if os.name == "nt" else "amc10-backend"
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    tmp = Path(tempfile.mkdtemp())
    env = {
        **os.environ,
        "AMC10_USER_DATA_DIR": str(tmp),
        "AMC10_DB_PATH": str(tmp / "smoke.sqlite3"),
        "AMC10_SEED_DIR": str(ROOT / "data" / "raw" / "live"),
        "AMC10_FRONTEND_DIST": str(ROOT / "frontend" / "dist"),
    }
    proc = subprocess.Popen([str(exe), "--port", str(port)], env=env)
    try:
        deadline = time.time() + 180
        while time.time() < deadline:
            if proc.poll() is not None:
                sys.exit(f"engine exited early with code {proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as r:
                    print("health:", json.loads(r.read()))
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
                    assert b"<div id=\"root\"" in r.read(), "UI index.html not served"
                # Exercises seed ingestion, SQLite and pandas inside the frozen build.
                url = f"http://127.0.0.1:{port}/api/contests"
                with urllib.request.urlopen(url, timeout=120) as r:
                    contests = json.loads(r.read())
                assert len(contests) > 0, "no contests loaded from seed data"
                print(f"engine OK: {len(contests)} contests")
                return
            except OSError:
                time.sleep(1)
        sys.exit("engine did not answer /api/health within 180 s")
    finally:
        proc.kill()


if __name__ == "__main__":
    main()
