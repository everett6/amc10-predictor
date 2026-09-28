import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest  # noqa: E402

from storage import schema  # noqa: E402


@pytest.fixture()
def db_path():
    return REPO_ROOT / "data" / "amc10.sqlite3"


@pytest.fixture()
def conn(db_path):
    if not db_path.exists():
        pytest.skip("data/amc10.sqlite3 not present -- run scripts/ingest_amc.py first")
    c = schema.connect(db_path)
    yield c
    c.close()
