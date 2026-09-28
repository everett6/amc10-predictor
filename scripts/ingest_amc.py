#!/usr/bin/env python3
"""Ingest AMC 10 contest data into the local SQLite database.

Usage:
    python scripts/ingest_amc.py --mode live
    python scripts/ingest_amc.py --mode seed

--mode live fetches all 18 target contests directly from
live.poshenloh.com (respecting robots.txt and a per-host rate limit;
see src/ingestion/http_client.py). This is the preferred mode whenever
outbound network access to live.poshenloh.com is available.

--mode seed reads previously-saved raw JSON snapshots from
data/raw/live/*.json instead of hitting the network -- a
fallback for environments without live network access. It will report
any target contest with no seed file as failed rather than silently
skipping it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ingestion.pipeline import TARGET_CONTESTS, run_ingestion  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["live", "seed"], default="live")
    parser.add_argument("--db", default=str(REPO_ROOT / "data" / "amc10.sqlite3"))
    args = parser.parse_args()

    print(f"Ingesting {len(TARGET_CONTESTS)} target contests in {args.mode!r} mode...")
    summary = run_ingestion(db_path=Path(args.db), mode=args.mode)

    print(f"\nOK ({len(summary['ok'])}/{len(TARGET_CONTESTS)}): {', '.join(summary['ok'])}")
    if summary["failed"]:
        print(f"\nFAILED ({len(summary['failed'])}):")
        for contest_id, reason in summary["failed"].items():
            print(f"  {contest_id}: {reason}")

    report_path = REPO_ROOT / "data" / "ingestion_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFull report written to {report_path}")

    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
