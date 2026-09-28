#!/usr/bin/env python3
"""Run the calibration/backtesting harness (rolling-window + leave-one-
test-out) against synthetic students and print MAE/RMSE/calibration
error/coverage. See src/analytics/calibration.py for methodology notes
-- this validates the *pipeline*, not real-world accuracy on real
students (no real longitudinal response logs were available)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from analytics.calibration import run_backtest, summarize_backtest  # noqa: E402
from analytics.data_access import load_problems  # noqa: E402
from storage import schema  # noqa: E402


def main() -> int:
    conn = schema.connect(REPO_ROOT / "data" / "amc10.sqlite3")
    problems = load_problems(conn)
    contest_ids = sorted(
        problems["contest_id"].unique(),
        key=lambda cid: tuple(problems[problems.contest_id == cid]["recency_rank"].iloc[0]),
    )

    n_students = 25
    n_sims = 3000
    results = {}
    for mode in ("rolling", "leave_one_out"):
        print(f"Running {mode} backtest with {n_students} synthetic students...")
        records = run_backtest(problems, contest_ids, n_synthetic_students=n_students, mode=mode, n_sims=n_sims, seed=42)
        summary = summarize_backtest(records)
        results[mode] = summary
        print(json.dumps(summary, indent=2))
        print()

    out_path = REPO_ROOT / "data" / "backtest_report.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Full report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
