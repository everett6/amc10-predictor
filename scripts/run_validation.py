#!/usr/bin/env python3
"""Old pipeline vs new on simulated students (see src/analytics/validation_v2.py
for what the simulated students are and why this is a stress test, not a
measurement of real-world accuracy)."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
warnings.filterwarnings("ignore")

from analytics.data_access import load_problems  # noqa: E402
from analytics.validation_v2 import run_validation, summarize  # noqa: E402
from storage import schema  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--students-per-target", type=int, default=8)
    parser.add_argument("--n-sims", type=int, default=3000)
    parser.add_argument("--first-year", type=int, default=2019)
    args = parser.parse_args()

    problems = load_problems(schema.connect(REPO_ROOT / "data" / "amc10.sqlite3"))
    df = run_validation(problems, students_per_target=args.students_per_target,
                        n_sims=args.n_sims, first_year=args.first_year)
    summary = summarize(df)
    print(summary.round(2).to_string(index=False))
    out = REPO_ROOT / "data" / "validation_v2_report.json"
    out.write_text(json.dumps({
        "note": "Simulated students from a process different from every model tested. "
                "Not a measurement of accuracy on real students.",
        "settings": vars(args),
        "summary": summary.round(4).to_dict(orient="records"),
    }, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
