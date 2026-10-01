#!/usr/bin/env python3
"""Real-data backtest of the contest forecast (no students involved).

For every contest from 2012 on, build the position pools from EARLIER
years only, simulate contests from them, and compare with the contest
that was actually set. The comparison is in score points: for reference
students at fixed abilities who answer everything, how far is the
forecast expected score from the expected score on the real contest,
and does the real contest fall inside the forecast's 90% band?

This is the one part of the system that can be checked against real
history, because it only needs the problems, not anyone's answers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from analytics.data_access import load_problems  # noqa: E402
from simulation.contest_model import build_contest_forecast  # noqa: E402
from storage import schema  # noqa: E402

ABILITIES = [1000, 1200, 1400, 1600, 1800, 2000]
# (label, pool half-life, level model, level half-life)
VARIANTS: list[tuple[str, float | None, str, float | None]] = [
    ("v1: equal, no level", None, "none", None),
    ("pool3, no level", 3.0, "none", None),
    ("pool1, no level", 1.0, "none", None),
    ("pool3 + mean(hl3)", 3.0, "mean", 3.0),
    ("pool3 + mean(hl5)", 3.0, "mean", 5.0),
    ("pool5 + trend(all)", 5.0, "trend", None),
    ("pool5 + trend(hl12)", 5.0, "trend", 12.0),
    ("pool5 + trend(hl8)", 5.0, "trend", 8.0),
    ("pool3 + trend(hl8)", 3.0, "trend", 8.0),
    ("pool8 + trend(hl8)", 8.0, "trend", 8.0),
    ("pool5 + trend(hl5)", 5.0, "trend", 5.0),
    ("all + trend(hl8)", None, "trend", 8.0),
]
FIRST_TEST_YEAR = 2012
N_SIM = 4000


def expected_score(difficulty: np.ndarray, theta: float) -> np.ndarray:
    """6 x expected number correct for a student who answers everything."""
    p = 1.0 / (1.0 + 10.0 ** ((difficulty - theta) / 400.0))
    return 6.0 * p.sum(axis=-1)


def main() -> int:
    conn = schema.connect(REPO_ROOT / "data" / "amc10.sqlite3")
    problems = load_problems(conn)
    rng = np.random.default_rng(7)
    contests = (
        problems[problems["year"] >= FIRST_TEST_YEAR][["contest_id", "year"]]
        .drop_duplicates()
        .sort_values(["year", "contest_id"])
    )

    report: dict[str, dict] = {}
    for label, hl, level, level_hl in VARIANTS:
        errors, covered = [], []
        for _, row in contests.iterrows():
            year = int(row["year"])
            forecast = build_contest_forecast(
                problems, target_year=year, half_life_years=hl, level=level,
                level_half_life_years=level_hl, max_year=year - 1,
            )
            sim_diff, _ = forecast.sample(N_SIM, rng)
            actual = (
                problems[problems["contest_id"] == row["contest_id"]]
                .sort_values("position")["seed_elo"]
                .to_numpy(dtype=float)
            )
            for theta in ABILITIES:
                sim = expected_score(sim_diff, theta)
                act = float(expected_score(actual, theta))
                errors.append(sim.mean() - act)
                lo, hi = np.percentile(sim, [5, 95])
                covered.append(lo <= act <= hi)
        err = np.array(errors)
        report[label] = {
            "n": int(len(err)),
            "bias_points": float(err.mean()),
            "mae_points": float(np.abs(err).mean()),
            "rmse_points": float(np.sqrt((err**2).mean())),
            "coverage_90": float(np.mean(covered)),
        }

    n = next(iter(report.values()))["n"]
    print(f"{'contest forecast':22s} {'bias':>7s} {'MAE':>7s} {'RMSE':>7s} {'cov90':>7s}   (score points, n={n})")
    for key, r in report.items():
        print(f"{key:22s} {r['bias_points']:7.2f} {r['mae_points']:7.2f} {r['rmse_points']:7.2f} {r['coverage_90']:7.2f}")

    out = REPO_ROOT / "data" / "contest_model_backtest.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
