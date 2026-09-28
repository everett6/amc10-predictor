#!/usr/bin/env python3
"""Validate the ingested AMC 10 dataset for completeness and consistency.

Checks:
  * All 18 target contests are present, each with exactly 25 problems.
  * Every problem has a non-empty question, 5 non-empty choices, a valid
    answer letter, a positive seed_elo and solve_time_seconds.
  * Every problem has at least one concept tag, and every concept tag
    maps to one of the 15 broad categories (flags any that don't, so
    src/ingestion/concepts.py can be extended).
  * No duplicate (contest_id, position) rows.

Exits non-zero and prints every error found if any check fails; this is
meant to run in CI after ingestion, not just once by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ingestion.concepts import UNCATEGORIZED  # noqa: E402
from ingestion.pipeline import TARGET_CONTESTS  # noqa: E402
from storage import schema  # noqa: E402


def validate(db_path: Path) -> list[str]:
    errors: list[str] = []
    conn = schema.connect(db_path)

    expected_ids = {f"{y}{l}" for y, l in TARGET_CONTESTS}
    present_ids = {row["contest_id"] for row in conn.execute("SELECT contest_id FROM contests")}

    missing = expected_ids - present_ids
    for cid in sorted(missing):
        errors.append(f"missing contest: {cid}")

    for row in conn.execute(
        "SELECT contest_id, num_problems FROM contests WHERE contest_id IN ({})".format(
            ",".join("?" * len(expected_ids))
        ),
        list(expected_ids),
    ):
        if row["num_problems"] != 25:
            errors.append(f"{row['contest_id']}: expected 25 problems, contest row says {row['num_problems']}")

    for cid in sorted(present_ids & expected_ids):
        rows = conn.execute(
            "SELECT position FROM problems WHERE contest_id = ? ORDER BY position", (cid,)
        ).fetchall()
        positions = [r["position"] for r in rows]
        if positions != list(range(1, 26)):
            errors.append(f"{cid}: problem positions are {positions}, expected 1..25 with no gaps/dupes")

    for row in conn.execute(
        """
        SELECT problem_id, contest_id, position, question, choice_a, choice_b,
               choice_c, choice_d, choice_e, answer, seed_elo, solve_time_seconds
        FROM problems
        """
    ):
        pid = row["problem_id"]
        if not row["question"].strip():
            errors.append(f"{pid}: empty question text")
        for letter in "abcde":
            if not row[f"choice_{letter}"].strip():
                errors.append(f"{pid}: empty choice {letter}")
        if row["answer"] not in "abcde":
            errors.append(f"{pid}: invalid answer letter {row['answer']!r}")
        if row["seed_elo"] <= 0:
            errors.append(f"{pid}: non-positive seed_elo {row['seed_elo']}")
        if row["solve_time_seconds"] <= 0:
            errors.append(f"{pid}: non-positive solve_time_seconds {row['solve_time_seconds']}")

        concept_rows = conn.execute(
            "SELECT concept, category FROM problem_concepts WHERE problem_id = ?", (pid,)
        ).fetchall()
        if not concept_rows:
            errors.append(f"{pid}: has no concept tags")
        for crow in concept_rows:
            if crow["category"] == UNCATEGORIZED:
                errors.append(f"{pid}: concept {crow['concept']!r} has no broad-category mapping")

    conn.close()
    return errors


def main() -> int:
    db_path = REPO_ROOT / "data" / "amc10.sqlite3"
    if not db_path.exists():
        print(f"ERROR: no database at {db_path}. Run scripts/ingest_amc.py first.")
        return 2

    errors = validate(db_path)
    if not errors:
        print("Validation passed: all 18 target contests present with 25 valid, fully-tagged problems each.")
        return 0

    print(f"Validation found {len(errors)} error(s):\n")
    for e in errors:
        print(f"  - {e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
