"""SQLite connection + schema management, and typed read/write helpers
used by both ingestion and analytics/modeling code."""
from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path) -> sqlite3.Connection:
    conn = connect(db_path)
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def upsert_contest(
    conn: sqlite3.Connection,
    contest_id: str,
    contest: str,
    year: int,
    label: str,
    num_problems: int,
    source_url: str,
) -> None:
    conn.execute(
        """
        INSERT INTO contests (contest_id, contest, year, label, num_problems, source_url, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(contest_id) DO UPDATE SET
            num_problems=excluded.num_problems,
            source_url=excluded.source_url,
            fetched_at=excluded.fetched_at
        """,
        (contest_id, contest, year, label, num_problems, source_url, utcnow_iso()),
    )


def upsert_problem(
    conn: sqlite3.Connection,
    problem_id: str,
    contest_id: str,
    position: int,
    question: str,
    choices: dict[str, str],
    answer: str,
    seed_elo: float,
    solve_time_seconds: float,
    solution_text: str,
    amc10_problem_number: int,
    amc12_problem_number: int,
    amc8_problem_number: int,
    aime_problem_number: int,
    concepts_with_categories: list[tuple[str, str]],
) -> None:
    conn.execute(
        """
        INSERT INTO problems (
            problem_id, contest_id, position, question,
            choice_a, choice_b, choice_c, choice_d, choice_e,
            answer, seed_elo, solve_time_seconds, solution_text,
            amc10_problem_number, amc12_problem_number, amc8_problem_number,
            aime_problem_number
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(problem_id) DO UPDATE SET
            question=excluded.question,
            choice_a=excluded.choice_a, choice_b=excluded.choice_b,
            choice_c=excluded.choice_c, choice_d=excluded.choice_d,
            choice_e=excluded.choice_e, answer=excluded.answer,
            seed_elo=excluded.seed_elo, solve_time_seconds=excluded.solve_time_seconds,
            solution_text=excluded.solution_text
        """,
        (
            problem_id, contest_id, position, question,
            choices["a"], choices["b"], choices["c"], choices["d"], choices["e"],
            answer, seed_elo, solve_time_seconds, solution_text,
            amc10_problem_number, amc12_problem_number, amc8_problem_number,
            aime_problem_number,
        ),
    )
    conn.execute("DELETE FROM problem_concepts WHERE problem_id = ?", (problem_id,))
    for concept, category in concepts_with_categories:
        conn.execute(
            "INSERT OR IGNORE INTO problem_concepts (problem_id, concept, category) VALUES (?, ?, ?)",
            (problem_id, concept, category),
        )


def save_user_responses(
    conn: sqlite3.Connection, user_id: str, contest_id: str, responses: list[str | None]
) -> None:
    now = utcnow_iso()
    for position, resp in enumerate(responses, start=1):
        conn.execute(
            """
            INSERT INTO user_responses (user_id, contest_id, position, response, recorded_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, contest_id, position) DO UPDATE SET
                response=excluded.response, recorded_at=excluded.recorded_at
            """,
            (user_id, contest_id, position, resp, now),
        )


def save_prediction(conn: sqlite3.Connection, prediction_id: str, user_id: str, input_contests: list[str], result: dict) -> None:
    conn.execute(
        """
        INSERT INTO predictions (prediction_id, user_id, created_at, input_contests, result_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(prediction_id) DO UPDATE SET result_json=excluded.result_json
        """,
        (prediction_id, user_id, utcnow_iso(), json.dumps(input_contests), json.dumps(result)),
    )
