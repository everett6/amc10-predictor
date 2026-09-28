"""Shared read helpers: load problem metadata (with broad categories) and
join it with a user's historical responses into one flat DataFrame that
every analytics/model module builds on."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd

# Recency rank: label ordering within a year for contests that share a
# year (Fall 2021's C/D come after the Feb A/B of the same year).
_LABEL_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}


@dataclass
class ContestResponses:
    """One student's raw responses to one historical contest.

    responses[i] is the letter chosen ('a'..'e') for problem position i+1,
    or None for a blank. Must have length 25.
    """

    contest_id: str
    responses: list[str | None]

    def __post_init__(self) -> None:
        if len(self.responses) != 25:
            raise ValueError(
                f"{self.contest_id}: expected 25 responses, got {len(self.responses)}"
            )


def load_problems(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per problem, with a `categories` column (list[str])."""
    problems = pd.read_sql_query(
        """
        SELECT p.problem_id, p.contest_id, p.position, p.answer, p.seed_elo,
               p.solve_time_seconds, c.year, c.label
        FROM problems p JOIN contests c ON p.contest_id = c.contest_id
        """,
        conn,
    )
    concept_rows = pd.read_sql_query(
        "SELECT problem_id, concept, category FROM problem_concepts", conn
    )
    cats = concept_rows.groupby("problem_id")["category"].apply(list).rename("categories")
    concepts = concept_rows.groupby("problem_id")["concept"].apply(list).rename("concepts")
    problems = problems.join(cats, on="problem_id").join(concepts, on="problem_id")
    problems["categories"] = problems["categories"].apply(lambda v: v if isinstance(v, list) else [])
    problems["concepts"] = problems["concepts"].apply(lambda v: v if isinstance(v, list) else [])
    problems["recency_rank"] = problems.apply(
        lambda r: (r["year"], _LABEL_ORDER.get(r["label"], 99)), axis=1
    )
    return problems


def build_response_frame(
    problems: pd.DataFrame, contest_responses: list[ContestResponses]
) -> pd.DataFrame:
    """Join problem metadata with a user's responses across one or more
    historical contests into one row-per-attempted-problem DataFrame with
    an `outcome` column in {"correct", "wrong", "blank"}."""
    rows = []
    for cr in contest_responses:
        subset = problems[problems["contest_id"] == cr.contest_id]
        if subset.empty:
            raise ValueError(f"no problem data found for contest {cr.contest_id!r}")
        by_position = subset.set_index("position")
        for position in range(1, 26):
            if position not in by_position.index:
                raise ValueError(f"{cr.contest_id}: missing problem at position {position}")
            prob = by_position.loc[position]
            resp = cr.responses[position - 1]
            if resp is None or str(resp).strip() == "":
                outcome = "blank"
            elif str(resp).strip().lower() == str(prob["answer"]).strip().lower():
                outcome = "correct"
            else:
                outcome = "wrong"
            rows.append(
                {
                    "contest_id": cr.contest_id,
                    "position": position,
                    "problem_id": prob["problem_id"],
                    "seed_elo": prob["seed_elo"],
                    "categories": prob["categories"],
                    "year": prob["year"],
                    "label": prob["label"],
                    "recency_rank": prob["recency_rank"],
                    "response": resp,
                    "answer": prob["answer"],
                    "outcome": outcome,
                }
            )
    return pd.DataFrame(rows)
