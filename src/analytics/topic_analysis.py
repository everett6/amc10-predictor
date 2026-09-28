"""Descriptive analytics over a user's response frame: topic accuracy,
difficulty-bucket accuracy, position accuracy, and topic x difficulty.

All functions take the flat DataFrame produced by
analytics.data_access.build_response_frame (columns: contest_id,
position, problem_id, seed_elo, categories, response, answer, outcome).
"""
from __future__ import annotations

import pandas as pd

DIFFICULTY_BUCKET_WIDTH = 200  # Elo-scale points per bucket
DIFFICULTY_BUCKET_EDGES = list(range(200, 2401, DIFFICULTY_BUCKET_WIDTH))  # 200..2400


def _outcome_counts(df: pd.DataFrame) -> dict:
    n = len(df)
    correct = int((df["outcome"] == "correct").sum())
    wrong = int((df["outcome"] == "wrong").sum())
    blank = int((df["outcome"] == "blank").sum())
    attempted = correct + wrong
    return {
        "n": n,
        "correct": correct,
        "wrong": wrong,
        "blank": blank,
        "attempted": attempted,
        # accuracy among attempted (excludes blanks) -- the natural
        # "skill" measure; accuracy_overall includes blanks as misses.
        "accuracy_attempted": (correct / attempted) if attempted else None,
        "accuracy_overall": (correct / n) if n else None,
    }


def topic_accuracy(response_df: pd.DataFrame) -> pd.DataFrame:
    """One row per broad category the user has seen at least one problem
    in, with correct/wrong/blank counts and accuracy."""
    exploded = response_df.explode("categories")
    rows = []
    for category, group in exploded.groupby("categories"):
        stats = _outcome_counts(group)
        rows.append({"category": category, **stats})
    return pd.DataFrame(rows).sort_values("accuracy_attempted", ascending=False, na_position="last").reset_index(drop=True)


def difficulty_bucket_label(seed_elo: float) -> str:
    lo = (int(seed_elo) // DIFFICULTY_BUCKET_WIDTH) * DIFFICULTY_BUCKET_WIDTH
    return f"{lo}-{lo + DIFFICULTY_BUCKET_WIDTH}"


def difficulty_bucket_accuracy(response_df: pd.DataFrame) -> pd.DataFrame:
    df = response_df.copy()
    df["bucket"] = df["seed_elo"].apply(difficulty_bucket_label)
    rows = []
    for bucket, group in df.groupby("bucket"):
        stats = _outcome_counts(group)
        lo = int(bucket.split("-")[0])
        rows.append({"bucket": bucket, "bucket_lo": lo, **stats})
    return pd.DataFrame(rows).sort_values("bucket_lo").drop(columns="bucket_lo").reset_index(drop=True)


def position_accuracy(response_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for position, group in response_df.groupby("position"):
        stats = _outcome_counts(group)
        rows.append({"position": int(position), **stats})
    return pd.DataFrame(rows).sort_values("position").reset_index(drop=True)


def topic_by_difficulty(response_df: pd.DataFrame) -> pd.DataFrame:
    """Category x difficulty-bucket crosstab of accuracy_attempted (NaN
    where the user has no attempts in that cell)."""
    df = response_df.copy()
    df["bucket"] = df["seed_elo"].apply(difficulty_bucket_label)
    exploded = df.explode("categories")
    rows = []
    for (category, bucket), group in exploded.groupby(["categories", "bucket"]):
        stats = _outcome_counts(group)
        rows.append({"category": category, "bucket": bucket, **stats})
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df
    pivot = long_df.pivot(index="category", columns="bucket", values="accuracy_attempted")
    return pivot
