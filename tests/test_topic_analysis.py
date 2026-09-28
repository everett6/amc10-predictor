import pandas as pd

from analytics.topic_analysis import (
    difficulty_bucket_accuracy,
    difficulty_bucket_label,
    position_accuracy,
    topic_accuracy,
    topic_by_difficulty,
)


def _sample_df():
    return pd.DataFrame(
        [
            {"contest_id": "2021A", "position": 1, "seed_elo": 500, "categories": ["Arithmetic & Number Sense"], "outcome": "correct"},
            {"contest_id": "2021A", "position": 2, "seed_elo": 600, "categories": ["Arithmetic & Number Sense"], "outcome": "wrong"},
            {"contest_id": "2021A", "position": 3, "seed_elo": 1500, "categories": ["Geometry: Circles & Curves"], "outcome": "blank"},
            {"contest_id": "2021A", "position": 4, "seed_elo": 1600, "categories": ["Geometry: Circles & Curves", "Algebra: Equations & Polynomials"], "outcome": "correct"},
        ]
    )


def test_difficulty_bucket_label():
    assert difficulty_bucket_label(450) == "400-600"
    assert difficulty_bucket_label(600) == "600-800"
    assert difficulty_bucket_label(1999) == "1800-2000"


def test_topic_accuracy_basic():
    df = _sample_df()
    result = topic_accuracy(df).set_index("category")
    assert result.loc["Arithmetic & Number Sense", "correct"] == 1
    assert result.loc["Arithmetic & Number Sense", "wrong"] == 1
    assert result.loc["Arithmetic & Number Sense", "accuracy_attempted"] == 0.5
    # A multi-category problem contributes to both its categories.
    assert result.loc["Algebra: Equations & Polynomials", "correct"] == 1
    assert result.loc["Geometry: Circles & Curves", "blank"] == 1
    assert result.loc["Geometry: Circles & Curves", "correct"] == 1


def test_difficulty_bucket_accuracy_groups_correctly():
    df = _sample_df()
    result = difficulty_bucket_accuracy(df).set_index("bucket")
    assert result.loc["400-600", "n"] == 1
    assert result.loc["600-800", "n"] == 1
    assert result.loc["1400-1600", "n"] == 1
    assert result.loc["1600-1800", "n"] == 1


def test_position_accuracy_one_row_per_position():
    df = _sample_df()
    result = position_accuracy(df)
    assert list(result["position"]) == [1, 2, 3, 4]
    assert result.loc[result.position == 1, "correct"].iloc[0] == 1


def test_topic_by_difficulty_pivot_shape():
    df = _sample_df()
    pivot = topic_by_difficulty(df)
    assert "Arithmetic & Number Sense" in pivot.index
    assert "Geometry: Circles & Curves" in pivot.index
