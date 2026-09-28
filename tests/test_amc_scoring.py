import pytest

from scoring.amc_scoring import (
    ALL_BLANK_SCORE,
    MAX_SCORE,
    NUM_QUESTIONS,
    ScoreBreakdown,
    is_valid_score,
    score_from_counts,
    score_from_responses,
)


def test_all_correct():
    b = score_from_counts(25, 0, 0)
    assert b.score == MAX_SCORE == 150.0


def test_all_blank():
    b = score_from_counts(0, 0, 25)
    assert b.score == ALL_BLANK_SCORE == 37.5


def test_all_wrong():
    b = score_from_counts(0, 25, 0)
    assert b.score == 0.0


def test_mixed():
    b = score_from_counts(15, 5, 5)
    assert b.score == 15 * 6 + 5 * 1.5


def test_invalid_counts_raise():
    with pytest.raises(ValueError):
        score_from_counts(20, 20, 20)


def test_negative_counts_raise():
    with pytest.raises(ValueError):
        score_from_counts(-1, 13, 13)


def test_score_from_responses_matches_counts():
    answer_key = ["a"] * NUM_QUESTIONS
    responses = ["a"] * 10 + ["b"] * 10 + [None] * 5
    b = score_from_responses(responses, answer_key)
    assert b.correct == 10
    assert b.wrong == 10
    assert b.blank == 5
    assert b.score == 10 * 6 + 5 * 1.5


def test_score_from_responses_case_insensitive():
    answer_key = ["C"] * NUM_QUESTIONS
    responses = ["c"] * NUM_QUESTIONS
    b = score_from_responses(responses, answer_key)
    assert b.correct == NUM_QUESTIONS


def test_score_from_responses_wrong_length_raises():
    with pytest.raises(ValueError):
        score_from_responses(["a"] * 24, ["a"] * 25)


@pytest.mark.parametrize("score", [150.0, 0.0, 37.5, 97.5, 90.0, 91.5])
def test_valid_scores(score):
    assert is_valid_score(score)


@pytest.mark.parametrize("score", [150.1, -1, 100.0, 91.0, 149.9])
def test_invalid_scores(score):
    assert not is_valid_score(score)


def test_low_score_with_some_blanks_is_valid():
    # 0 correct, 2 blank, 23 wrong -> 3.0 points, a valid (if bad) score.
    assert is_valid_score(3.0)
