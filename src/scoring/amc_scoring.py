"""AMC 10 scoring rules.

Since 2001 (which covers our entire 2018-2025 target window), AMC 10/12
scoring has been:
    +6 points for each correct answer
    +1.5 points for each blank (unanswered) question
    +0 points for each incorrect answer
across 25 multiple-choice questions, giving a maximum of 150 and a
"all blank" floor of 37.5.

This has been true continuously since 2001 (it replaced the older
30-point-correct/0-blank/varied-wrong scale used through 1999, and the
brief -0-for-wrong-with-2.5-per-blank AMC scale used in 2000), so no
year-dependent branching is needed for our 2018-2025 target contests.
"""
from __future__ import annotations

from dataclasses import dataclass

NUM_QUESTIONS = 25
POINTS_CORRECT = 6.0
POINTS_BLANK = 1.5
POINTS_WRONG = 0.0
MAX_SCORE = NUM_QUESTIONS * POINTS_CORRECT  # 150.0
MIN_SCORE = 0.0
ALL_BLANK_SCORE = NUM_QUESTIONS * POINTS_BLANK  # 37.5


@dataclass(frozen=True)
class ScoreBreakdown:
    correct: int
    wrong: int
    blank: int
    score: float

    def __post_init__(self) -> None:
        total = self.correct + self.wrong + self.blank
        if total != NUM_QUESTIONS:
            raise ValueError(
                f"correct+wrong+blank must equal {NUM_QUESTIONS}, got {total}"
            )
        if min(self.correct, self.wrong, self.blank) < 0:
            raise ValueError("correct/wrong/blank counts must be non-negative")


def score_from_counts(correct: int, wrong: int, blank: int) -> ScoreBreakdown:
    """Compute the official AMC 10 score from raw counts."""
    score = correct * POINTS_CORRECT + blank * POINTS_BLANK + wrong * POINTS_WRONG
    return ScoreBreakdown(correct=correct, wrong=wrong, blank=blank, score=score)


def score_from_responses(responses: list[str | None], answer_key: list[str]) -> ScoreBreakdown:
    """Compute score from a per-problem response vector.

    responses[i] is the student's chosen answer letter ('a'..'e', case
    insensitive) for problem i, or None/'' for a blank.
    answer_key[i] is the correct answer letter for problem i.
    Both lists must have length NUM_QUESTIONS and be aligned by problem
    position (1-indexed problem i is responses[i-1]).
    """
    if len(responses) != NUM_QUESTIONS or len(answer_key) != NUM_QUESTIONS:
        raise ValueError(
            f"responses and answer_key must each have length {NUM_QUESTIONS}"
        )
    correct = wrong = blank = 0
    for resp, key in zip(responses, answer_key):
        if resp is None or str(resp).strip() == "":
            blank += 1
        elif str(resp).strip().lower() == str(key).strip().lower():
            correct += 1
        else:
            wrong += 1
    return score_from_counts(correct, wrong, blank)


def is_valid_score(score: float) -> bool:
    """A score is achievable iff it's of the form 6c + 1.5b for integers
    c, b >= 0 with c + b <= 25 (equivalently 2*score is an integer
    multiple of 3 in the achievable range)."""
    if score < MIN_SCORE or score > MAX_SCORE:
        return False
    doubled = round(score * 2)
    if abs(doubled - score * 2) > 1e-6:
        return False
    if doubled % 3 != 0:
        return False
    # doubled = 12c + 3b -> divide by 3: quotient = 4c + b
    quotient = doubled // 3
    # Find any non-negative integer solution c,b with 4c+b=quotient, c+b<=25
    for c in range(NUM_QUESTIONS + 1):
        b = quotient - 4 * c
        if b < 0:
            break
        if c + b <= NUM_QUESTIONS:
            return True
    return False
