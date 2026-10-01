"""AMC 10 scoring rules.

Current rules (2007 onward), which are what a 2026 prediction uses:
    +6 per correct answer, +1.5 per blank, +0 per wrong answer,
25 questions, maximum 150, all-blank score 37.5.

The blank value has changed over the contest's history. Per Wikipedia's
"American Mathematics Competitions" article (checked 2026-09-30): the
AMC format of 25 questions at 6 points began in 2000 with 2 points per
blank; the blank value was raised to 2.5 in 2001 and cut to 1.5 starting
with the 2007 contests. (The AoPS wiki dates the 2 -> 2.5 change to 2002
instead; the two sources disagree about 2001 only. We follow Wikipedia.)

`blank_points_for_year` gives the value in force for a given contest
year, so a score on an old paper can be shown both as it was officially
scored then and under today's rules.
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


def blank_points_for_year(year: int) -> float:
    """Points per blank answer under the rules in force that year."""
    if year >= 2007:
        return 1.5
    if year >= 2001:
        return 2.5
    return 2.0


def score_from_counts(correct: int, wrong: int, blank: int, blank_points: float = POINTS_BLANK) -> ScoreBreakdown:
    """Compute the AMC 10 score from raw counts (current rules by default)."""
    score = correct * POINTS_CORRECT + blank * blank_points + wrong * POINTS_WRONG
    return ScoreBreakdown(correct=correct, wrong=wrong, blank=blank, score=score)


def score_from_responses(
    responses: list[str | None], answer_key: list[str], blank_points: float = POINTS_BLANK
) -> ScoreBreakdown:
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
    return score_from_counts(correct, wrong, blank, blank_points)


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
