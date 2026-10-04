"""Monte Carlo engine (v2).

One simulated sitting =
  1. a contest drawn from the contest forecast (simulation/contest_model.py),
  2. one ensemble member picked with probability equal to its stacking
     weight, and one draw of that member's parameters from its posterior
     (or bootstrap) distribution, including a fresh good-day/bad-day shift,
  3. one draw of the answer-or-blank model's parameters,
  4. per problem: answered? then, if answered, correct?
  5. the official score: 6 per correct, 1.5 per blank.

Three sources of uncertainty therefore reach the final distribution:
what the paper will look like, how well the student's ability is known,
and ordinary luck on the day. v1 only had the last one plus independent
per-slot difficulty noise.

All random numbers are drawn once (`make_draws`) and reused, so a
what-if run (one topic made stronger) differs from the baseline only
through the change itself, not through fresh noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.attempt_model import AttemptModel
from models.ensemble import Ensemble
from scoring.amc_scoring import POINTS_BLANK, POINTS_CORRECT
from simulation.contest_model import N_CATEGORIES, ContestForecast, category_weights

POSITIONS = np.arange(1, 26, dtype=float)


@dataclass
class FixedContest:
    """A known paper (for backtests): same interface as ContestForecast."""

    difficulty: np.ndarray       # (25,)
    cat: np.ndarray              # (25, 15)

    @classmethod
    def from_problems(cls, contest_problems) -> "FixedContest":
        ordered = contest_problems.sort_values("position")
        return cls(
            ordered["seed_elo"].to_numpy(dtype=float),
            np.vstack([category_weights(c) for c in ordered["categories"]]),
        )

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        return np.tile(self.difficulty, (n, 1)), np.zeros((n, 25), dtype=int)

    def category_matrix(self, idx: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.cat, (idx.shape[0], 25, N_CATEGORIES))


@dataclass
class Draws:
    difficulty: np.ndarray       # (n,25)
    cat: np.ndarray              # (n,25,15)
    member_of: np.ndarray        # (n,) index into ensemble.members
    member_params: dict          # member index -> params for its rows
    gamma: np.ndarray            # (n,3) attempt-model parameters
    u_answer: np.ndarray         # (n,25) uniforms
    u_correct: np.ndarray        # (n,25) uniforms


def make_draws(
    contest: ContestForecast | FixedContest,
    ensemble: Ensemble,
    attempt_model: AttemptModel,
    n: int,
    rng: np.random.Generator,
) -> Draws:
    difficulty, idx = contest.sample(n, rng)
    member_of = rng.choice(len(ensemble.members), size=n, p=ensemble.weights)
    params = {
        j: ensemble.members[j].draw(int((member_of == j).sum()), rng)
        for j in np.unique(member_of)
    }
    return Draws(
        difficulty=difficulty,
        cat=contest.category_matrix(idx),
        member_of=member_of,
        member_params=params,
        gamma=attempt_model.draw(n, rng),
        u_answer=rng.random((n, 25)),
        u_correct=rng.random((n, 25)),
    )


@dataclass
class SimResult:
    scores: np.ndarray
    correct: np.ndarray
    wrong: np.ndarray
    blank: np.ndarray
    p_answer: np.ndarray         # (n,25)
    p_correct_if_answered: np.ndarray   # (n,25)
    difficulty: np.ndarray       # (n,25)

    def summary(self) -> dict:
        q = lambda v: float(np.percentile(self.scores, v))  # noqa: E731
        return {
            "n_simulations": int(len(self.scores)),
            "mean_score": float(self.scores.mean()),
            "median_score": q(50),
            "std_score": float(self.scores.std()),
            "percentiles": {f"p{v}": q(v) for v in (5, 10, 25, 50, 75, 90, 95)},
            "expected_correct": float(self.correct.mean()),
            "expected_wrong": float(self.wrong.mean()),
            "expected_blank": float(self.blank.mean()),
        }

    def histogram(self, bin_width: float = 6.0) -> list[dict]:
        edges = np.arange(0.0, 150.0 + bin_width, bin_width)
        counts, edges = np.histogram(self.scores, bins=edges)
        return [
            {"range_low": float(edges[i]), "range_high": float(edges[i + 1]),
             "count": int(counts[i]), "probability": float(counts[i] / len(self.scores))}
            for i in range(len(counts))
        ]

    def prob_at_least(self, threshold: float) -> float:
        return float(np.mean(self.scores >= threshold))


def simulate(
    draws: Draws,
    ensemble: Ensemble,
    attempt_model: AttemptModel,
    ability_global: float,
    ability_offsets: np.ndarray,
    category_boost: np.ndarray | None = None,
) -> SimResult:
    """ability_global / ability_offsets (15,) are the point estimates used
    for the answer-or-blank gap. category_boost (15,), in Elo points, makes
    the student that much stronger in those categories (what-if runs)."""
    difficulty = draws.difficulty
    if category_boost is not None:
        # A stronger student on a topic is equivalent to an easier problem on it.
        difficulty = difficulty - draws.cat @ category_boost
    n = difficulty.shape[0]

    p_correct = np.empty((n, 25))
    for j, params in draws.member_params.items():
        rows = draws.member_of == j
        p_correct[rows] = ensemble.members[j].predict_with(params, difficulty[rows], draws.cat[rows], POSITIONS)

    ability_item = ability_global + draws.cat @ ability_offsets
    p_answer = attempt_model.predict_with(draws.gamma, difficulty - ability_item, POSITIONS)

    answered = draws.u_answer < p_answer
    is_correct = answered & (draws.u_correct < p_correct)
    correct = is_correct.sum(axis=1)
    blank = (~answered).sum(axis=1)
    wrong = 25 - correct - blank
    scores = correct * POINTS_CORRECT + blank * POINTS_BLANK
    return SimResult(scores.astype(float), correct, wrong, blank, p_answer, p_correct, draws.difficulty)
