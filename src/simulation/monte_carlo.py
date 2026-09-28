"""Monte Carlo simulation of a student's 2026 AMC 10 score.

A "future" AMC 10 doesn't exist yet, so we can't know its exact
problems. Instead, for each of the 25 problem *positions*, we treat the
18 historical target contests as an empirical sample of "what a problem
in that position tends to look like" (difficulty rating + topic mix),
and bootstrap-sample from that empirical distribution to build many
plausible synthetic 2026 contests. For each synthetic problem we apply
the student's ability model to get P(correct), then apply the blank
model to split misses into wrong/blank, then score the simulated
contest with the official AMC 10 rubric. Repeating this thousands of
times gives a full predictive distribution over 2026 scores, not just a
point estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from models.ability_model import AbilityProfile, p_correct
from models.blank_model import BlankModel
from scoring.amc_scoring import POINTS_BLANK, POINTS_CORRECT

DEFAULT_NUM_SIMULATIONS = 20_000


@dataclass
class PositionPool:
    position: int
    seed_elos: np.ndarray
    categories_per_sample: list[list[str]]


def build_position_pools(problems: pd.DataFrame) -> dict[int, PositionPool]:
    """One pool per position (1..25) drawn from ALL ingested historical
    contests (not just the ones the user attempted) -- this is the
    empirical model of "what does a 2026 problem in this slot look
    like", independent of any one student."""
    pools: dict[int, PositionPool] = {}
    for position, group in problems.groupby("position"):
        pools[int(position)] = PositionPool(
            position=int(position),
            seed_elos=group["seed_elo"].to_numpy(),
            categories_per_sample=list(group["categories"]),
        )
    return pools


@dataclass
class SimulationResult:
    scores: np.ndarray                 # shape (n_sims,)
    correct_counts: np.ndarray
    wrong_counts: np.ndarray
    blank_counts: np.ndarray
    per_position_p_correct: np.ndarray  # shape (n_sims, 25), for question-level stats
    per_position_difficulty: np.ndarray  # shape (n_sims, 25)
    per_position_categories: list[list[list[str]]]  # [sim][position] -> categories

    def summary(self) -> dict:
        pct = lambda q: float(np.percentile(self.scores, q))  # noqa: E731
        return {
            "n_simulations": int(len(self.scores)),
            "mean_score": float(np.mean(self.scores)),
            "median_score": pct(50),
            "std_score": float(np.std(self.scores)),
            "percentiles": {
                "p5": pct(5), "p10": pct(10), "p25": pct(25),
                "p50": pct(50), "p75": pct(75), "p90": pct(90), "p95": pct(95),
            },
            "expected_correct": float(np.mean(self.correct_counts)),
            "expected_wrong": float(np.mean(self.wrong_counts)),
            "expected_blank": float(np.mean(self.blank_counts)),
        }

    def histogram(self, bin_width: float = 6.0) -> list[dict]:
        """Score histogram (bin_width defaults to one problem's worth of
        points) for charting the predictive distribution."""
        max_score = 150.0
        edges = np.arange(0, max_score + bin_width, bin_width)
        counts, edges = np.histogram(self.scores, bins=edges)
        total = len(self.scores)
        return [
            {
                "range_low": float(edges[i]),
                "range_high": float(edges[i + 1]),
                "count": int(counts[i]),
                "probability": float(counts[i] / total),
            }
            for i in range(len(counts))
        ]


def _category_ability(categories: list[str], ability_profile: AbilityProfile) -> float:
    if not categories:
        return ability_profile.theta_global
    return float(np.mean([ability_profile.ability_for(c) for c in categories]))


def simulate_known_contest(
    contest_problems: pd.DataFrame,
    ability_profile: AbilityProfile,
    blank_model: BlankModel,
    n_simulations: int = DEFAULT_NUM_SIMULATIONS,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Like run_simulation, but for a *known* contest (its actual 25
    problems, in order) rather than bootstrapped position pools. Used by
    calibration/backtesting to predict a held-out historical contest
    that we know the true problems (and true outcome) for."""
    rng = rng or np.random.default_rng()
    n = n_simulations
    ordered = contest_problems.sort_values("position")
    if len(ordered) != 25:
        raise ValueError(f"expected 25 problems, got {len(ordered)}")

    difficulties = ordered["seed_elo"].to_numpy()
    cats_list = list(ordered["categories"])
    abilities = np.array([_category_ability(c, ability_profile) for c in cats_list])
    p_corr = 1.0 / (1.0 + 10.0 ** ((difficulties - abilities) / 400.0))
    gap = difficulties - abilities
    p_blank = blank_model.p_blank_given_not_correct(gap)

    is_correct = rng.random((n, 25)) < p_corr[None, :]
    is_blank_given_wrong = rng.random((n, 25)) < p_blank[None, :]
    outcome_blank = (~is_correct) & is_blank_given_wrong
    outcome_wrong = (~is_correct) & (~is_blank_given_wrong)

    correct_counts = is_correct.sum(axis=1)
    blank_counts = outcome_blank.sum(axis=1)
    wrong_counts = outcome_wrong.sum(axis=1)
    scores = correct_counts * POINTS_CORRECT + blank_counts * POINTS_BLANK

    return SimulationResult(
        scores=scores.astype(float),
        correct_counts=correct_counts,
        wrong_counts=wrong_counts,
        blank_counts=blank_counts,
        per_position_p_correct=np.tile(p_corr, (n, 1)),
        per_position_difficulty=np.tile(difficulties, (n, 1)),
        per_position_categories=[cats_list for _ in range(n)],
    )


def run_simulation(
    position_pools: dict[int, PositionPool],
    ability_profile: AbilityProfile,
    blank_model: BlankModel,
    n_simulations: int = DEFAULT_NUM_SIMULATIONS,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    rng = rng or np.random.default_rng()
    n = n_simulations
    scores = np.zeros(n)
    correct_counts = np.zeros(n, dtype=int)
    wrong_counts = np.zeros(n, dtype=int)
    blank_counts = np.zeros(n, dtype=int)
    per_position_p = np.zeros((n, 25))
    per_position_diff = np.zeros((n, 25))
    per_position_cats: list[list[list[str]]] = [[[] for _ in range(25)] for _ in range(n)]

    for position in range(1, 26):
        pool = position_pools.get(position)
        if pool is None or len(pool.seed_elos) == 0:
            continue
        idx_col = position - 1
        sample_idx = rng.integers(0, len(pool.seed_elos), size=n)
        difficulties = pool.seed_elos[sample_idx]
        cats_list = [pool.categories_per_sample[i] for i in sample_idx]
        abilities = np.array([_category_ability(c, ability_profile) for c in cats_list])
        p_corr = 1.0 / (1.0 + 10.0 ** ((difficulties - abilities) / 400.0))

        per_position_p[:, idx_col] = p_corr
        per_position_diff[:, idx_col] = difficulties
        for i, c in enumerate(cats_list):
            per_position_cats[i][idx_col] = c

        is_correct = rng.random(n) < p_corr
        gap = difficulties - abilities
        p_blank = blank_model.p_blank_given_not_correct(gap)
        is_blank_given_wrong = rng.random(n) < p_blank

        outcome_correct = is_correct
        outcome_blank = (~is_correct) & is_blank_given_wrong
        outcome_wrong = (~is_correct) & (~is_blank_given_wrong)

        correct_counts += outcome_correct.astype(int)
        blank_counts += outcome_blank.astype(int)
        wrong_counts += outcome_wrong.astype(int)
        scores += outcome_correct * POINTS_CORRECT + outcome_blank * POINTS_BLANK

    return SimulationResult(
        scores=scores,
        correct_counts=correct_counts,
        wrong_counts=wrong_counts,
        blank_counts=blank_counts,
        per_position_p_correct=per_position_p,
        per_position_difficulty=per_position_diff,
        per_position_categories=per_position_cats,
    )
