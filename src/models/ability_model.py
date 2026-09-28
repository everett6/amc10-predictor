"""IRT-inspired hierarchical ability model with Bayesian smoothing, and
recency-weighted combination of multiple past-contest attempts.

Model
-----
LIVE's `seedElo` difficulty ratings are already on a chess-like Elo
scale, so we reuse the standard Elo win-probability form as our item
response function (a 1-parameter logistic / Rasch model reparametrized
on the 400-point-per-decade-of-odds scale instead of natural log-odds):

    P(correct | ability=theta, difficulty=b) = 1 / (1 + 10^((b - theta) / 400))

This is IRT-inspired rather than a full 2PL/3PL fit: we don't estimate a
per-item discrimination or guessing parameter from data (25 items x <=18
contests is far too little data to fit those reliably per item), and we
lean on LIVE's own large-sample difficulty calibration for `b` instead of
re-estimating it. What we *do* estimate per user is ability, and we
estimate it hierarchically:

    theta_global ~ found by MLE over ALL of the user's attempted problems
    theta_category = shrink(theta_category_mle, theta_global, n_category)

using precision-weighted (empirical-Bayes-style) shrinkage: a category
with few attempts pulls strongly toward theta_global; a category with
many attempts is dominated by its own MLE. This is the "Bayesian
smoothing" -- equivalent to a Normal(theta_global, tau^2) prior on each
theta_category with a fixed effective prior sample size.

Multi-test combination with recency weighting
----------------------------------------------
When a user supplies responses to several past contests, each attempt's
contribution to the ability MLE is weighted by a recency weight (more
recent contests count more), via `recency_half_life` contests of
exponential decay. This lets the model track genuine
improvement/decline over time rather than treating a contest from 2018
and one from 2025 as equally informative about the user's *current*
ability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ELO_SCALE = 400.0
# Effective prior sample size (in attempted-problem units) pulling a
# category's estimate toward the global ability. Larger = more smoothing.
DEFAULT_SHRINKAGE_K = 6.0
# Recency half-life in *contests* (not years): weight halves every this
# many contests further back in the user's ordered attempt history.
DEFAULT_RECENCY_HALF_LIFE = 4.0


def p_correct(theta: float, difficulty: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((difficulty - theta) / ELO_SCALE))


def _log_likelihood_and_grad(theta: float, difficulties: np.ndarray, corrects: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    p = 1.0 / (1.0 + 10.0 ** ((difficulties - theta) / ELO_SCALE))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    ll = np.sum(weights * (corrects * np.log(p) + (1 - corrects) * np.log(1 - p)))
    ln10_over_scale = math.log(10.0) / ELO_SCALE
    dp_dtheta = p * (1 - p) * ln10_over_scale
    grad = np.sum(weights * (corrects / p - (1 - corrects) / (1 - p)) * dp_dtheta)
    return ll, grad


def mle_ability(difficulties: np.ndarray, corrects: np.ndarray, weights: np.ndarray | None = None) -> float:
    """MLE of a single scalar ability theta given attempted items'
    difficulties and binary correctness (weighted log-likelihood).

    Falls back to a wide-but-finite estimate at the data's edges (all
    correct / all wrong) instead of +/-inf, since a real student's true
    ability is virtually never literally infinite.
    """
    if len(difficulties) == 0:
        return 0.0
    weights = np.ones(len(difficulties)) if weights is None else np.asarray(weights, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    difficulties = np.asarray(difficulties, dtype=float)

    if np.all(corrects == 1):
        return float(np.max(difficulties) + 800.0)
    if np.all(corrects == 0):
        return float(np.min(difficulties) - 800.0)

    def grad_fn(theta: float) -> float:
        return _log_likelihood_and_grad(theta, difficulties, corrects, weights)[1]

    lo, hi = -3000.0, 3000.0
    try:
        return float(brentq(grad_fn, lo, hi, xtol=1e-3))
    except ValueError:
        # Gradient doesn't change sign in range (degenerate weights etc.)
        # -- return the weighted-average difficulty as a safe fallback.
        return float(np.average(difficulties, weights=weights))


@dataclass
class CategoryAbility:
    category: str
    theta_mle: float | None      # None if zero attempts in this category
    theta_smoothed: float
    n_attempted: int
    n_correct: int


@dataclass
class AbilityProfile:
    theta_global: float
    categories: dict[str, CategoryAbility] = field(default_factory=dict)
    n_total_attempted: int = 0
    n_total_correct: int = 0

    def ability_for(self, category: str) -> float:
        cat = self.categories.get(category)
        return cat.theta_smoothed if cat is not None else self.theta_global


def recency_weights(recency_ranks: pd.Series, half_life_contests: float = DEFAULT_RECENCY_HALF_LIFE) -> pd.Series:
    """Exponential recency weighting: the most recent contest in the
    provided data gets weight 1.0; each contest further back (by
    distinct recency rank, i.e. contest order, not raw calendar gap)
    is discounted by 0.5 per half_life_contests steps back."""
    distinct_ranks = sorted(recency_ranks.unique())
    rank_to_index = {r: i for i, r in enumerate(distinct_ranks)}  # 0 = oldest
    max_index = len(distinct_ranks) - 1
    decay = math.log(2.0) / half_life_contests

    def weight_for(rank):
        steps_back = max_index - rank_to_index[rank]
        return math.exp(-decay * steps_back)

    return recency_ranks.map(weight_for)


def fit_ability_profile(
    response_df: pd.DataFrame,
    shrinkage_k: float = DEFAULT_SHRINKAGE_K,
    recency_half_life: float = DEFAULT_RECENCY_HALF_LIFE,
    exclude_blanks: bool = True,
) -> AbilityProfile:
    """Fit the hierarchical ability model from a user's response frame
    (see analytics.data_access.build_response_frame).

    exclude_blanks=True (default) treats blanks as missing data for
    ability estimation (a blank tells us about the user's time
    management / risk tolerance, not necessarily their knowledge of
    that topic) -- consistent with standard IRT practice of not
    penalizing omitted items in ability estimation.
    """
    df = response_df.copy()
    if exclude_blanks:
        df = df[df["outcome"] != "blank"]
    df["is_correct"] = (df["outcome"] == "correct").astype(float)
    df["weight"] = recency_weights(df["recency_rank"], recency_half_life)

    theta_global = mle_ability(df["seed_elo"].to_numpy(), df["is_correct"].to_numpy(), df["weight"].to_numpy())

    exploded = df.explode("categories")
    categories: dict[str, CategoryAbility] = {}
    for category, group in exploded.groupby("categories"):
        if not category:
            continue
        n = len(group)
        n_correct = int(group["is_correct"].sum())
        theta_cat_mle = mle_ability(group["seed_elo"].to_numpy(), group["is_correct"].to_numpy(), group["weight"].to_numpy())
        # Precision-weighted shrinkage toward the global estimate: with
        # `n` attempts and prior pseudo-count `shrinkage_k`, the smoothed
        # estimate is a weighted average favoring theta_global when n is
        # small and theta_cat_mle when n is large.
        effective_n = group["weight"].sum()
        theta_smoothed = (effective_n * theta_cat_mle + shrinkage_k * theta_global) / (effective_n + shrinkage_k)
        categories[category] = CategoryAbility(
            category=category,
            theta_mle=theta_cat_mle,
            theta_smoothed=theta_smoothed,
            n_attempted=n,
            n_correct=n_correct,
        )

    return AbilityProfile(
        theta_global=theta_global,
        categories=categories,
        n_total_attempted=len(df),
        n_total_correct=int(df["is_correct"].sum()),
    )
