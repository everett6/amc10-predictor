"""Models a student's tendency to leave a question blank vs. guess when
they don't actually know the answer, as a function of how far over their
head the problem is (difficulty - ability).

This matters for simulation: two students with identical ability
profiles can get very different scores if one guesses on hard problems
(expected value under random guessing on 5 choices is 6/5 = 1.2 points,
worse than the 1.5-point blank bonus, so *rational* guessing should be
rare) while the other leaves them blank. We fit this per-student from
their own historical blank/wrong pattern rather than assuming everyone
behaves identically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Fallback slope/intercept for a logistic P(blank | not correct) model,
# used when a student has too little non-correct history to fit their
# own curve (e.g. a near-perfect scorer, or a single contest with very
# few misses). Calibrated to a mild, generic pattern: roughly a coin
# flip between blank/wrong right at the edge of ability, trending
# toward "mostly blank" the further over their head a problem is --
# reflecting that most students eventually stop guessing on problems
# far beyond them rather than answering every single one.
DEFAULT_INTERCEPT = -0.3
DEFAULT_SLOPE = 0.0022  # per Elo point of (difficulty - ability)


@dataclass
class BlankModel:
    intercept: float
    slope: float
    n_fit: int
    fitted_from_data: bool

    def p_blank_given_not_correct(self, gap: float) -> float:
        z = self.intercept + self.slope * gap
        return 1.0 / (1.0 + np.exp(-z))


def fit_blank_model(response_df: pd.DataFrame, ability_profile) -> BlankModel:
    """Fit P(blank | not correct) ~ logistic(gap) from the student's own
    non-correct historical responses, where gap = difficulty - ability
    for the problem's (average-of-)categories.

    Requires at least a few blanks AND a few wrongs to fit a meaningful
    slope; otherwise falls back to DEFAULT_INTERCEPT/DEFAULT_SLOPE.
    """
    df = response_df[response_df["outcome"].isin(["wrong", "blank"])].copy()
    if df.empty:
        return BlankModel(DEFAULT_INTERCEPT, DEFAULT_SLOPE, 0, False)

    def gap_for(row) -> float:
        cats = row["categories"] or []
        if not cats:
            ability = ability_profile.theta_global
        else:
            ability = float(np.mean([ability_profile.ability_for(c) for c in cats]))
        return row["seed_elo"] - ability

    df["gap"] = df.apply(gap_for, axis=1)
    df["is_blank"] = (df["outcome"] == "blank").astype(float)

    n_blank = int(df["is_blank"].sum())
    n_wrong = len(df) - n_blank
    if n_blank < 3 or n_wrong < 3:
        return BlankModel(DEFAULT_INTERCEPT, DEFAULT_SLOPE, len(df), False)

    # Simple 1-D logistic regression via Newton's method (avoids a hard
    # sklearn dependency for such a small, well-conditioned problem).
    x = df["gap"].to_numpy()
    y = df["is_blank"].to_numpy()
    x_std = x.std() if x.std() > 1e-6 else 1.0
    x_norm = x / x_std

    beta = np.zeros(2)  # [intercept, slope on x_norm]
    design = np.column_stack([np.ones_like(x_norm), x_norm])
    for _ in range(50):
        z = design @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        grad = design.T @ (y - p)
        w = p * (1 - p)
        hessian = -(design.T * w) @ design - 1e-3 * np.eye(2)  # tiny ridge for stability
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta - step
        if np.max(np.abs(step)) < 1e-6:
            break

    intercept, slope_norm = beta
    slope = slope_norm / x_std
    return BlankModel(float(intercept), float(slope), len(df), True)
