"""Forecast of what the *next* AMC 10 looks like, slot by slot.

AMC 10 difficulty has drifted upward over 25 years (mean LIVE rating
about 1320 in 2001 vs about 1640 in 2025; the last five problems went
from about 1800 to about 2180). Pooling all 51 contests with equal weight
would therefore simulate a 2026 contest that is too easy. This module
builds per-position pools of (difficulty, topics) samples with a
year-based exponential weight, so recent contests dominate but older
ones still add topic variety:

    weight(year) = 0.5 ** ((target_year - year) / half_life_years)

The half-life is not a guess: scripts/backtest_contest_model.py replays
history (forecast each year's contests from earlier years only) and the
default below is the value that scored best there.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ingestion.concepts import BROAD_CATEGORIES

DEFAULT_TARGET_YEAR = 2026
# Chosen by scripts/backtest_contest_model.py (see docs/methodology.md).
DEFAULT_HALF_LIFE_YEARS = 3.0
DEFAULT_LEVEL_MODEL = "trend"
DEFAULT_LEVEL_HALF_LIFE_YEARS: float | None = 8.0

CATEGORY_INDEX = {c: i for i, c in enumerate(BROAD_CATEGORIES)}
N_CATEGORIES = len(BROAD_CATEGORIES)


def category_weights(categories: list[str]) -> np.ndarray:
    """Row vector over the 15 categories: 1/k on each of a problem's k
    categories (a problem's topic effect is the mean of its categories')."""
    row = np.zeros(N_CATEGORIES)
    known = [c for c in categories if c in CATEGORY_INDEX]
    for c in known:
        row[CATEGORY_INDEX[c]] = 1.0 / len(known)
    return row


@dataclass
class PositionPool:
    position: int
    seed_elos: np.ndarray          # (m,) raw difficulty ratings
    residuals: np.ndarray          # (m,) rating minus its own contest's mean
    category_matrix: np.ndarray    # (m, 15), rows from category_weights()
    probs: np.ndarray              # (m,), sampling probabilities, sum to 1
    categories_per_sample: list[list[str]]

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.choice(len(self.seed_elos), size=n, p=self.probs)


@dataclass
class ContestForecast:
    """Generative model of a future contest:

        difficulty[pos] = level + residual[pos]

    `level` is the contest's overall difficulty (mean rating), drawn from
    a forecast distribution N(level_mean, level_sd^2) fitted to the
    history of contest means; `residual[pos]` is a weighted bootstrap
    draw from that slot's historical (rating - contest mean) values.
    The shared level term is what lets a whole simulated contest come
    out hard or easy together, as real ones do.
    """

    pools: dict[int, PositionPool]
    level_mean: float | None       # None => legacy mode: raw ratings, no shared level
    level_sd: float
    target_year: int

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Returns (difficulty[n,25], idx[n,25]); idx indexes each slot's pool."""
        difficulty = np.zeros((n, 25))
        idx = np.zeros((n, 25), dtype=int)
        for position in range(1, 26):
            pool = self.pools[position]
            draw = pool.sample(rng, n)
            idx[:, position - 1] = draw
            base = pool.seed_elos if self.level_mean is None else pool.residuals
            difficulty[:, position - 1] = base[draw]
        if self.level_mean is not None:
            level = rng.normal(self.level_mean, self.level_sd, size=n)
            difficulty += level[:, None]
        return difficulty, idx

    def category_matrix(self, idx: np.ndarray) -> np.ndarray:
        """(n,25,15) category weights for sampled contests."""
        out = np.zeros(idx.shape + (N_CATEGORIES,))
        for position in range(1, 26):
            out[:, position - 1, :] = self.pools[position].category_matrix[idx[:, position - 1]]
        return out


def _year_weights(years: np.ndarray, target_year: int, half_life: float | None) -> np.ndarray:
    if half_life is None:
        return np.ones(len(years))
    return 0.5 ** ((target_year - years) / half_life)


def build_contest_forecast(
    problems: pd.DataFrame,
    target_year: int = DEFAULT_TARGET_YEAR,
    half_life_years: float | None = DEFAULT_HALF_LIFE_YEARS,
    level: str = DEFAULT_LEVEL_MODEL,
    level_half_life_years: float | None = DEFAULT_LEVEL_HALF_LIFE_YEARS,
    max_year: int | None = None,
) -> ContestForecast:
    """level: "none" (v1: raw ratings, independent slots), "mean" (weighted
    mean of past contest means) or "trend" (weighted linear trend in year,
    extrapolated to target_year). max_year restricts history (backtests)."""
    df = problems if max_year is None else problems[problems["year"] <= max_year]
    contest_mean = df.groupby("contest_id")["seed_elo"].transform("mean")
    df = df.assign(residual=df["seed_elo"] - contest_mean)

    pools: dict[int, PositionPool] = {}
    for position, group in df.groupby("position"):
        w = _year_weights(group["year"].to_numpy(dtype=float), target_year, half_life_years)
        cats = list(group["categories"])
        pools[int(position)] = PositionPool(
            position=int(position),
            seed_elos=group["seed_elo"].to_numpy(dtype=float),
            residuals=group["residual"].to_numpy(dtype=float),
            category_matrix=np.vstack([category_weights(c) for c in cats]),
            probs=w / w.sum(),
            categories_per_sample=cats,
        )

    if level == "none":
        return ContestForecast(pools, None, 0.0, target_year)

    per_contest = df.groupby("contest_id").agg(year=("year", "first"), mean=("seed_elo", "mean"))
    x = per_contest["year"].to_numpy(dtype=float) - target_year
    y = per_contest["mean"].to_numpy(dtype=float)
    w = _year_weights(per_contest["year"].to_numpy(dtype=float), target_year, level_half_life_years)
    design = np.ones((len(x), 1)) if level == "mean" else np.column_stack([np.ones_like(x), x])
    wx = design * w[:, None]
    xtwx_inv = np.linalg.inv(design.T @ wx)
    beta = xtwx_inv @ (wx.T @ y)
    resid = y - design @ beta
    # Weighted residual variance with an effective-sample-size correction.
    n_eff = w.sum() ** 2 / (w**2).sum()
    dof = max(n_eff - design.shape[1], 1.0)
    sigma2 = float((w * resid**2).sum() / w.sum() * n_eff / dof)
    # Predicting at x=0 (the target year): new-contest noise plus the
    # uncertainty of the fitted level itself.
    x0 = np.zeros(design.shape[1]); x0[0] = 1.0
    param_var = sigma2 * float(x0 @ xtwx_inv @ (wx.T @ wx) @ xtwx_inv @ x0) / 1.0
    return ContestForecast(pools, float(beta[0]), float(np.sqrt(sigma2 + param_var)), target_year)


def build_position_pools(
    problems: pd.DataFrame,
    target_year: int = DEFAULT_TARGET_YEAR,
    half_life_years: float | None = DEFAULT_HALF_LIFE_YEARS,
    max_year: int | None = None,
) -> dict[int, PositionPool]:
    return build_contest_forecast(problems, target_year, half_life_years, "none", None, max_year).pools
