"""Calibration and backtesting harness.

We don't have real longitudinal student response logs (that data isn't
public), so we validate the predictor against **synthetic students**
with known, fixed ability profiles: we draw a ground-truth ability per
broad category from a prior, simulate that student's response to every
real historical problem using the *same* Elo-style item response
function the model itself assumes (plus a small amount of independent
`response noise` -- extra per-attempt logit jitter representing the
messiness of a real test-taker (careless errors, lucky guesses, etc.))
so the harness isn't simply asking the model to recover parameters it
was directly initialized with, then check whether our pipeline (fit on
some contests, predict on others) recovers scores close to that
student's *actual simulated* performance.

This is explicitly a **synthetic-data validation**, not a claim about
real-world accuracy on real students -- that would require access to
real longitudinal AMC/LIVE response logs, which this project does not
have. Every function/report here says so.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.data_access import ContestResponses, build_response_frame
from models.ability_model import fit_ability_profile
from models.blank_model import fit_blank_model
from simulation.monte_carlo import simulate_known_contest


@dataclass
class SyntheticStudent:
    category_thetas: dict[str, float]
    global_theta: float
    blank_intercept: float
    blank_slope: float
    noise_sd: float


def generate_synthetic_student(rng: np.random.Generator, categories: list[str], skill_level: float = 0.0, noise_sd: float = 120.0) -> SyntheticStudent:
    """skill_level shifts the student's overall mean ability (Elo-scale);
    per-category thetas are drawn around it with realistic spread."""
    global_theta = skill_level + rng.normal(0, 150)
    category_thetas = {c: global_theta + rng.normal(0, 200) for c in categories}
    return SyntheticStudent(
        category_thetas=category_thetas,
        global_theta=global_theta,
        blank_intercept=rng.uniform(-1.0, 0.5),
        blank_slope=rng.uniform(0.001, 0.004),
        noise_sd=noise_sd,
    )


def simulate_student_on_contest(rng: np.random.Generator, student: SyntheticStudent, contest_problems: pd.DataFrame) -> list[str | None]:
    """Simulate one response vector (25 letters/None) for a synthetic
    student against a real historical contest's actual problems, using
    the Elo item-response function plus response noise and the
    student's own blank tendency."""
    responses: list[str | None] = []
    ordered = contest_problems.sort_values("position")
    for _, row in ordered.iterrows():
        cats = row["categories"] or []
        ability = np.mean([student.category_thetas.get(c, student.global_theta) for c in cats]) if cats else student.global_theta
        ability_noisy = ability + rng.normal(0, student.noise_sd)
        difficulty = row["seed_elo"]
        p_corr = 1.0 / (1.0 + 10.0 ** ((difficulty - ability_noisy) / 400.0))
        if rng.random() < p_corr:
            responses.append(str(row["answer"]))
        else:
            gap = difficulty - ability
            p_blank = 1.0 / (1.0 + np.exp(-(student.blank_intercept + student.blank_slope * gap)))
            if rng.random() < p_blank:
                responses.append(None)
            else:
                # a uniformly random *wrong* choice among the other 4 letters
                wrong_letters = [l for l in "abcde" if l != str(row["answer"])]
                responses.append(rng.choice(wrong_letters))
    return responses


def _score_of_responses(contest_problems: pd.DataFrame, responses: list[str | None]) -> float:
    ordered = contest_problems.sort_values("position")
    correct = sum(1 for r, (_, row) in zip(responses, ordered.iterrows()) if r is not None and str(r).lower() == str(row["answer"]).lower())
    blank = sum(1 for r in responses if r is None)
    return correct * 6.0 + blank * 1.5


@dataclass
class BacktestRecord:
    student_idx: int
    held_out_contest: str
    n_train_contests: int
    actual_score: float
    predicted_median: float
    predicted_mean: float
    predicted_p5: float
    predicted_p95: float

    @property
    def abs_error(self) -> float:
        return abs(self.actual_score - self.predicted_median)

    @property
    def in_90pct_interval(self) -> bool:
        return self.predicted_p5 <= self.actual_score <= self.predicted_p95


def _predict_held_out(train_response_df: pd.DataFrame, held_out_problems: pd.DataFrame, n_sims: int, rng: np.random.Generator) -> dict:
    profile = fit_ability_profile(train_response_df)
    blank_model = fit_blank_model(train_response_df, profile)
    sim = simulate_known_contest(held_out_problems, profile, blank_model, n_simulations=n_sims, rng=rng)
    s = sim.summary()
    return {
        "median": s["median_score"],
        "mean": s["mean_score"],
        "p5": s["percentiles"]["p5"],
        "p95": s["percentiles"]["p95"],
    }


def run_backtest(
    problems: pd.DataFrame,
    contest_ids_in_order: list[str],
    n_synthetic_students: int = 30,
    mode: str = "rolling",  # "rolling" or "leave_one_out"
    n_sims: int = 4000,
    seed: int = 42,
) -> list[BacktestRecord]:
    """mode="rolling": for each student, for each contest index i>=1, train
    on contests[0:i] and predict contests[i] (a genuine forward-in-time
    rolling-window backtest).
    mode="leave_one_out": for each student, for each contest, train on
    all OTHER contests and predict that one (leave-one-test-out)."""
    rng = np.random.default_rng(seed)
    categories = sorted({c for cats in problems["categories"] for c in cats})
    records: list[BacktestRecord] = []

    for student_idx in range(n_synthetic_students):
        skill_level = rng.uniform(-400, 400)
        student = generate_synthetic_student(rng, categories, skill_level=skill_level)
        responses_by_contest: dict[str, list[str | None]] = {}
        for cid in contest_ids_in_order:
            contest_problems = problems[problems["contest_id"] == cid]
            responses_by_contest[cid] = simulate_student_on_contest(rng, student, contest_problems)

        if mode == "rolling":
            pairs = [(contest_ids_in_order[:i], contest_ids_in_order[i]) for i in range(1, len(contest_ids_in_order))]
        elif mode == "leave_one_out":
            pairs = [
                ([c for c in contest_ids_in_order if c != held], held)
                for held in contest_ids_in_order
            ]
        else:
            raise ValueError(f"unknown mode {mode!r}")

        for train_ids, held_out_id in pairs:
            train_cr = [ContestResponses(cid, responses_by_contest[cid]) for cid in train_ids]
            train_df = build_response_frame(problems, train_cr)
            held_out_problems = problems[problems["contest_id"] == held_out_id]
            pred = _predict_held_out(train_df, held_out_problems, n_sims, rng)
            actual = _score_of_responses(held_out_problems, responses_by_contest[held_out_id])
            records.append(
                BacktestRecord(
                    student_idx=student_idx,
                    held_out_contest=held_out_id,
                    n_train_contests=len(train_ids),
                    actual_score=actual,
                    predicted_median=pred["median"],
                    predicted_mean=pred["mean"],
                    predicted_p5=pred["p5"],
                    predicted_p95=pred["p95"],
                )
            )
    return records


def summarize_backtest(records: list[BacktestRecord]) -> dict:
    if not records:
        return {"n": 0}
    errors = np.array([r.abs_error for r in records])
    sq_errors = errors ** 2
    coverage = np.mean([r.in_90pct_interval for r in records])
    return {
        "n": len(records),
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(sq_errors))),
        "coverage_90pct_interval": float(coverage),
        "calibration_error": float(abs(coverage - 0.90)),
        "note": (
            "Computed against SYNTHETIC students with known simulated ability "
            "profiles (see analytics/calibration.py), not real AMC test-takers -- "
            "real longitudinal response data was not available."
        ),
    }
