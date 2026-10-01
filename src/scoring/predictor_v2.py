"""v2 prediction pipeline: sittings in, full 2026 forecast out.

fit_student   fits the stacked ensemble and the answer-or-blank model
predict_v2    adds the contest forecast, runs the Monte Carlo engine and
              assembles everything the UI shows
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.data_access import ContestResponses, build_response_frame
from analytics.topic_analysis import difficulty_bucket_accuracy, position_accuracy, topic_accuracy
from ingestion.concepts import BROAD_CATEGORIES
from models.attempt_model import AttemptModel, fit_attempt_model
from models.ensemble import Ensemble, fit_ensemble, items_from_frame
from models.members import ELO_K, Items
from scoring.amc_scoring import blank_points_for_year, score_from_responses
from simulation.contest_model import (
    DEFAULT_TARGET_YEAR, N_CATEGORIES, ContestForecast, build_contest_forecast,
)
from simulation.engine import FixedContest, SimResult, make_draws, simulate

# AMC 10A is held in early November. The exact 2026 date is an assumption;
# it only matters when sittings are dated and a growth trend is fitted.
DEFAULT_TARGET_DATE = datetime.date(2026, 11, 5)
DEFAULT_N_SIMULATIONS = 20_000
CHUNK = 20_000
COUNTERFACTUAL_BOOST = 150.0
DEFAULT_TARGETS = (90.0, 100.0, 110.0, 120.0)


@dataclass
class StudentModel:
    ensemble: Ensemble
    attempt_model: AttemptModel
    ability_global: float
    ability_offsets: np.ndarray          # (15,)
    response_df: pd.DataFrame
    items: Items


def fit_student(
    problems: pd.DataFrame,
    contest_responses: list[ContestResponses],
    target_date: datetime.date = DEFAULT_TARGET_DATE,
    rng: np.random.Generator | None = None,
    members: list | None = None,
) -> StudentModel:
    rng = rng or np.random.default_rng(0)
    response_df = build_response_frame(problems, contest_responses)
    items = items_from_frame(response_df, target_date)
    if len(items) == 0:
        # Nothing answered at all: fit on a single neutral pseudo-observation
        # so every member falls back to its prior.
        items = Items(np.array([1400.0]), np.zeros((1, N_CATEGORIES)), np.array([13.0]),
                      np.array([0]), np.zeros(1), np.array([0.0]))
        ensemble = fit_ensemble(items, rng, members)
        ensemble.cv_report["note"] = "No answered problems: abilities come from the prior alone."
    else:
        ensemble = fit_ensemble(items, rng, members)

    hier = ensemble.member("irt_hier") if "irt_hier" in ensemble.names else ensemble.members[0]
    summary = hier.ability_summary()
    ability_global = summary["global_mean"]
    offsets = np.array([summary["categories"].get(i, (ability_global, 0.0))[0] - ability_global
                        for i in range(N_CATEGORIES)])

    every = items_from_frame(response_df, target_date, attempted_only=False)
    answered = (response_df["outcome"] != "blank").to_numpy()
    ability_item = ability_global + every.cat @ offsets
    attempt_model = fit_attempt_model(every.difficulty, ability_item, every.position, answered)
    return StudentModel(ensemble, attempt_model, ability_global, offsets, response_df, items)


def run_simulation(
    student: StudentModel,
    contest: ContestForecast | FixedContest,
    n: int,
    rng: np.random.Generator,
    boosts: list[np.ndarray] | None = None,
) -> tuple[SimResult, list[SimResult]]:
    """Baseline result plus one result per what-if boost, on shared draws."""
    base_parts: list[SimResult] = []
    boost_parts: list[list[SimResult]] = [[] for _ in (boosts or [])]
    remaining = n
    while remaining > 0:
        m = min(CHUNK, remaining)
        draws = make_draws(contest, student.ensemble, student.attempt_model, m, rng)
        args = (draws, student.ensemble, student.attempt_model, student.ability_global, student.ability_offsets)
        base_parts.append(simulate(*args))
        for k, boost in enumerate(boosts or []):
            boost_parts[k].append(simulate(*args, category_boost=boost))
        remaining -= m

    def merge(parts: list[SimResult]) -> SimResult:
        cat = lambda name: np.concatenate([getattr(p, name) for p in parts])  # noqa: E731
        return SimResult(cat("scores"), cat("correct"), cat("wrong"), cat("blank"),
                         cat("p_answer"), cat("p_correct_if_answered"), cat("difficulty"))

    return merge(base_parts), [merge(p) for p in boost_parts]


def _input_scores(problems: pd.DataFrame, contest_responses: list[ContestResponses]) -> list[dict]:
    out = []
    for cr in contest_responses:
        subset = problems[problems["contest_id"] == cr.contest_id].sort_values("position")
        key = list(subset["answer"])
        year = int(subset["year"].iloc[0])
        now = score_from_responses(cr.responses, key)
        then = score_from_responses(cr.responses, key, blank_points_for_year(year))
        out.append({
            "contest_id": cr.contest_id, "year": year,
            "taken_on": cr.taken_on.isoformat() if cr.taken_on else None,
            "correct": now.correct, "wrong": now.wrong, "blank": now.blank,
            "score_current_rules": now.score,
            "score_rules_of_the_year": then.score,
            "blank_points_that_year": blank_points_for_year(year),
        })
    return out


def _topic_table(student: StudentModel) -> list[dict]:
    hier = student.ensemble.member("irt_hier")
    summary = hier.ability_summary()
    counts = {c: [0, 0] for c in BROAD_CATEGORIES}
    for cats, outcome in zip(student.response_df["categories"], student.response_df["outcome"]):
        if outcome == "blank":
            continue
        for c in set(cats):
            if c in counts:
                counts[c][0] += 1
                counts[c][1] += int(outcome == "correct")
    rows = []
    for i, c in enumerate(BROAD_CATEGORIES):
        mean, sd = summary["categories"][i]
        rows.append({
            "category": c, "ability": mean, "ability_sd": sd,
            "ability_low": mean - 1.645 * sd, "ability_high": mean + 1.645 * sd,
            "n_answered": counts[c][0], "n_correct": counts[c][1],
        })
    return sorted(rows, key=lambda r: -r["ability"])


def _difficulty_curve(student: StudentModel) -> list[dict]:
    """Model curve at a typical mid-paper slot with no topic effect."""
    grid = np.arange(400.0, 2801.0, 100.0)
    items = Items(grid, np.zeros((len(grid), N_CATEGORIES)), np.full(len(grid), 13.0),
                  np.full(len(grid), -1), np.zeros(len(grid)))
    p_correct = student.ensemble.predict(items)
    gamma = student.attempt_model.fit_.beta[None, :]
    gap = (grid - student.ability_global)[None, :]
    p_answer = student.attempt_model.predict_with(gamma, gap, np.full(len(grid), 13.0))[0]
    return [{"difficulty": float(d), "p_correct_if_answered": float(pc), "p_answer": float(pa),
             "p_correct": float(pc * pa)} for d, pc, pa in zip(grid, p_correct, p_answer)]


def _explanation(student: StudentModel, inputs: list[dict], summary: dict, forecast_note: str, topics: list[dict]) -> str:
    n_ans = int((student.response_df["outcome"] != "blank").sum())
    n_cor = int((student.response_df["outcome"] == "correct").sum())
    hier = student.ensemble.member("irt_hier").ability_summary()
    seen = [t for t in topics if t["n_answered"] >= 2]
    parts = [
        f"Based on {len(inputs)} sitting{'s' if len(inputs) != 1 else ''} "
        f"({', '.join(i['contest_id'] for i in inputs)}): "
        f"{n_ans} answered problems, {n_cor} correct.",
        f"Estimated overall ability is {hier['global_mean']:.0f} on LIVE's difficulty scale, give or take "
        f"{hier['global_sd']:.0f} (a problem rated at your ability is a coin flip for you).",
    ]
    if seen:
        parts.append(f"Strongest topics: {', '.join(t['category'] for t in seen[:3])}. "
                     f"Weakest: {', '.join(t['category'] for t in seen[-3:][::-1])}.")
    if hier["growth_per_year"] is not None:
        parts.append(f"Your dated sittings show a trend of {hier['growth_per_year']:+.0f} rating points per year "
                     f"(give or take {hier['growth_sd']:.0f}), carried forward to the contest date.")
    parts.append(forecast_note)
    if inputs and max(i["year"] for i in inputs) < DEFAULT_TARGET_YEAR - 4:
        parts.append(
            "All of your sittings are on older papers, which LIVE rates as easier than recent ones, so this forecast "
            "leans on those ratings being comparable across years. Sitting one paper from the last three years "
            "would pin it down much better."
        )
    parts.append(
        f"Across {summary['n_simulations']:,} simulated sittings the median score is {summary['median_score']:.1f} "
        f"(mean {summary['mean_score']:.1f}); 90% of them fall between {summary['percentiles']['p5']:.1f} and "
        f"{summary['percentiles']['p95']:.1f}, with on average {summary['expected_correct']:.1f} correct, "
        f"{summary['expected_wrong']:.1f} wrong and {summary['expected_blank']:.1f} blank."
    )
    return " ".join(parts)


def predict_v2(
    problems: pd.DataFrame,
    contest_responses: list[ContestResponses],
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    seed: int | None = None,
    target_date: datetime.date = DEFAULT_TARGET_DATE,
    targets: tuple[float, ...] = DEFAULT_TARGETS,
    target_year: int = DEFAULT_TARGET_YEAR,
) -> dict:
    rng = np.random.default_rng(seed)
    student = fit_student(problems, contest_responses, target_date, rng)
    forecast = build_contest_forecast(problems, target_year=target_year)
    topics = _topic_table(student)

    # What-if runs: the three weakest topics the student has actually answered.
    answered_topics = [t for t in topics if t["n_answered"] >= 1]
    weakest = answered_topics[-3:][::-1]
    boosts = []
    for t in weakest:
        boost = np.zeros(N_CATEGORIES)
        boost[BROAD_CATEGORIES.index(t["category"])] = COUNTERFACTUAL_BOOST
        boosts.append(boost)

    sim, boosted = run_simulation(student, forecast, n_simulations, rng, boosts)
    summary = sim.summary()

    counterfactuals = [{
        "category": t["category"], "current_ability": t["ability"],
        "improved_ability": t["ability"] + COUNTERFACTUAL_BOOST, "delta_elo": COUNTERFACTUAL_BOOST,
        "baseline_mean_score": summary["mean_score"],
        "improved_mean_score": float(b.scores.mean()),
        "score_gain": float(b.scores.mean() - summary["mean_score"]),
    } for t, b in zip(weakest, boosted)]

    question_level = []
    for pos in range(1, 26):
        pool = forecast.pools[pos]
        share = pool.probs @ (pool.category_matrix > 0)
        top = [BROAD_CATEGORIES[i] for i in np.argsort(-share)[:3]]
        col = pos - 1
        question_level.append({
            "position": pos,
            "typical_difficulty": float(sim.difficulty[:, col].mean()),
            "p_answer": float(sim.p_answer[:, col].mean()),
            "p_correct_if_answered": float(sim.p_correct_if_answered[:, col].mean()),
            "expected_p_correct": float((sim.p_answer[:, col] * sim.p_correct_if_answered[:, col]).mean()),
            "likely_categories": top,
        })

    inputs = _input_scores(problems, contest_responses)
    forecast_note = (
        f"The {target_year} paper is simulated from all {problems['contest_id'].nunique()} contests since 2001, "
        f"weighted toward recent years, with an expected mean problem rating of {forecast.level_mean:.0f} "
        f"(give or take {forecast.level_sd:.0f}) following the upward trend in difficulty."
    )
    hier = student.ensemble.member("irt_hier").ability_summary()
    return {
        "engine": "v2",
        "inputs": inputs,
        "input_scores": {i["contest_id"]: i["score_current_rules"] for i in inputs},
        "ability_global": hier["global_mean"],
        "ability_global_sd": hier["global_sd"],
        "growth_per_year": hier["growth_per_year"],
        "growth_sd": hier["growth_sd"],
        "day_to_day_sd": hier["day_sd"],
        "ability_by_category": {t["category"]: t["ability"] for t in topics},
        "topics": topics,
        "topic_accuracy": topic_accuracy(student.response_df).to_dict(orient="records"),
        "difficulty_bucket_accuracy": difficulty_bucket_accuracy(student.response_df).to_dict(orient="records"),
        "position_accuracy": position_accuracy(student.response_df).to_dict(orient="records"),
        "difficulty_curve": _difficulty_curve(student),
        "model_weights": {n: float(w) for n, w in zip(student.ensemble.names, student.ensemble.weights)},
        "self_check": student.ensemble.cv_report,
        "contest_forecast": {
            "target_year": target_year, "target_date": target_date.isoformat(),
            "level_mean": forecast.level_mean, "level_sd": forecast.level_sd,
            "n_contests": int(problems["contest_id"].nunique()),
        },
        "simulation_summary": summary,
        "histogram": sim.histogram(),
        "target_probabilities": [{"score": float(t), "probability": sim.prob_at_least(t)} for t in targets],
        "question_level": question_level,
        "counterfactuals": counterfactuals,
        "explanation": _explanation(student, inputs, summary, forecast_note, topics),
        "n_simulations": n_simulations,
        "contests_used": [cr.contest_id for cr in contest_responses],
    }
