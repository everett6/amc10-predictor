"""Top-level orchestration: turn a student's historical AMC 10 responses
into a full 2026 prediction (score distribution + explanation +
counterfactual + question-level detail)."""
from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from analytics.data_access import ContestResponses, build_response_frame, load_problems
from analytics.topic_analysis import (
    difficulty_bucket_accuracy,
    position_accuracy,
    topic_accuracy,
)
from models.ability_model import AbilityProfile, fit_ability_profile
from models.blank_model import BlankModel, fit_blank_model
from scoring.amc_scoring import score_from_responses
from simulation.monte_carlo import DEFAULT_NUM_SIMULATIONS, build_position_pools, run_simulation


@dataclass
class QuestionLevelPrediction:
    position: int
    expected_p_correct: float           # averaged over simulated difficulty/topic at this slot
    typical_difficulty: float           # mean bootstrapped difficulty at this slot
    likely_categories: list[str]        # most common historical categories at this slot


@dataclass
class CounterfactualResult:
    category: str
    current_ability: float
    improved_ability: float
    delta_elo: int
    baseline_mean_score: float
    improved_mean_score: float
    score_gain: float


@dataclass
class PredictionResult:
    input_scores: dict[str, float]                 # contest_id -> actual historical score
    ability_global: float
    ability_by_category: dict[str, float]
    topic_accuracy: list[dict]
    difficulty_bucket_accuracy: list[dict]
    position_accuracy: list[dict]
    simulation_summary: dict
    histogram: list[dict]
    question_level: list[dict]
    counterfactuals: list[dict]
    explanation: str
    n_simulations: int
    contests_used: list[str]


def _weakest_strongest_categories(ability_profile: AbilityProfile, min_attempts: int = 2) -> tuple[list[str], list[str]]:
    seen = [(cat, info) for cat, info in ability_profile.categories.items() if info.n_attempted >= min_attempts]
    seen.sort(key=lambda kv: kv[1].theta_smoothed)
    weakest = [cat for cat, _ in seen[:3]]
    strongest = [cat for cat, _ in seen[-3:]][::-1]
    return weakest, strongest


def build_explanation(
    ability_profile: AbilityProfile,
    sim_summary: dict,
    contests_used: list[str],
) -> str:
    weakest, strongest = _weakest_strongest_categories(ability_profile)
    parts = [
        f"This prediction is built from {len(contests_used)} historical contest attempt(s) "
        f"({', '.join(contests_used)}), covering {ability_profile.n_total_attempted} attempted "
        f"(non-blank) problems with {ability_profile.n_total_correct} correct.",
        "We fit a per-topic ability rating for each of the 15 broad concept categories using an "
        "Elo-style item-response model (LIVE by Po-Shen Loh's own crowd-calibrated difficulty rating "
        "for each historical problem vs. your correct/incorrect outcome), with more recent contests "
        "weighted more heavily and small-sample categories smoothed toward your overall ability via "
        "hierarchical Bayesian shrinkage so a single lucky/unlucky problem in a rarely-tested topic "
        "doesn't swing that topic's estimate too far.",
    ]
    if strongest:
        parts.append(f"Your strongest topics so far: {', '.join(strongest)}.")
    if weakest:
        parts.append(f"Your weakest topics so far: {', '.join(weakest)}.")
    parts.append(
        f"To project a 2026 score, we ran {sim_summary['n_simulations']:,} Monte Carlo simulations of a "
        "synthetic 2026 AMC 10: for each of the 25 problem slots, we bootstrap-sampled a difficulty rating "
        "and topic mix from the corresponding slot across all 18 historical contests we ingested (2018-2025 "
        "A/B, including the Fall 2021 makeup contests), applied your topic ability to get a probability of "
        "answering correctly, and used a model of your own blank-vs-guess tendency (fit from your history) "
        "to split misses into blanks (+1.5 pts) vs. wrong answers (+0 pts)."
    )
    parts.append(
        f"Result: a median projected score of {sim_summary['median_score']:.1f} "
        f"(mean {sim_summary['mean_score']:.1f}), with a 90% interval of "
        f"[{sim_summary['percentiles']['p5']:.1f}, {sim_summary['percentiles']['p95']:.1f}], "
        f"averaging {sim_summary['expected_correct']:.1f} correct, "
        f"{sim_summary['expected_wrong']:.1f} wrong, and {sim_summary['expected_blank']:.1f} blank."
    )
    return " ".join(parts)


def question_level_predictions(problems: pd.DataFrame, sim_result) -> list[QuestionLevelPrediction]:
    out = []
    for position in range(1, 26):
        col = position - 1
        p_mean = float(np.mean(sim_result.per_position_p_correct[:, col]))
        diff_mean = float(np.mean(sim_result.per_position_difficulty[:, col]))
        subset = problems[problems["position"] == position]
        cat_counts: dict[str, int] = {}
        for cats in subset["categories"]:
            for c in cats:
                cat_counts[c] = cat_counts.get(c, 0) + 1
        top_cats = sorted(cat_counts, key=cat_counts.get, reverse=True)[:3]
        out.append(
            QuestionLevelPrediction(
                position=position,
                expected_p_correct=p_mean,
                typical_difficulty=diff_mean,
                likely_categories=top_cats,
            )
        )
    return out


def compute_counterfactuals(
    position_pools,
    ability_profile: AbilityProfile,
    blank_model: BlankModel,
    baseline_mean: float,
    n_sims: int,
    rng: np.random.Generator,
    delta_elo: int = 150,
    top_k: int = 3,
) -> list[CounterfactualResult]:
    """"What if you improved topic X by delta_elo Elo points?" -- re-run
    the simulation with only that one category's ability bumped up, all
    else held fixed, to isolate its marginal effect on the score
    distribution. Limited to the student's weakest attempted topics
    (the ones improvement advice is actually about)."""
    weakest, _ = _weakest_strongest_categories(ability_profile, min_attempts=1)
    results = []
    for category in weakest[:top_k]:
        current = ability_profile.ability_for(category)
        improved_profile = AbilityProfile(
            theta_global=ability_profile.theta_global,
            categories={**ability_profile.categories},
            n_total_attempted=ability_profile.n_total_attempted,
            n_total_correct=ability_profile.n_total_correct,
        )
        from models.ability_model import CategoryAbility

        existing = ability_profile.categories.get(category)
        improved_profile.categories[category] = CategoryAbility(
            category=category,
            theta_mle=existing.theta_mle if existing else None,
            theta_smoothed=current + delta_elo,
            n_attempted=existing.n_attempted if existing else 0,
            n_correct=existing.n_correct if existing else 0,
        )
        sim = run_simulation(position_pools, improved_profile, blank_model, n_simulations=n_sims, rng=rng)
        improved_mean = sim.summary()["mean_score"]
        results.append(
            CounterfactualResult(
                category=category,
                current_ability=current,
                improved_ability=current + delta_elo,
                delta_elo=delta_elo,
                baseline_mean_score=baseline_mean,
                improved_mean_score=improved_mean,
                score_gain=improved_mean - baseline_mean,
            )
        )
    return results


def predict_2026(
    conn: sqlite3.Connection,
    contest_responses: list[ContestResponses],
    n_simulations: int = DEFAULT_NUM_SIMULATIONS,
    seed: int | None = None,
) -> PredictionResult:
    rng = np.random.default_rng(seed)
    problems = load_problems(conn)
    response_df = build_response_frame(problems, contest_responses)

    input_scores = {}
    for cr in contest_responses:
        subset = problems[problems["contest_id"] == cr.contest_id].sort_values("position")
        answer_key = list(subset["answer"])
        breakdown = score_from_responses(cr.responses, answer_key)
        input_scores[cr.contest_id] = breakdown.score

    ability_profile = fit_ability_profile(response_df)
    blank_model = fit_blank_model(response_df, ability_profile)
    position_pools = build_position_pools(problems)

    sim = run_simulation(position_pools, ability_profile, blank_model, n_simulations=n_simulations, rng=rng)
    sim_summary = sim.summary()

    counterfactuals = compute_counterfactuals(
        position_pools, ability_profile, blank_model, sim_summary["mean_score"], n_sims=max(8000, n_simulations), rng=rng
    )

    explanation = build_explanation(ability_profile, sim_summary, [cr.contest_id for cr in contest_responses])
    ql = question_level_predictions(problems, sim)

    return PredictionResult(
        input_scores=input_scores,
        ability_global=ability_profile.theta_global,
        ability_by_category={c: info.theta_smoothed for c, info in ability_profile.categories.items()},
        topic_accuracy=topic_accuracy(response_df).to_dict(orient="records"),
        difficulty_bucket_accuracy=difficulty_bucket_accuracy(response_df).to_dict(orient="records"),
        position_accuracy=position_accuracy(response_df).to_dict(orient="records"),
        simulation_summary=sim_summary,
        histogram=sim.histogram(),
        question_level=[asdict(q) for q in ql],
        counterfactuals=[asdict(c) for c in counterfactuals],
        explanation=explanation,
        n_simulations=n_simulations,
        contests_used=[cr.contest_id for cr in contest_responses],
    )
