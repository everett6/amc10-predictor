"""End-to-end validation on simulated students, old pipeline vs new.

No dataset of real students' answer sheets across several AMC 10 papers
is available, so real-world accuracy cannot be measured here. What can
be done is a stress test: simulate students from a process that is
deliberately NOT the model being tested, and see which pipeline predicts
their next score best. Results say which design is more robust; they are
not a claim about error on real students.

The simulated student ("world") differs from every model in the app:
  * skills vary by fine-grained concept (256 of them), not just by the
    15 broad categories the models use;
  * each student has their own response slope (discrimination 0.6-1.3);
  * answers come from a know-it / don't-know-it process: knowing gives
    the right answer 97% of the time; not knowing leads to a guess
    (right 25% of the time) or a blank, by a personal guessing habit;
  * time runs out: each student has a personal cut-off position;
  * ability shifts from sitting to sitting (sd 70) and grows over time.

Protocol (forward in time, like real use): pick a target contest from
year Y, give the pipeline K earlier sittings on papers from before Y,
build the contest forecast from papers before Y only, then compare the
predicted score distribution with the student's simulated score on the
real target paper.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.data_access import ContestResponses, build_response_frame
from ingestion.concepts import BROAD_CATEGORIES, concept_to_category
from models.ability_model import fit_ability_profile
from models.blank_model import fit_blank_model
from models.members import ELO_K, IrtMember
from scoring.predictor_v2 import fit_student, run_simulation
from simulation.contest_model import build_contest_forecast
from simulation.monte_carlo import build_position_pools as v1_pools
from simulation.monte_carlo import run_simulation as v1_run


@dataclass
class WorldStudent:
    theta: float
    cat_offset: dict[str, float]
    concept_offset: dict[str, float]
    discrimination: float
    guess_habit: float           # P(guess | don't know) baseline
    cutoff: float                # position around which time runs out
    growth_per_year: float
    day_sd: float = 70.0


def make_student(rng: np.random.Generator, concepts: list[str]) -> WorldStudent:
    return WorldStudent(
        theta=rng.normal(1450.0, 300.0),
        cat_offset={c: rng.normal(0.0, 130.0) for c in BROAD_CATEGORIES},
        concept_offset={c: rng.normal(0.0, 100.0) for c in concepts},
        discrimination=rng.uniform(0.6, 1.3),
        guess_habit=rng.uniform(0.1, 0.8),
        cutoff=rng.uniform(18.0, 30.0),
        growth_per_year=rng.normal(60.0, 80.0),
    )


def sit(rng: np.random.Generator, s: WorldStudent, contest_problems: pd.DataFrame, years_before_target: float) -> list[str | None]:
    theta_day = s.theta - s.growth_per_year * years_before_target + rng.normal(0.0, s.day_sd)
    responses: list[str | None] = []
    for _, row in contest_problems.sort_values("position").iterrows():
        concepts = row["concepts"] or []
        if concepts:
            skill = np.mean([s.cat_offset.get(concept_to_category(c), 0.0) + s.concept_offset.get(c, 0.0) for c in concepts])
        else:
            skill = 0.0
        gap = row["seed_elo"] - (theta_day + skill)
        reached = rng.random() < 1.0 / (1.0 + np.exp((row["position"] - s.cutoff) / 2.0))
        if not reached:
            responses.append(None)
            continue
        knows = rng.random() < 1.0 / (1.0 + np.exp(s.discrimination * ELO_K * gap))
        key = str(row["answer"])
        wrong = [l for l in "abcde" if l != key]
        if knows:
            responses.append(key if rng.random() < 0.97 else str(rng.choice(wrong)))
        elif rng.random() < s.guess_habit * (1.0 / (1.0 + np.exp((gap - 400.0) / 300.0))):
            responses.append(key if rng.random() < 0.25 else str(rng.choice(wrong)))
        else:
            responses.append(None)
    return responses


def actual_score(contest_problems: pd.DataFrame, responses: list[str | None]) -> float:
    key = list(contest_problems.sort_values("position")["answer"])
    correct = sum(1 for r, k in zip(responses, key) if r is not None and r == k)
    return 6.0 * correct + 1.5 * sum(1 for r in responses if r is None)


def crps(samples: np.ndarray, actual: float, rng: np.random.Generator) -> float:
    """Continuous ranked probability score from samples (lower is better)."""
    sub = samples if len(samples) <= 1500 else rng.choice(samples, 1500, replace=False)
    return float(np.mean(np.abs(sub - actual)) - 0.5 * np.mean(np.abs(sub[:, None] - sub[None, :])))


SYSTEMS = ("v1", "v2_irt_only", "v2_old_contest_model", "v2_full")


def predict_scores(system: str, problems: pd.DataFrame, history: pd.DataFrame, sittings: list[ContestResponses],
                   target_year: int, target_date: datetime.date, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Simulated score samples for the unseen target-year paper."""
    if system == "v1":
        frame = build_response_frame(problems, sittings)
        profile = fit_ability_profile(frame)
        blank = fit_blank_model(frame, profile)
        return v1_run(v1_pools(history), profile, blank, n_simulations=n_sims, rng=rng).scores
    members = [IrtMember("irt_hier")] if system == "v2_irt_only" else None
    student = fit_student(problems, sittings, target_date, rng, members)
    if system == "v2_old_contest_model":
        forecast = build_contest_forecast(history, target_year, half_life_years=None, level="none")
    else:
        forecast = build_contest_forecast(history, target_year)
    return run_simulation(student, forecast, n_sims, rng)[0].scores


def run_validation(problems: pd.DataFrame, k_values=(1, 3, 6), first_year: int = 2019,
                   students_per_target: int = 8, n_sims: int = 3000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    concepts = sorted({c for cs in problems["concepts"] for c in cs})
    targets = problems[problems["year"] >= first_year][["contest_id", "year"]].drop_duplicates()
    rows = []
    for _, target in targets.iterrows():
        year = int(target["year"])
        target_date = datetime.date(year, 11, 5)
        history = problems[problems["year"] < year]
        history_ids = sorted(history["contest_id"].unique())
        target_problems = problems[problems["contest_id"] == target["contest_id"]]
        for _ in range(students_per_target):
            student = make_student(rng, concepts)
            truth = actual_score(target_problems, sit(rng, student, target_problems, 0.0))
            for k in k_values:
                chosen = rng.choice(history_ids, size=k, replace=False)
                # sittings spread over the year before the contest, oldest first
                gaps = np.sort(rng.uniform(0.05, 1.0, size=k))[::-1]
                sittings = [
                    ContestResponses(cid, sit(rng, student, problems[problems["contest_id"] == cid], float(g)),
                                     target_date - datetime.timedelta(days=int(g * 365.25)))
                    for cid, g in zip(chosen, gaps)
                ]
                for system in SYSTEMS:
                    scores = predict_scores(system, problems, history, sittings, year, target_date, n_sims, rng)
                    lo5, lo25, med, hi75, hi95 = np.percentile(scores, [5, 25, 50, 75, 95])
                    rows.append({
                        "system": system, "k": k, "target": target["contest_id"], "actual": truth,
                        "median": med, "mean": float(scores.mean()),
                        "in90": lo5 <= truth <= hi95, "in50": lo25 <= truth <= hi75,
                        "width90": hi95 - lo5, "crps": crps(scores, truth, rng),
                    })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(err=df["median"] - df["actual"])
    g = df.groupby(["k", "system"])
    out = pd.DataFrame({
        "n": g.size(),
        "bias": g["err"].mean(),
        "mae": g["err"].apply(lambda e: e.abs().mean()),
        "rmse": g["err"].apply(lambda e: float(np.sqrt((e**2).mean()))),
        "crps": g["crps"].mean(),
        "cover90": g["in90"].mean(),
        "cover50": g["in50"].mean(),
        "width90": g["width90"].mean(),
    })
    return out.reset_index()
