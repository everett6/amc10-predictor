import numpy as np
import pytest

from analytics.data_access import ContestResponses, build_response_frame, load_problems
from models.ability_model import fit_ability_profile
from models.blank_model import fit_blank_model
from scoring.amc_scoring import MAX_SCORE, MIN_SCORE, score_from_responses
from scoring.predictor import predict_2026
from simulation.monte_carlo import build_position_pools, run_simulation, simulate_known_contest


def _perfect_responses(conn, contest_id):
    rows = conn.execute(
        "SELECT position, answer FROM problems WHERE contest_id=? ORDER BY position", (contest_id,)
    ).fetchall()
    return [r["answer"] for r in rows]


def test_load_problems_has_every_contest(conn):
    problems = load_problems(conn)
    assert problems["contest_id"].nunique() == 51
    assert len(problems) == 51 * 25
    assert problems["categories"].apply(len).min() >= 1


def test_run_simulation_score_bounds(conn):
    problems = load_problems(conn)
    pools = build_position_pools(problems)
    profile = fit_ability_profile(build_response_frame(problems, [ContestResponses("2021A", _perfect_responses(conn, "2021A"))]))
    blank_model = fit_blank_model(build_response_frame(problems, [ContestResponses("2021A", _perfect_responses(conn, "2021A"))]), profile)
    rng = np.random.default_rng(0)
    result = run_simulation(pools, profile, blank_model, n_simulations=500, rng=rng)
    assert result.scores.min() >= MIN_SCORE
    assert result.scores.max() <= MAX_SCORE
    assert len(result.scores) == 500


def test_perfect_scorer_predicts_high_score(conn):
    responses = _perfect_responses(conn, "2021A")
    result = predict_2026(conn, [ContestResponses("2021A", responses)], n_simulations=3000, seed=1)
    assert result.input_scores["2021A"] == MAX_SCORE
    assert result.simulation_summary["median_score"] > 120  # should predict a strong future score


def test_all_blank_scorer_predicts_low_score(conn):
    responses = [None] * 25
    result = predict_2026(conn, [ContestResponses("2021A", responses)], n_simulations=3000, seed=1)
    assert result.input_scores["2021A"] == 37.5
    assert result.simulation_summary["median_score"] < 60


def test_prediction_result_has_all_expected_fields(conn):
    responses = _perfect_responses(conn, "2021A")
    result = predict_2026(conn, [ContestResponses("2021A", responses)], n_simulations=1000, seed=2)
    assert len(result.question_level) == 25
    assert result.explanation
    assert "histogram" not in result.explanation  # sanity: not accidentally dumping raw data into text
    assert sum(h["probability"] for h in result.histogram) == pytest.approx(1.0, abs=1e-6)


def test_simulate_known_contest_matches_actual_when_ability_far_exceeds_difficulty(conn):
    problems = load_problems(conn)
    contest_problems = problems[problems.contest_id == "2021A"]
    from models.ability_model import AbilityProfile
    from models.blank_model import BlankModel

    profile = AbilityProfile(theta_global=3000.0)  # absurdly high ability
    blank_model = BlankModel(intercept=-10, slope=0, n_fit=0, fitted_from_data=False)
    rng = np.random.default_rng(0)
    sim = simulate_known_contest(contest_problems, profile, blank_model, n_simulations=200, rng=rng)
    # Ability so far above every item's difficulty means near-certain full marks.
    assert sim.scores.mean() > 145
