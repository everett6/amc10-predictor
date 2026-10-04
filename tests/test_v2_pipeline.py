import datetime

import numpy as np
import pytest

from analytics.data_access import ContestResponses, load_problems
from scoring.amc_scoring import blank_points_for_year
from scoring.predictor_v2 import fit_student, predict_v2, run_simulation
from simulation.contest_model import build_contest_forecast
from simulation.engine import FixedContest


@pytest.fixture(scope="module")
def problems():
    from pathlib import Path

    from storage import schema
    db = Path(__file__).resolve().parent.parent / "data" / "amc10.sqlite3"
    if not db.exists():
        pytest.skip("run scripts/ingest_amc.py first")
    conn = schema.connect(db)
    try:
        return load_problems(conn)
    finally:
        conn.close()


def _responses(problems, contest_id, n_right, n_wrong):
    key = list(problems[problems.contest_id == contest_id].sort_values("position").answer)
    out = []
    for i, k in enumerate(key):
        if i < n_right:
            out.append(k)
        elif i < n_right + n_wrong:
            out.append(next(l for l in "abcde" if l != k))
        else:
            out.append(None)
    return out


def test_dataset_covers_2001_to_2025(problems):
    assert problems.contest_id.nunique() == 51
    assert len(problems) == 51 * 25
    assert problems.year.min() == 2001 and problems.year.max() == 2025
    assert problems.categories.apply(len).min() >= 1


def test_blank_points_by_year():
    assert blank_points_for_year(2000) == 2.0
    assert blank_points_for_year(2001) == 2.5
    assert blank_points_for_year(2006) == 2.5
    assert blank_points_for_year(2007) == 1.5
    assert blank_points_for_year(2025) == 1.5


def test_forecast_weights_recent_years_and_tracks_the_trend(problems):
    recent = build_contest_forecast(problems, 2026)
    flat = build_contest_forecast(problems, 2026, half_life_years=None, level="none")
    rng = np.random.default_rng(0)
    d_recent, idx = recent.sample(4000, rng)
    d_flat, _ = flat.sample(4000, rng)
    assert d_recent.mean() > d_flat.mean() + 100          # difficulty has risen since 2001
    assert d_recent.shape == (4000, 25) and recent.category_matrix(idx).shape == (4000, 25, 15)
    # contests come out hard or easy as a whole: contest means vary more than independent slots would
    assert d_recent.mean(axis=1).std() > d_flat.mean(axis=1).std()
    assert 1450 < recent.level_mean < 1800 and recent.level_sd > 20


def test_forecast_max_year_excludes_the_future(problems):
    f = build_contest_forecast(problems, 2020, max_year=2019)
    assert sum(len(p.seed_elos) for p in f.pools.values()) == len(problems[problems.year <= 2019])


def test_predict_v2_shape_and_bounds(problems):
    r = predict_v2(problems, [ContestResponses("2021A", _responses(problems, "2021A", 14, 5))], n_simulations=3000, seed=1)
    s = r["simulation_summary"]
    assert 0 <= s["percentiles"]["p5"] <= s["median_score"] <= s["percentiles"]["p95"] <= 150
    assert s["expected_correct"] + s["expected_wrong"] + s["expected_blank"] == pytest.approx(25)
    assert sum(h["probability"] for h in r["histogram"]) == pytest.approx(1.0)
    assert len(r["question_level"]) == 25 and len(r["topics"]) == 15
    assert sum(r["model_weights"].values()) == pytest.approx(1.0)
    assert r["inputs"][0]["score_current_rules"] == 14 * 6 + 6 * 1.5
    probs = [t["probability"] for t in r["target_probabilities"]]
    assert probs == sorted(probs, reverse=True)
    # early problems are likelier than late ones
    assert r["question_level"][0]["expected_p_correct"] > r["question_level"][24]["expected_p_correct"]


def test_predict_v2_is_reproducible_with_a_seed(problems):
    sit = [ContestResponses("2022A", _responses(problems, "2022A", 12, 6))]
    a = predict_v2(problems, sit, n_simulations=2000, seed=5)["simulation_summary"]
    b = predict_v2(problems, sit, n_simulations=2000, seed=5)["simulation_summary"]
    assert a == b


def test_stronger_student_gets_a_higher_forecast(problems):
    weak = predict_v2(problems, [ContestResponses("2023A", _responses(problems, "2023A", 8, 6))], n_simulations=3000, seed=2)
    strong = predict_v2(problems, [ContestResponses("2023A", _responses(problems, "2023A", 20, 3))], n_simulations=3000, seed=2)
    assert strong["simulation_summary"]["median_score"] > weak["simulation_summary"]["median_score"] + 20
    assert strong["ability_global"] > weak["ability_global"]


def test_more_sittings_narrow_the_ability_estimate(problems):
    one = predict_v2(problems, [ContestResponses("2022A", _responses(problems, "2022A", 13, 6))], n_simulations=1000, seed=3)
    four = predict_v2(problems, [ContestResponses(c, _responses(problems, c, 13, 6)) for c in ("2022A", "2022B", "2023A", "2023B")],
                      n_simulations=1000, seed=3)
    assert four["ability_global_sd"] < one["ability_global_sd"]


def test_all_blank_sheet_predicts_near_the_blank_floor(problems):
    r = predict_v2(problems, [ContestResponses("2021A", [None] * 25)], n_simulations=2000, seed=4)
    assert r["inputs"][0]["score_current_rules"] == 37.5
    assert r["simulation_summary"]["median_score"] < 60


def test_old_paper_reports_both_scoring_rules(problems):
    r = predict_v2(problems, [ContestResponses("2004A", _responses(problems, "2004A", 12, 3))], n_simulations=1000, seed=1)
    i = r["inputs"][0]
    assert i["score_current_rules"] == 12 * 6 + 10 * 1.5
    assert i["score_rules_of_the_year"] == 12 * 6 + 10 * 2.5


def test_growth_is_fitted_only_with_dated_sittings(problems):
    D = datetime.date
    undated = [ContestResponses(c, _responses(problems, c, n, 5)) for c, n in (("2019A", 9), ("2022A", 12), ("2024A", 15))]
    dated = [ContestResponses(c.contest_id, c.responses, d) for c, d in zip(undated, (D(2025, 10, 1), D(2026, 3, 1), D(2026, 9, 1)))]
    assert predict_v2(problems, undated, n_simulations=1000, seed=1)["growth_per_year"] is None
    assert predict_v2(problems, dated, n_simulations=1000, seed=1)["growth_per_year"] is not None


def test_what_if_never_lowers_the_score(problems):
    r = predict_v2(problems, [ContestResponses("2021A", _responses(problems, "2021A", 14, 5))], n_simulations=4000, seed=1)
    assert r["counterfactuals"]
    assert all(c["score_gain"] >= 0 for c in r["counterfactuals"])      # shared random numbers: no noise


def test_simulating_a_known_paper_brackets_the_input_score(problems):
    sit = [ContestResponses("2021A", _responses(problems, "2021A", 14, 5))]
    rng = np.random.default_rng(0)
    student = fit_student(problems, sit, rng=rng)
    paper = FixedContest.from_problems(problems[problems.contest_id == "2021A"])
    sim, _ = run_simulation(student, paper, 4000, rng)
    lo, hi = np.percentile(sim.scores, [5, 95])
    assert lo <= 93.0 <= hi
