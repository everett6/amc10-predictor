import numpy as np
import pandas as pd

from models.ability_model import fit_ability_profile, mle_ability, p_correct, recency_weights


def test_p_correct_equal_ability_and_difficulty_is_half():
    assert abs(p_correct(1500, 1500) - 0.5) < 1e-9


def test_p_correct_higher_ability_is_more_likely():
    assert p_correct(1800, 1500) > p_correct(1500, 1500) > p_correct(1200, 1500)


def test_mle_ability_recovers_known_ability_with_many_items():
    rng = np.random.default_rng(0)
    true_theta = 1600.0
    difficulties = rng.uniform(1000, 2000, size=2000)
    p = 1.0 / (1.0 + 10.0 ** ((difficulties - true_theta) / 400.0))
    corrects = (rng.random(2000) < p).astype(float)
    est = mle_ability(difficulties, corrects)
    assert abs(est - true_theta) < 60  # should be close with this much data


def test_mle_ability_all_correct_returns_finite_high_estimate():
    est = mle_ability(np.array([1000.0, 1200.0]), np.array([1.0, 1.0]))
    assert np.isfinite(est)
    assert est > 1200


def test_mle_ability_all_wrong_returns_finite_low_estimate():
    est = mle_ability(np.array([1000.0, 1200.0]), np.array([0.0, 0.0]))
    assert np.isfinite(est)
    assert est < 1000


def test_mle_ability_empty_returns_zero():
    assert mle_ability(np.array([]), np.array([])) == 0.0


def test_recency_weights_most_recent_is_one_and_monotonic():
    ranks = pd.Series([(2018, 0), (2018, 0), (2020, 0), (2022, 0)])
    weights = recency_weights(ranks, half_life_contests=2.0)
    assert weights.iloc[3] == 1.0  # most recent (2022)
    assert weights.iloc[0] == weights.iloc[1]  # same contest, same weight
    assert weights.iloc[0] < weights.iloc[2] < weights.iloc[3]


def _fake_response_df(n_per_category=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    categories = ["Algebra: Equations & Polynomials", "Geometry: Circles & Curves"]
    for i, cat in enumerate(categories):
        true_theta = 1400 + i * 400
        for j in range(n_per_category):
            elo = rng.uniform(1000, 2000)
            p = 1.0 / (1.0 + 10.0 ** ((elo - true_theta) / 400.0))
            outcome = "correct" if rng.random() < p else "wrong"
            rows.append(
                {
                    "contest_id": "2021A", "position": j + 1, "seed_elo": elo,
                    "categories": [cat], "outcome": outcome,
                    "recency_rank": (2021, 0),
                }
            )
    return pd.DataFrame(rows)


def test_fit_ability_profile_separates_strong_and_weak_categories():
    df = _fake_response_df(n_per_category=40, seed=1)
    profile = fit_ability_profile(df, shrinkage_k=2.0)
    weak = profile.categories["Algebra: Equations & Polynomials"].theta_smoothed
    strong = profile.categories["Geometry: Circles & Curves"].theta_smoothed
    assert strong > weak


def test_fit_ability_profile_shrinks_small_sample_toward_global():
    rng = np.random.default_rng(2)
    rows = []
    # Lots of data in one category around a moderate ability level
    # (mixed outcomes, so theta_global isn't degenerate).
    true_theta_a = 1300.0
    for _ in range(100):
        elo = rng.uniform(1000, 1600)
        p = 1.0 / (1.0 + 10.0 ** ((elo - true_theta_a) / 400.0))
        outcome = "correct" if rng.random() < p else "wrong"
        rows.append({"categories": ["Cat A"], "seed_elo": elo, "outcome": outcome, "recency_rank": (2021, 0)})
    # A single lucky guess in a rarely-seen, much harder category.
    rows.append({"categories": ["Cat B"], "seed_elo": 2200.0, "outcome": "correct", "recency_rank": (2021, 0)})
    df = pd.DataFrame(rows)
    df["position"] = range(len(df))
    df["contest_id"] = "2021A"
    profile = fit_ability_profile(df, shrinkage_k=6.0)
    # Cat B's raw MLE would be extremely high (single correct at 2200);
    # smoothing should pull it much closer to theta_global than that.
    cat_b = profile.categories["Cat B"]
    assert cat_b.theta_smoothed < cat_b.theta_mle - 100
    assert abs(profile.theta_global - true_theta_a) < 300
