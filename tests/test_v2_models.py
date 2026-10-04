import numpy as np
import pytest

from models.bayes_glm import fit_map, predict_proba
from models.ensemble import fit_ensemble, stack_weights
from models.members import ELO_K, IrtMember, Items, pava_decreasing
from simulation.contest_model import N_CATEGORIES


def _items(rng, n, theta, n_cat=3, offsets=None, attempts=1):
    difficulty = rng.uniform(700, 2300, n)
    cat = np.zeros((n, N_CATEGORIES))
    which = rng.integers(0, n_cat, n)
    cat[np.arange(n), which] = 1.0
    off = np.zeros(N_CATEGORIES) if offsets is None else offsets
    p = 1 / (1 + np.exp(-ELO_K * (theta + cat @ off - difficulty)))
    y = (rng.random(n) < p).astype(float)
    return Items(difficulty, cat, rng.integers(1, 26, n).astype(float), rng.integers(0, attempts, n), np.zeros(n), y)


def test_glm_recovers_logistic_coefficients():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(4000), rng.normal(size=4000)])
    y = (rng.random(4000) < predict_proba(X, np.zeros(4000), np.array([0.5, -1.2]))).astype(float)
    fit = fit_map(X, np.zeros(4000), y, np.zeros(2), np.full(2, 100.0))
    assert fit.converged
    assert fit.beta == pytest.approx([0.5, -1.2], abs=0.12)
    assert np.all(np.linalg.eigvalsh(fit.cov) > 0)


def test_glm_prior_dominates_without_data():
    X = np.ones((1, 1))
    fit = fit_map(X, np.zeros(1), np.array([1.0]), np.array([3.0]), np.array([1e-6]))
    assert fit.beta[0] == pytest.approx(3.0, abs=1e-3)


def test_irt_recovers_ability_and_reports_uncertainty():
    rng = np.random.default_rng(1)
    m = IrtMember("irt_global", use_categories=False)
    m.fit(_items(rng, 600, theta=1650.0))
    s = m.ability_summary()
    assert abs(s["global_mean"] - 1650) < 3 * s["global_sd"]
    assert 5 < s["global_sd"] < 60


def test_irt_uncertainty_shrinks_with_more_data():
    rng = np.random.default_rng(2)
    small, large = IrtMember("a", use_categories=False), IrtMember("b", use_categories=False)
    small.fit(_items(rng, 25, 1500.0))
    large.fit(_items(rng, 500, 1500.0))
    assert large.ability_summary()["global_sd"] < small.ability_summary()["global_sd"]


def test_irt_all_correct_stays_finite():
    n = 25
    items = Items(np.linspace(500, 2500, n), np.zeros((n, N_CATEGORIES)), np.arange(1, n + 1.0),
                  np.zeros(n, dtype=int), np.zeros(n), np.ones(n))
    m = IrtMember("irt_global", use_categories=False)
    m.fit(items)
    assert 2000 < m.ability_summary()["global_mean"] < 4000


def test_hierarchical_irt_orders_strong_and_weak_topics():
    rng = np.random.default_rng(3)
    offsets = np.zeros(N_CATEGORIES); offsets[0], offsets[1] = 300.0, -300.0
    m = IrtMember("irt_hier")
    m.fit(_items(rng, 900, 1500.0, n_cat=3, offsets=offsets))
    cats = m.ability_summary()["categories"]
    assert cats[0][0] > cats[2][0] > cats[1][0]
    # A topic never seen keeps the overall estimate.
    assert cats[10][0] == pytest.approx(m.ability_summary()["global_mean"], abs=1.0)


def test_guess_floor_dampens_a_lucky_hit():
    n = 20
    difficulty = np.concatenate([np.linspace(800, 1500, n - 1), [2600.0]])
    y = np.concatenate([(np.linspace(800, 1500, n - 1) < 1250).astype(float), [1.0]])   # one lucky hit at 2600
    items = Items(difficulty, np.zeros((n, N_CATEGORIES)), np.arange(1, n + 1.0), np.zeros(n, dtype=int), np.zeros(n), y)
    plain, guess = IrtMember("p", use_categories=False), IrtMember("g", guess_floor=0.2, use_categories=False)
    plain.fit(items); guess.fit(items)
    assert guess.ability_summary()["global_mean"] < plain.ability_summary()["global_mean"]


def test_pava_is_monotone_and_preserves_mean():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, 200); y = (rng.random(200) < 1 - x).astype(float)
    xs, fitted = pava_decreasing(x, y)
    assert np.all(np.diff(fitted) <= 1e-12)
    assert fitted.mean() == pytest.approx(y.mean())


def test_stack_weights_prefers_the_better_model():
    rng = np.random.default_rng(5)
    p_true = rng.uniform(0.05, 0.95, 400)
    y = (rng.random(400) < p_true).astype(float)
    oof = np.column_stack([p_true, np.full(400, 0.5)])
    w = stack_weights(oof, y, np.array([0.5, 0.5]))
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > 0.8


def test_stack_weights_stay_near_prior_with_little_data():
    oof = np.array([[0.7, 0.4], [0.6, 0.5]]); y = np.array([1.0, 0.0])
    w = stack_weights(oof, y, np.array([0.8, 0.2]))
    assert w == pytest.approx([0.8, 0.2], abs=0.05)


def test_ensemble_beats_base_rate_on_held_out_answers():
    rng = np.random.default_rng(6)
    ens = fit_ensemble(_items(rng, 75, 1500.0, attempts=3), rng)
    r = ens.cv_report
    assert r["stacked"] and "gbt" in ens.names
    assert ens.weights.sum() == pytest.approx(1.0)
    assert r["ensemble"]["log_loss"] < r["base_rate"]["log_loss"]


def test_ensemble_falls_back_to_prior_weights_when_too_small():
    rng = np.random.default_rng(7)
    ens = fit_ensemble(_items(rng, 8, 1500.0), rng)
    assert ens.cv_report["stacked"] is False
    assert "gbt" not in ens.names
