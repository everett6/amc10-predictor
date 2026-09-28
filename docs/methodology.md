# Methodology

This document describes exactly how the AMC 10 Future Score Predictor
turns a student's past-contest answers into a 2026 score prediction, and
is explicit about every simplifying assumption and known limitation.

## 1. Data

**Source**: [LIVE by Po-Shen Loh](https://live.poshenloh.com/past-contests/amc10/<year><label>),
scraped live over HTTP (respecting `robots.txt`, cached on disk, rate-limited
per host — see `src/ingestion/http_client.py`). Each contest page embeds a
Next.js `__NEXT_DATA__` JSON blob with, per problem: the 5 answer choices,
correct answer, a free-text `concepts` tag (e.g. `"ratio and proportion;
linear equation"`), a `seedElo` difficulty rating, and expected solve time.

**Coverage**: all 18 target contests (2018–2025, A/B, plus the Fall 2021
makeup contests labeled C/D) were fetched successfully — 450 problems, 25
per contest, no gaps. `scripts/validate_data.py` enforces this on every run
(25 problems/contest, non-empty fields, valid answer letters, positive
difficulty, at least one concept tag, and every concept tag mapped to a
broad category) and fails loudly rather than silently tolerating gaps.

**Topic taxonomy**: LIVE tags problems with 256 fine-grained concepts. We
map all 256 to 15 broader categories (`src/ingestion/concepts.py`), sized
so each has enough historical problems to fit a per-topic ability estimate
from a realistic amount of student data (a student who's taken 3-5 past
contests sees maybe 5-15 problems per broad category, but often 0-2 per
fine-grained concept). The mapping is a manual, documented editorial choice
(e.g. splitting geometry's 86 concepts into triangles/polygons, circles,
coordinate/transformations, and solid geometry) — it is not learned from
data, and reasonable people could bucket differently.

## 2. Scoring

Official AMC 10 rules (unchanged since 2001, so no year-branching needed
for 2018-2025): +6 per correct, +1.5 per blank, +0 per wrong, out of 25
questions (max 150, all-blank floor 37.5). See `src/scoring/amc_scoring.py`.

## 3. Ability model: IRT-inspired hierarchical, Bayesian-smoothed

We reuse LIVE's own crowd-calibrated `seedElo` difficulty rating as the
item-response-theory "difficulty" parameter, and estimate a **student
ability** on the same scale via the standard chess-Elo win-probability form:

```
P(correct | ability θ, difficulty b) = 1 / (1 + 10^((b - θ) / 400))
```

This is a **1-parameter (Rasch-like) IRT model, not a full 2PL/3PL fit**:
we do not estimate a per-item discrimination or guessing parameter from
data, because 25 items × up to 17 contests per student is far too little
data to fit those per-item parameters reliably — and LIVE's own
large-sample difficulty calibration is a better `b` estimate than anything
we could re-derive from one student's history anyway.

**Hierarchical smoothing.** We fit:
- `theta_global`: one MLE ability over *all* the student's attempted (non-blank)
  problems, via `scipy.optimize.brentq` on the log-likelihood gradient
  (closed-form-adjacent since the log-likelihood is concave in θ for fixed
  difficulties).
- `theta_category` per broad category: its own MLE, then **shrunk toward
  theta_global** by a precision-weighted average:

  ```
  theta_smoothed = (n_eff * theta_category_mle + k * theta_global) / (n_eff + k)
  ```

  where `n_eff` is the (recency-weighted) number of attempts in that
  category and `k` (default 6) is an effective prior sample size — this is
  algebraically equivalent to a `Normal(theta_global, τ²)` empirical-Bayes
  prior on each category, without needing to numerically integrate a
  posterior. A category with 1 lucky guess at a very hard difficulty gets
  pulled hard toward the global estimate; a category with 40 attempts is
  dominated by its own data. See `tests/test_ability_model.py` for a
  concrete before/after example.

**Recency weighting.** When several past contests are supplied, each
attempt's contribution to every MLE above is weighted by
`exp(-ln(2)/H * steps_back)`, where `H` (default 4 contests) is a half-life
in *contest count* (not calendar time) and `steps_back` counts distinct
contest "slots" back from the most recent one supplied. This lets the
model track real improvement/decline instead of averaging a 2018 contest
and a 2025 contest as equally informative about *current* ability.

**Blanks are excluded from ability estimation** (standard IRT practice —
a blank tells us about time management/risk tolerance, not necessarily
topic mastery), but are explicitly modeled for scoring (next section).

## 4. Blank-vs-guess model

Two students with identical ability can score very differently if one
guesses on hopeless problems (expected value of a random 5-choice guess is
6/5 = 1.2 points, *worse* than the 1.5-point blank bonus) while the other
leaves them blank. We fit a per-student logistic model,
`P(blank | not correct) = sigmoid(a + b * (difficulty - ability))`, from
the student's own historical wrong/blank split via a small hand-rolled
Newton's-method logistic regression (`src/models/blank_model.py`) —
avoiding a full sklearn dependency for a well-conditioned 1-D fit. If a
student has fewer than 3 blanks or 3 wrongs in their history (too little
signal), we fall back to a generic default slope/intercept.

## 5. Monte Carlo simulation of 2026

A 2026 AMC 10 doesn't exist yet, so we can't know its exact 25 problems.
Instead, for each of the 25 problem **positions**, we treat all 18
historical contests as an empirical sample of "what does a problem in this
slot tend to look like" (difficulty + topic mix), and **bootstrap-sample**
from that pool to build each of 20,000+ synthetic 2026 contests
(`src/simulation/monte_carlo.py`). For each simulated problem we:

1. sample a `(difficulty, categories)` pair from that position's 18-contest
   historical pool,
2. compute the student's ability for that problem as the mean of their
   `theta_smoothed` across its categories,
3. compute `P(correct)` via the Elo formula and draw an outcome,
4. if incorrect, use the blank model to decide blank vs. wrong,
5. accumulate the official AMC 10 score.

Repeating this thousands of times gives the full predictive distribution
(median, mean, percentiles, expected correct/wrong/blank, and a histogram)
rather than a single point estimate. `simulate_known_contest` does the same
thing but against a specific *known* set of 25 problems (used for
calibration/backtesting against a held-out historical contest, where we
know the ground truth).

## 6. Counterfactuals

For each of the student's weakest attempted topics, we re-run the full
simulation with *only that topic's* ability bumped by +150 Elo (all else
held fixed) and report the resulting change in **mean** score (not
median — the median moves in 6-point/1.5-point-quantized steps and is
often flat under small ability changes, which would misleadingly show
"0 gain"; mean score responds smoothly).

## 7. Validation: synthetic-student backtesting

No real longitudinal AMC/LIVE response-log dataset is publicly available,
so real-world accuracy claims aren't possible here. Instead,
`src/analytics/calibration.py` generates **synthetic students** with known
ground-truth per-category ability, simulates their responses to every real
historical problem using the same Elo item-response function (plus
independent per-attempt noise, representing real-world messiness like
careless errors), and then:

- **Rolling-window backtest**: for each student, train on contests
  `[0..i)` (chronological), predict contest `i`, for every `i`.
- **Leave-one-test-out backtest**: train on all contests except one held
  out, predict the held-out one.

We report MAE/RMSE of the predicted median score vs. the synthetic
student's actual simulated score, plus 90%-interval coverage and
calibration error (`|coverage - 0.90|`). Results (25 synthetic students):

| Backtest | MAE | RMSE | Coverage | Calibration error |
|---|---|---|---|---|
| Rolling window | 3.88 | 6.69 | 83.3% | 6.7% |
| Leave-one-test-out | 3.16 | 4.95 | 89.1% | 0.9% |

This validates the **pipeline's internal consistency** (can it recover a
score close to a student's true simulated performance, and are its
uncertainty intervals honest) — it is *not* a claim about accuracy on real
AMC test-takers, whose actual response patterns (partial knowledge,
time pressure, problem-order effects, real careless-error correlations)
are messier than our synthetic generative model. This limitation is
called out again in every JSON report `run_backtest.py` produces.

## 8. Model comparison

`scripts/compare_models.py` compares four models' held-out per-problem
correctness predictions across the same synthetic-student rolling-window
splits: a global-rate baseline, plain logistic regression (features:
difficulty, position, one-hot topic), gradient-boosted trees (same
features), and our IRT-hierarchical model. Results:

| Model | Mean log-loss | Mean Brier | Mean accuracy |
|---|---|---|---|
| Baseline | 0.430 | 0.101 | 89.2% |
| Logistic regression | 0.363 | 0.084 | 87.4% |
| Gradient-boosted trees | 0.543 | 0.108 | 87.5% |
| IRT-hierarchical | **0.257** | **0.081** | 89.1% |

IRT-hierarchical wins on log-loss and Brier score (the metrics that matter
for a *simulation*, which needs well-calibrated probabilities, not just a
correct/incorrect classification) while matching the baseline's accuracy.
The generic ML models (logistic regression, GBT) have too many free
parameters (16-17 one-hot topic dummies) relative to the ~25×N data points
available per student and overfit the training split.

## Known limitations

- Synthetic-student validation, not real-student validation (see §7).
- The 15-category topic taxonomy is a manual editorial choice, not learned.
- The blank-model's default fallback (used when a student has too little
  blank/wrong history) is a generic hand-picked curve, not
  population-calibrated (no real population data to calibrate it against).
- Bootstrap position pools assume 2026's difficulty-by-position and
  topic-by-position distribution will resemble 2018-2025's; a genuine
  format change would break this assumption.
- Ability is estimated per broad category, not per fine-grained concept or
  per individual skill — a student strong in "circles" but weak in
  "3D geometry" both count toward one "Geometry" ability if they land in
  the same broad bucket in our mapping (they don't here — those are
  separate categories — but the general point holds for any bucketing).
