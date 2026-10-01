# Methodology

How a set of answer sheets becomes a 2026 score forecast, with every
assumption stated. Section 9 lists what was removed or corrected from the
first version.

## 1. Data

Source: `https://live.poshenloh.com/past-contests/amc10/<id>`, one page per
paper. Each page embeds JSON with 25 problems in order: answer letter,
concept tags, a difficulty rating (`seedElo`), and an expected solve time.
All 51 papers from 2001 to 2025 are ingested: one paper in 2001, A and B
from 2002, and four in 2021 (LIVE labels the autumn 2021 pair C and D).

Only the answer letter, rating, solve time and concept tags are kept. The
fetch step checks that each page had a statement and five choices, then
discards that text (`ingestion/pipeline.py: strip_text`).

`scripts/validate_data.py` fails unless every paper has positions 1–25,
valid answer letters, positive ratings, and at least one concept tag that
maps to a topic category.

**Ratings.** LIVE describes its ratings as Elo-style and updated from
solve attempts on its platform. The field exposed on the page is named
`seedElo`. This project cannot tell how much real-attempt calibration it
contains, or whether papers from different years were rated on a common
footing. Section 8 explains why that matters.

**Topics.** About 260 concept tags are mapped by hand to 15 categories
(`ingestion/concepts.py`). A problem tagged with concepts from several
categories counts toward each, with equal weight.

## 2. Scoring

Current rules: 6 per right answer, 1.5 per blank, 0 per wrong answer.
The blank value was 2 in 2000, 2.5 from 2001 to 2006, and 1.5 from 2007
(Wikipedia, "American Mathematics Competitions"; the AoPS wiki dates the
change to 2.5 a year later). Forecasts always use current rules. A sitting
on an older paper is shown under both.

## 3. Forecasting the paper

Mean rating per year rose from about 1320 (2001) to about 1640 (2025), and
the last five problems from about 1800 to about 2180. A simulated 2026
paper is

    difficulty[slot] = level + residual[slot]

- `level`: the paper's mean rating. A weighted linear regression of past
  contest means on year (weights halve every 8 years back) is extrapolated
  to 2026, giving a mean and a standard deviation that includes both
  contest-to-contest scatter and uncertainty in the fitted line. For 2026:
  1596 ± 76.
- `residual[slot]`: drawn from that slot's historical (rating − own
  contest mean) values, with weights that halve every 3 years back. The
  drawn problem's topic mix comes with it.

The shared `level` is what makes a simulated paper hard or easy as a
whole. Without it, slot-by-slot resampling averages out and understates
how much papers differ.

**Check against history** (`scripts/backtest_contest_model.py`). For each
of the 30 papers from 2012 on, the forecast was built from earlier years
only. For reference students at six ability levels who answer everything,
the forecast expected score was compared with the expected score on the
real paper:

| Variant | Bias | MAE | RMSE | Real paper in 90% band |
|---|---|---|---|---|
| all years equal, no level (v1) | +8.4 | 10.4 | 12.5 | 37% |
| 3-year pools, no level | +4.9 | 7.4 | 9.2 | 53% |
| 3-year pools + weighted mean level | +4.8 | 7.3 | 9.2 | 82% |
| 3-year pools + trend level (8-year weights) | +2.2 | 6.6 | 8.2 | 85% |

The half-lives were chosen from this table, so the last row is a little
optimistic about itself. The 180 comparisons are 30 papers times 6
abilities and are not independent.

## 4. Chance of a right answer, given an attempt

Blanks are handled separately (section 5). The models here are fitted to
answered problems only.

### 4.1 Bayesian rating models

The response curve is the Elo curve with optional guessing floor `c` and
discrimination `a`:

    P(right) = c + (1 − c) · sigmoid(a · ln10/400 · (θ − difficulty))

and the student's level on a problem is

    θ = overall + growth · (time of sitting − contest date)
        + mean over the problem's categories of topic offsets
        + sitting effect

Priors: overall ~ N(1400, 500²); growth ~ N(0, 150²) per year; topic
offsets ~ N(0, τ²); sitting effects ~ N(0, σ_day²). This is a logistic
regression with Gaussian priors and an offset, fitted by Fisher scoring to
the posterior mode, with a Laplace (Gaussian) approximation around it
(`models/bayes_glm.py`).

- τ ∈ {40, 100, 160, 240} and, with three or more sittings,
  σ_day ∈ {20, 60, 100} are chosen by the Laplace approximation to the
  marginal likelihood. With fewer sittings σ_day is fixed at 60, an
  assumption.
- Growth is fitted only when every sitting has a date and they span at
  least 30 days. The "date" is when the student sat the paper, not the
  year it was set.
- A topic with no answered problems keeps offset 0 with its prior
  uncertainty, so it is simulated as "unknown", not as "average".
- All right or all wrong answers give a finite estimate because of the
  prior (v1 added an arbitrary 800 points).

Four variants are fitted: `irt_hier` (c = 0, a = 1, topics), `irt_guess`
(c = 0.2), `irt_flat` (a = 0.6), `irt_global` (no topics).

### 4.2 Other models

- `logit_feat`: logistic regression on rating, position in the paper and
  topics, with its own slope on rating (prior centred on the Elo slope).
- `isotonic`: pool-adjacent-violators fit of accuracy as a non-increasing
  function of rating, blended 75/25 with `irt_global`; uncertainty from 30
  bootstrap refits.
- `gbt`: gradient-boosted trees (40 trees, depth 2) on rating, position
  and topics; only with at least 60 answered problems; uncertainty from
  bootstrap refits.

### 4.3 Stacking

Answered problems are split into folds: whole sittings when there are four
or more, otherwise five random folds. Every model is refitted without each
fold and predicts it. Weights on the simplex are chosen to maximise the
likelihood of the real outcomes under the weighted average of the models
(EM), with a Dirichlet prior worth 20 observations centred on
35% `irt_hier`, 15% `irt_guess`, 15% `logit_feat`, 13% `irt_global`,
12% `irt_flat`, 5% each `isotonic` and `gbt`. With fewer than 15 answered
problems the prior weights are used unchanged.

The held-out predictions are also reported to the user as log loss, Brier
score and accuracy per model, beside a constant-rate baseline. The combined
row is slightly flattering because the weights were tuned on the same
held-out predictions.

## 5. Answer or blank

    logit P(answers) = b0 + b1 · (difficulty − θ)/400 + b2 · (position − 13)/12

fitted on all 25 problems of every sitting, Bayesian with priors
b0 ~ N(2, 1.5²), b1 ~ N(−1.5, 1.5²), b2 ~ N(−0.5, 1²). The priors describe
a generic student and are assumptions; they fade as blanks accumulate.

v1 drew "right" for all 25 problems first and then split the misses into
wrong and blank. That credits the student with right answers on problems
they would have skipped.

## 6. Simulation

For each of 20,000 sittings (`simulation/engine.py`):

1. draw a paper (section 3);
2. pick a model with probability equal to its stacking weight and draw its
   parameters from its posterior or bootstrap distribution, plus a new
   sitting effect ~ N(0, σ_day²);
3. draw the blank model's parameters;
4. for each problem: answered? if so, right?
5. score 6 per right answer, 1.5 per blank.

The spread of the result therefore includes uncertainty about the paper,
uncertainty about the student, day-to-day variation and item-level luck.

What-if runs reuse the same random numbers and change only one topic's
strength (+150), so the reported gain is free of simulation noise. The
gain is in mean score, because the median moves in 1.5-point steps.

## 7. Validation on simulated students

`analytics/validation_v2.py`. Students are generated from a process that
differs from every model above: skills vary by fine-grained concept, each
student has a personal response slope, answers come from a
know/guess/blank process with a 3% slip rate and 25% guess success, time
runs out at a personal cut-off position, ability shifts between sittings
and grows over time.

For each of 16 papers from 2019–2025 and 12 students per paper, the
pipeline receives K sittings on earlier papers, forecasts the target year
from earlier papers only, and is compared with the student's simulated
score on the real target paper.

| K | Pipeline | Bias | MAE | RMSE | CRPS | 90% cover | 50% cover | 90% width |
|---|---|---|---|---|---|---|---|---|
| 1 | v1 | +26.1 | 28.7 | 35.6 | 24.4 | 32% | 15% | 28 |
| 1 | v2, main rating model only | −4.6 | 12.6 | 15.9 | 8.9 | 88% | 44% | 45 |
| 1 | v2, old paper forecast | +1.3 | 11.9 | 15.1 | 8.4 | 89% | 52% | 46 |
| 1 | v2 | −4.2 | 12.3 | 15.5 | 8.7 | 90% | 45% | 46 |
| 3 | v1 | +27.9 | 28.2 | 32.1 | 23.7 | 22% | 8% | 28 |
| 3 | v2 | −2.2 | 10.2 | 12.9 | 7.3 | 88% | 52% | 39 |
| 6 | v1 | +27.5 | 27.8 | 31.5 | 23.4 | 22% | 7% | 28 |
| 6 | v2 | −2.0 | 9.7 | 12.4 | 7.0 | 85% | 55% | 37 |

Reading it:

- v1's error is mostly bias from crediting skipped problems and from the
  too-easy paper forecast, in a world where students skip a lot. In the
  first version's own validation, where simulated students followed v1's
  assumptions exactly, v1 looked accurate (MAE about 3). That test could
  not have failed and has been removed.
- The ensemble beats the main rating model alone by 0.1 to 0.3 points of
  MAE. The extra models are a small refinement.
- In this simulated world the new paper forecast is no better than the
  old one, and slightly more negative in bias. Its support is the
  real-history check in section 3, where it is clearly better.
- v2 runs about 2 to 4 points low. One cause is that the growth prior is
  centred on zero while these students improve on average.

None of this measures error on real students.

## 8. Limits

- **Rating comparability across years.** The forecast for a student who
  only enters old papers depends on LIVE's ratings of those papers being
  on the same footing as its ratings of recent ones. A raw 93 on 2021A
  (mean rating 1364) forecasts about 78; a 93 on 2025A (1630) forecasts
  about 91. If recent papers are over-rated, the first number is too low.
  Sitting a recent paper removes most of the dependence.
- **No real-student validation.**
- **One sitting is thin evidence.** Expect a 90% range about 45 points
  wide.
- **Priors are assumptions**: ability centre and spread, day-to-day
  swing, growth, and the blank model's generic student.
- **Topic categories are hand-made**, and with few sittings the topic
  estimates are barely distinguishable; the app says so when that is the
  case.
- **The 2026 date** defaults to 5 November 2026, an assumption that only
  affects the growth term.

## 9. Changes from the first version

- Papers: 18 (2018–2025) to 51 (2001–2025).
- Paper forecast: equal pooling replaced by section 3.
- Blanks: modelled before correctness, not after.
- Ability: point estimate replaced by a posterior that is sampled in the
  simulation.
- Time: weighting by the year a paper was set replaced by the date the
  student sat it, with an optional growth term.
- Models: one replaced by a stacked ensemble of seven.
- Scoring note corrected: v1's code comment said blanks had been worth 1.5
  since 2001; they were worth 2.5 until 2006.
- Validation: the self-consistent synthetic backtest and the four-model
  comparison built on it were removed in favour of sections 3 and 7.
- Stored data no longer includes problem statements or solutions.
