# AMC 10 Future Score Predictor

Predicts a student's likely **2026 AMC 10 score** from their answers to
past AMC 10 contests: a per-topic ability profile, a Monte Carlo-simulated
score distribution (median, mean, percentiles, expected correct/wrong/blank),
a plain-language explanation, "what if you improved topic X" counterfactuals,
and question-level detail for a typical future contest.

A live React dashboard lets you pick a past contest, fill in your answers,
and see the full prediction. See [`docs/methodology.md`](docs/methodology.md)
for how the model actually works, including its assumptions and limitations.

## Data source

All problem data (statements, answer choices, correct answers, topic tags,
and a crowd-calibrated Elo-style difficulty rating) is scraped from
[LIVE by Po-Shen Loh](https://live.poshenloh.com), which respects
`robots.txt` (`Allow: /`) and is fetched through a rate-limited, on-disk-cached
HTTP client (`src/ingestion/http_client.py`) so re-running ingestion never
re-hammers the site.

**All 18 target contests were successfully ingested live** — no fallback or
seed data was needed:

2018A, 2018B, 2019A, 2019B, 2020A, 2020B, 2021A, 2021B, 2021C (Fall), 2021D
(Fall), 2022A, 2022B, 2023A, 2023B, 2024A, 2024B, 2025A, 2025B — 450 problems
total, each with a validated answer key, 5 answer choices, an LIVE
difficulty (Elo) rating, and at least one topic tag mapped to one of 15
broad categories (see `src/ingestion/concepts.py`; all 256 of LIVE's
fine-grained concept tags are covered, verified by
`tests/test_concepts.py`).

The official MAA site (`maa.org`, `amc-reg.maa.org`) returned HTTP 403/521
in this environment and was not used; LIVE's dataset was sufficient and
is itself sourced from official AMC problems.

## Quickstart

```bash
# 1. Python environment (ingestion, modeling, API)
pip install numpy scipy pandas scikit-learn fastapi "uvicorn[standard]" pytest

# 2. Ingest data (re-fetches from live.poshenloh.com; cached after first run)
python scripts/ingest_amc.py --mode live
python scripts/validate_data.py

# 3. Run the test suite
pytest -q

# 4. Start the API
python -m uvicorn api.app:app --app-dir src --port 8000

# 5. Start the frontend (in another terminal)
cd frontend && npm install && npm run dev
```

Then open the printed Vite URL (default `http://localhost:5173`), pick a
past contest, enter your answers (or click "Fill all correct" for a demo),
and click **Predict my 2026 score**.

## Architecture

```
src/
  ingestion/    HTTP client, LIVE page parser, sanitization, concept taxonomy, pipeline
  storage/      SQLite schema + typed read/write helpers
  scoring/      Official AMC 10 scoring rules + top-level predictor orchestration
  analytics/    Topic/difficulty/position accuracy, data-access joins, calibration harness
  models/       Ability model (IRT-inspired hierarchical), blank/guess model, model comparison
  simulation/   Monte Carlo 2026 contest simulator
  api/          FastAPI backend
scripts/        ingest_amc.py, validate_data.py, run_backtest.py, compare_models.py
frontend/       React + TypeScript dashboard (Vite)
tests/          pytest suite (53 tests)
data/           SQLite DB, raw JSON snapshots, ingestion/backtest/comparison reports
docs/           methodology.md
```

## How prediction works (short version)

1. **Per-topic ability**: an Elo-style item-response model (reusing LIVE's
   own difficulty ratings) fits a separate ability rating for each of 15
   broad topic categories from the student's correct/incorrect history,
   with recency weighting across multiple past contests and hierarchical
   Bayesian shrinkage so small-sample topics don't overreact to one lucky
   or unlucky problem.
2. **Blank-vs-guess model**: a logistic model of the student's own
   historical tendency to leave a too-hard problem blank vs. guess,
   fit from their response history.
3. **Monte Carlo simulation**: 20,000+ simulated 2026 contests, each built
   by bootstrap-sampling a difficulty + topic mix for each of the 25 slots
   from all 18 historical contests, then applying the ability + blank
   models to get correct/wrong/blank outcomes and an official AMC 10 score.
4. **Counterfactuals**: re-running the simulation with one weak topic's
   ability bumped up in isolation shows its marginal score impact.

Full detail, including the exact math and every simplifying assumption, is
in [`docs/methodology.md`](docs/methodology.md).

## Validation

Since no real longitudinal AMC/LIVE response-log dataset is available, the
model is validated against **synthetic students** with known, simulated
ability profiles (`src/analytics/calibration.py` — this is clearly labeled
throughout, including in the JSON reports; it is not a claim about
real-world accuracy on real students):

| Backtest | n | MAE | RMSE | 90% interval coverage | Calibration error |
|---|---|---|---|---|---|
| Rolling window | 425 | 3.88 | 6.69 | 83.3% | 6.7% |
| Leave-one-test-out | 450 | 3.16 | 4.95 | 89.1% | 0.9% |

(`scripts/run_backtest.py`; full numbers in `data/backtest_report.json`)

Model comparison (`scripts/compare_models.py`, `data/model_comparison_report.json`),
predicting per-problem correctness on held-out contests across 20 synthetic
students:

| Model | Mean log-loss | Mean Brier score | Mean accuracy |
|---|---|---|---|
| Baseline (global rate) | 0.430 | 0.101 | 89.2% |
| Logistic regression | 0.363 | 0.084 | 87.4% |
| Gradient-boosted trees | 0.543 | 0.108 | 87.5% |
| **IRT-hierarchical (ours)** | **0.257** | **0.081** | **89.1%** |

The IRT-hierarchical model wins on log-loss and Brier score (better
*calibrated* probabilities, which is what the Monte Carlo simulation
actually needs) while matching the baseline's raw accuracy — logistic
regression and GBT overfit the one-hot topic features with this little
per-student data (25 questions x up to 17 training contests).

## Tests

```bash
pytest -q   # 53 tests: scoring rules, concept taxonomy, LIVE parsing,
            # ability model, simulation bounds, full predictor pipeline
```
