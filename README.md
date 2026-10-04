# AMC 10 Predictor

Enter the AMC 10 papers you have sat. Get the range your 2026 score is
likely to fall in, your strength by topic, and what would move the number.

It runs as a desktop app (Electron) or in a browser against a local
server. Nothing leaves your machine.

## What it does

- **Answer sheet entry** for any AMC 10 from 2001 to 2025 (51 papers):
  click bubbles, type A–E with the keyboard, or paste all 25 answers.
  Sittings are saved between sessions.
- **Score forecast**: 20,000 simulated sittings of a 2026 paper, shown as a
  distribution with the median, the 90% range, and the chance of reaching a
  target score.
- **Strength by topic** across 15 topic groups, each with an uncertainty
  range, so a topic with three answers is not presented as confidently as
  one with thirty.
- **Self-check on your own answers**: every answered problem is hidden in
  turn, predicted by models that did not see it, and compared with what you
  did. The app shows that error next to a naive baseline.
- **What-if**: the score gain from getting 150 rating points stronger in
  each of your lowest topics.

## Run it

### Desktop app

A Linux build is produced by the steps below (`desktop/dist/*.AppImage`,
about 240 MB because it carries its own Python runtime). Windows and macOS
builds use the same commands but must be run on those systems.

```bash
cd frontend && npm install && npm run build && cd ..
./scripts/build_backend.sh            # freezes the Python engine (own virtualenv)
cd desktop && npm install && npm run dist
```

To run the desktop shell from source without packaging (needs Python with
the packages in `requirements.txt`):

```bash
cd frontend && npm install && npm run build && cd ../desktop && npm install && npm start
```

### Browser

```bash
pip install -r requirements.txt
python scripts/ingest_amc.py --mode seed     # builds data/amc10.sqlite3 from the bundled snapshots
python -m uvicorn api.app:app --app-dir src --port 8000
```

Open http://localhost:8000 (the server also serves the built UI from
`frontend/dist`). For UI development, `npm run dev` in `frontend/` proxies
`/api` to port 8000.

### Tests and checks

```bash
pytest -q                               # 84 tests
python scripts/validate_data.py         # dataset completeness
python scripts/backtest_contest_model.py    # contest forecast vs real history
python scripts/run_validation.py        # old pipeline vs new on simulated students
```

## Data

Answer keys, concept tags, difficulty ratings and solve times come from
[LIVE by Po-Shen Loh](https://live.poshenloh.com/past-contests), fetched
through a cached, rate-limited client that honours `robots.txt`. All 51
AMC 10 papers from 2001 to 2025 are ingested (1,275 problems).
`python scripts/ingest_amc.py --mode live` refetches them.

Problem statements and worked solutions are **not** stored or
redistributed: they belong to the MAA and to LIVE. The app links to each
paper on LIVE instead.

LIVE's concept tags (about 260) are grouped into 15 topic categories
(`src/ingestion/concepts.py`).

## How the forecast is made

1. **The paper.** AMC 10 difficulty has risen: the mean LIVE rating was
   about 1320 in 2001 and about 1640 in 2025. The 2026 paper is simulated
   as a contest-wide difficulty level (a weighted linear trend over past
   contest means, with its own uncertainty) plus per-problem variation
   resampled from the same slot in past papers, recent years weighted most.
2. **You.** Seven models estimate the chance you get an answered problem
   right: four Bayesian rating models (with topic strengths, with a guessing
   floor, with a gentler curve, with no topics), a logistic model with a
   position effect, a free-form monotone curve, and boosted trees. They are
   combined by cross-validated stacking on your own answers.
3. **Blanks.** A separate model estimates whether you answer a problem at
   all, from how far it is above your level and how late it comes.
4. **Simulation.** Each simulated sitting draws a paper, a model, that
   model's parameters from their uncertainty, a good-day/bad-day shift,
   then answers problem by problem and scores 6 per right answer and 1.5
   per blank.

Details, formulas and assumptions: [`docs/methodology.md`](docs/methodology.md).

## How accurate is it?

There is no public dataset of real students' answer sheets across several
AMC 10 papers, so accuracy on real students has not been measured.
Two things have been:

**The paper forecast, against real history.** Each contest from 2012 on was
forecast from earlier years only and compared with the paper actually set,
in score points for reference students (`data/contest_model_backtest.json`):

| Paper forecast | Bias | Mean abs. error | Real paper inside 90% band |
|---|---|---|---|
| v1: all years pooled equally | +8.4 | 10.4 | 37% |
| v2: recent-weighted, trend, shared level | +2.2 | 6.6 | 85% |

**The whole pipeline, on simulated students** built from a process unlike
any model in the app (concept-level skills, know/guess/blank behaviour,
time running out, growth). 192 forecasts per row, each for a real paper
from 2019–2025 using only earlier papers (`data/validation_v2_report.json`):

| Sittings entered | Pipeline | Bias | Mean abs. error | Actual inside 90% range |
|---|---|---|---|---|
| 1 | v1 | +26.1 | 28.7 | 32% |
| 1 | v2 | −4.2 | 12.3 | 90% |
| 3 | v1 | +27.9 | 28.2 | 22% |
| 3 | v2 | −2.2 | 10.2 | 88% |
| 6 | v1 | +27.5 | 27.8 | 22% |
| 6 | v2 | −2.0 | 9.7 | 85% |

What this does and does not show:

- Most of the gain comes from three fixes: modelling blanks properly,
  carrying uncertainty about ability into the simulation, and the paper
  forecast. Adding the six extra models to the main rating model improved
  error by only about 0.2 points (12.6 to 12.3 with one sitting).
- A single sitting cannot pin a score down: the 90% range is about 45
  points wide with one paper and about 37 with six. A real score on one
  day moves that much.
- Simulated students are not real students. Treat the table as evidence
  that v2 is the sounder design, not as its error on you.

**Known weak point.** Everything rests on LIVE's difficulty ratings being
comparable across years. A raw 93 on the 2021A paper (rated easy) forecasts
about 78 for 2026; the same 93 on 2025A forecasts about 91. If recent
papers are rated harder than they really were, forecasts from old papers
come out low. Entering papers from the last three years avoids the issue.
The published field is called `seedElo`; how much of it reflects real
solve attempts is not something this project can see.

## Layout

```
src/ingestion/    fetch, parse, sanitise, topic taxonomy, pipeline
src/storage/      SQLite schema and helpers
src/models/       Bayesian GLM, ensemble members, stacking, blank model (+ v1 models)
src/simulation/   contest forecast, Monte Carlo engine (+ v1 simulator)
src/scoring/      scoring rules, predictor_v2 (+ v1 predictor, kept as a baseline)
src/analytics/    descriptive tables, validation harness
src/api/          FastAPI app (also the packaged backend entry point)
frontend/         React + TypeScript UI
desktop/          Electron shell and packaging config
scripts/          ingest, validate, backtests, backend build
tests/            pytest suite
```
