"""FastAPI backend for the AMC 10 Future Score Predictor.

Endpoints:
  GET  /api/contests            -- list ingested contests (for the "pick a
                                    contest to enter your answers" UI)
  GET  /api/contests/{id}       -- one contest's 25 problems (answer key
                                    included, since this is retrospective
                                    practice data, not a live exam)
  POST /api/predict             -- the main endpoint: given responses to
                                    one or more historical contests, return
                                    a full 2026 PredictionResult
  GET  /api/health              -- liveness check
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import math  # noqa: E402
from dataclasses import asdict  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from analytics.data_access import ContestResponses, load_problems  # noqa: E402
from scoring.predictor import predict_2026  # noqa: E402
from storage import schema  # noqa: E402

DB_PATH = REPO_ROOT / "data" / "amc10.sqlite3"

app = FastAPI(title="AMC 10 Future Score Predictor API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local/dev use; tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


def _nan_to_none(obj):
    """pandas turns Python None into NaN when a DataFrame column mixes
    None with floats (e.g. accuracy_attempted for a topic with zero
    non-blank attempts); NaN isn't valid strict JSON, so scrub it back
    to null recursively before returning any response."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def get_conn():
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail=f"database not found at {DB_PATH}; run scripts/ingest_amc.py first")
    return schema.connect(DB_PATH)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/contests")
def list_contests():
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT contest_id, contest, year, label, num_problems FROM contests ORDER BY year, label"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/api/contests/{contest_id}")
def get_contest(contest_id: str):
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT problem_id, position, question, choice_a, choice_b, choice_c,
                   choice_d, choice_e, answer, seed_elo
            FROM problems WHERE contest_id = ? ORDER BY position
            """,
            (contest_id,),
        ).fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail=f"contest {contest_id!r} not found")
        problems = []
        for r in rows:
            concept_rows = conn.execute(
                "SELECT concept, category FROM problem_concepts WHERE problem_id = ?", (r["problem_id"],)
            ).fetchall()
            problems.append(
                {
                    **dict(r),
                    "categories": sorted({c["category"] for c in concept_rows}),
                }
            )
        return {"contest_id": contest_id, "problems": problems}
    finally:
        conn.close()


class ContestResponseInput(BaseModel):
    contest_id: str
    responses: list[str | None] = Field(..., min_length=25, max_length=25)


class PredictRequest(BaseModel):
    contests: list[ContestResponseInput]
    n_simulations: int = Field(default=20_000, ge=1000, le=200_000)
    seed: int | None = None


@app.post("/api/predict")
def predict(req: PredictRequest):
    if not req.contests:
        raise HTTPException(status_code=400, detail="at least one contest's responses is required")

    conn = get_conn()
    try:
        problems = load_problems(conn)
        known_ids = set(problems["contest_id"].unique())
        contest_responses = []
        for c in req.contests:
            if c.contest_id not in known_ids:
                raise HTTPException(status_code=404, detail=f"unknown contest_id {c.contest_id!r}")
            try:
                contest_responses.append(ContestResponses(c.contest_id, c.responses))
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e)) from e

        result = predict_2026(conn, contest_responses, n_simulations=req.n_simulations, seed=req.seed)
        return _nan_to_none(asdict(result))
    finally:
        conn.close()
