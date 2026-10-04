"""Local backend for the AMC 10 Future Score Predictor.

Runs on 127.0.0.1 only. In the desktop app, Electron starts this process,
and it serves both the API and the built UI from one origin.

Endpoints
  GET  /api/health
  GET  /api/contests              all ingested contests
  GET  /api/contests/{id}         answer key, difficulty and topics per problem
  GET  /api/attempts              the user's saved sittings
  PUT  /api/attempts              replace the saved sittings
  POST /api/predict               sittings in, 2026 forecast out

Paths can be overridden with environment variables (the desktop app sets
them): AMC10_DB_PATH, AMC10_SEED_DIR, AMC10_USER_DATA_DIR,
AMC10_FRONTEND_DIST.
"""
from __future__ import annotations

import sys
from pathlib import Path

if getattr(sys, "frozen", False):                       # PyInstaller bundle
    BUNDLE_ROOT = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
else:
    BUNDLE_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(BUNDLE_ROOT / "src"))

import datetime  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from dataclasses import asdict  # noqa: E402
from functools import lru_cache  # noqa: E402

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from analytics.data_access import ContestResponses, load_problems  # noqa: E402
from ingestion.pipeline import run_ingestion  # noqa: E402
from scoring.predictor import predict_2026  # noqa: E402
from scoring.predictor_v2 import DEFAULT_TARGET_DATE, predict_v2  # noqa: E402
from storage import schema  # noqa: E402

USER_DATA_DIR = Path(os.environ.get("AMC10_USER_DATA_DIR", Path.home() / ".amc10-predictor"))
SEED_DIR = Path(os.environ.get("AMC10_SEED_DIR", BUNDLE_ROOT / "data" / "raw" / "live"))
DEFAULT_DB = BUNDLE_ROOT / "data" / "amc10.sqlite3"
DB_PATH = Path(os.environ.get("AMC10_DB_PATH", DEFAULT_DB))
FRONTEND_DIST = Path(os.environ.get("AMC10_FRONTEND_DIST", BUNDLE_ROOT / "frontend" / "dist"))
ATTEMPTS_FILE = USER_DATA_DIR / "attempts.json"
LETTERS = {"a", "b", "c", "d", "e"}

app = FastAPI(title="AMC 10 Future Score Predictor", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    # Only the Vite dev server needs cross-origin access; the packaged UI is same-origin.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _jsonable(obj):
    """numpy scalars -> Python, NaN/inf -> null (strict JSON has neither)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not math.isfinite(obj) else float(obj)
    return obj


def ensure_db() -> Path:
    """Use the configured database, building it from the bundled seed
    snapshots on first run if it doesn't exist yet."""
    global DB_PATH
    if DB_PATH.exists():
        return DB_PATH
    if not SEED_DIR.exists():
        raise HTTPException(500, f"no database at {DB_PATH} and no seed data at {SEED_DIR}")
    if not os.access(DB_PATH.parent, os.W_OK):
        DB_PATH = USER_DATA_DIR / "amc10.sqlite3"
        if DB_PATH.exists():
            return DB_PATH
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = run_ingestion(DB_PATH, mode="seed", seed_dir=SEED_DIR)
    if summary["failed"]:
        raise HTTPException(500, f"seed ingestion failed: {summary['failed']}")
    return DB_PATH


@lru_cache(maxsize=1)
def problems_frame():
    conn = schema.connect(ensure_db())
    try:
        return load_problems(conn)
    finally:
        conn.close()


@app.get("/api/health")
def health():
    return {"status": "ok", "version": app.version}


@app.get("/api/contests")
def list_contests():
    p = problems_frame()
    contests = p.groupby("contest_id").agg(year=("year", "first"), label=("label", "first"),
                                           mean_difficulty=("seed_elo", "mean")).reset_index()
    contests = contests.sort_values(["year", "label"])
    return _jsonable(contests.to_dict(orient="records"))


@app.get("/api/contests/{contest_id}")
def get_contest(contest_id: str):
    p = problems_frame()
    subset = p[p["contest_id"] == contest_id].sort_values("position")
    if subset.empty:
        raise HTTPException(404, f"contest {contest_id!r} not found")
    return _jsonable({
        "contest_id": contest_id,
        "live_url": f"https://live.poshenloh.com/past-contests/amc10/{contest_id}",
        "problems": [
            {"position": int(r.position), "answer": r.answer, "seed_elo": r.seed_elo,
             "categories": sorted(set(r.categories))}
            for r in subset.itertuples()
        ],
    })


class Sitting(BaseModel):
    id: str | None = None
    contest_id: str
    responses: list[str | None] = Field(..., min_length=25, max_length=25)
    taken_on: datetime.date | None = None


class AttemptsPayload(BaseModel):
    attempts: list[Sitting]


def _validate(sittings: list[Sitting]) -> list[ContestResponses]:
    known = set(problems_frame()["contest_id"].unique())
    out = []
    for s in sittings:
        if s.contest_id not in known:
            raise HTTPException(404, f"unknown contest_id {s.contest_id!r}")
        cleaned: list[str | None] = []
        for r in s.responses:
            r = (r or "").strip().lower()
            if r and r not in LETTERS:
                raise HTTPException(422, f"{s.contest_id}: answers must be A-E or blank, got {r!r}")
            cleaned.append(r or None)
        out.append(ContestResponses(s.contest_id, cleaned, s.taken_on))
    return out


@app.get("/api/attempts")
def get_attempts():
    if not ATTEMPTS_FILE.exists():
        return {"attempts": []}
    try:
        return json.loads(ATTEMPTS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(500, f"{ATTEMPTS_FILE} is not valid JSON") from None


@app.put("/api/attempts")
def put_attempts(payload: AttemptsPayload):
    _validate(payload.attempts)
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ATTEMPTS_FILE.with_suffix(".tmp")
    tmp.write_text(payload.model_dump_json(indent=1), encoding="utf-8")
    tmp.replace(ATTEMPTS_FILE)
    return {"saved": len(payload.attempts)}


class PredictRequest(BaseModel):
    contests: list[Sitting]
    n_simulations: int = Field(default=20_000, ge=1000, le=100_000)
    seed: int | None = None
    engine: str = Field(default="v2", pattern="^(v1|v2)$")
    target_date: datetime.date = DEFAULT_TARGET_DATE
    targets: list[float] = Field(default=[90.0, 100.0, 110.0, 120.0], max_length=8)


@app.post("/api/predict")
def predict(req: PredictRequest):
    if not req.contests:
        raise HTTPException(400, "at least one sitting is required")
    sittings = _validate(req.contests)
    problems = problems_frame()
    if req.engine == "v1":
        conn = schema.connect(ensure_db())
        try:
            result = asdict(predict_2026(conn, sittings, n_simulations=req.n_simulations, seed=req.seed))
        finally:
            conn.close()
        result["engine"] = "v1"
        return _jsonable(result)
    return _jsonable(predict_v2(problems, sittings, n_simulations=req.n_simulations, seed=req.seed,
                                target_date=req.target_date, targets=tuple(req.targets)))


# Serve the built UI last so /api/* routes take precedence.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="ui")


def main() -> None:
    """Entry point for the packaged backend: `amc10-backend --port N`."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # If the desktop shell that launched us dies without cleaning up, the
    # parent PID changes (POSIX re-parents orphans); stop instead of lingering.
    if os.name == "posix":
        import threading
        import time

        parent = os.getppid()

        def watch_parent() -> None:
            while os.getppid() == parent:
                time.sleep(2.0)
            os._exit(0)

        threading.Thread(target=watch_parent, daemon=True).start()

    ensure_db()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
