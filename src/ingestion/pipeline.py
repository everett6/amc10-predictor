"""Orchestrates fetching + parsing + storing AMC 10 contests.

Two modes:
  * "live": fetch each contest page from live.poshenloh.com over HTTP
    (through the rate-limited cached client) and parse it.
  * "seed": read previously-saved raw JSON snapshots from
    data/raw/live/*.json (used when live network access isn't
    available -- see scripts/ingest_amc.py --help).

Either way, results are written to:
  * data/raw/live/<contest_id>.json   -- raw fetched HTML/page data (live mode only)
  * the SQLite database (contests/problems/problem_concepts tables)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from ingestion.concepts import concept_to_category
from ingestion.http_client import RateLimitedCachedClient
from ingestion.live_parser import ParsedContest, contest_url, parse_contest_page
from storage import schema

logger = logging.getLogger(__name__)

# The 18 target contests per the project spec: every AMC 10 A/B from
# 2018-2025 plus the Fall 2021 A/B contests (labeled C/D in LIVE's
# numbering to distinguish them from the Feb 2021 A/B contests).
TARGET_CONTESTS: list[tuple[int, str]] = [
    (2018, "A"), (2018, "B"),
    (2019, "A"), (2019, "B"),
    (2020, "A"), (2020, "B"),
    (2021, "A"), (2021, "B"), (2021, "C"), (2021, "D"),
    (2022, "A"), (2022, "B"),
    (2023, "A"), (2023, "B"),
    (2024, "A"), (2024, "B"),
    (2025, "A"), (2025, "B"),
]


class IngestionError(RuntimeError):
    pass


def fetch_live_contest(
    client: RateLimitedCachedClient, year: int, label: str, raw_dir: Path
) -> ParsedContest:
    url = contest_url(year, label)
    result = client.get(url)
    if result.status != 200:
        raise IngestionError(f"GET {url} returned HTTP {result.status}")
    parsed = parse_contest_page(result.text, year=year, label=label)

    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{year}{label}.json"
    raw_path.write_text(
        json.dumps(
            {
                "source_url": url,
                "year": year,
                "label": label,
                "problems": [asdict(p) for p in parsed.problems],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return parsed


def load_seed_contest(seed_dir: Path, year: int, label: str) -> ParsedContest | None:
    """Load a previously-fetched contest from a seed JSON snapshot.

    Seed files are expected at seed_dir/<YEAR><LABEL>.json with the same
    shape written by fetch_live_contest (or an earlier WebFetch-derived
    equivalent produced outside this repo).
    """
    path = seed_dir / f"{year}{label}.json"
    if not path.exists():
        return None
    from ingestion.live_parser import ParsedProblem

    data = json.loads(path.read_text(encoding="utf-8"))
    problems = [ParsedProblem(**p) for p in data["problems"]]
    if len(problems) != 25:
        raise IngestionError(
            f"seed file {path} has {len(problems)} problems, expected 25"
        )
    return ParsedContest(contest="amc10", year=year, label=label, problems=problems)


def store_contest(conn, parsed: ParsedContest, source_url: str) -> None:
    contest_id = parsed.contest_id
    schema.upsert_contest(
        conn,
        contest_id=contest_id,
        contest=parsed.contest,
        year=parsed.year,
        label=parsed.label,
        num_problems=len(parsed.problems),
        source_url=source_url,
    )
    for p in parsed.problems:
        problem_id = f"{contest_id}-{p.position}"
        # Each fine-grained concept tag maps to its own broad category
        # (a problem tagged with several concepts can span several
        # categories, e.g. "modular arithmetic" + "casework").
        concepts_with_categories = [(concept, concept_to_category(concept)) for concept in p.concepts]
        schema.upsert_problem(
            conn,
            problem_id=problem_id,
            contest_id=contest_id,
            position=p.position,
            question=p.question,
            choices=p.choices,
            answer=p.answer,
            seed_elo=p.seed_elo,
            solve_time_seconds=p.solve_time_seconds,
            solution_text=p.solution_text,
            amc10_problem_number=p.amc10_problem_number,
            amc12_problem_number=p.amc12_problem_number,
            amc8_problem_number=p.amc8_problem_number,
            aime_problem_number=p.aime_problem_number,
            concepts_with_categories=concepts_with_categories,
        )
    conn.commit()


def run_ingestion(
    db_path: Path,
    mode: str = "live",
    contests: list[tuple[int, str]] | None = None,
    raw_live_dir: Path | None = None,
    seed_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> dict:
    """Run ingestion for the given contests (default: TARGET_CONTESTS).

    Returns a summary dict: {"ok": [...contest_ids...], "failed": {contest_id: reason}}
    """
    contests = contests or TARGET_CONTESTS
    raw_live_dir = raw_live_dir or Path("data/raw/live")
    # Seed mode reads from the same directory live mode writes to, so the
    # raw JSON snapshots committed to the repo (data/raw/live/*.json,
    # fetched live from live.poshenloh.com during development) double as
    # offline seed data when live network access isn't available.
    seed_dir = seed_dir or Path("data/raw/live")
    cache_dir = cache_dir or Path("data/raw/.http_cache")

    conn = schema.init_db(db_path)
    ok: list[str] = []
    failed: dict[str, str] = {}

    client = RateLimitedCachedClient(cache_dir=cache_dir) if mode == "live" else None

    for year, label in contests:
        contest_id = f"{year}{label}"
        try:
            if mode == "live":
                assert client is not None
                parsed = fetch_live_contest(client, year, label, raw_live_dir)
                source_url = contest_url(year, label)
            elif mode == "seed":
                parsed = load_seed_contest(seed_dir, year, label)
                if parsed is None:
                    raise IngestionError(f"no seed file found for {contest_id}")
                source_url = contest_url(year, label) + " (from seed snapshot)"
            else:
                raise ValueError(f"unknown mode {mode!r}")

            store_contest(conn, parsed, source_url)
            ok.append(contest_id)
            logger.info("ingested %s (%s)", contest_id, mode)
        except Exception as e:  # noqa: BLE001 - we want to keep going and report
            failed[contest_id] = str(e)
            logger.error("failed to ingest %s: %s", contest_id, e)

    conn.close()
    return {"ok": ok, "failed": failed}
