"""Parses AMC 10 contest pages from live.poshenloh.com.

Each contest page (https://live.poshenloh.com/past-contests/amc10/<YEAR><LETTER>)
is a server-rendered Next.js page embedding a `__NEXT_DATA__` JSON blob
containing `props.pageProps.baseQuestions`, a 25-element array (one per
problem, in contest order) with fields including the answer choices,
correct answer letter, concepts, an Elo-style difficulty rating
(`seedElo`), and expected solve time.

This module is pure parsing logic (no network I/O) so it can be unit
tested against saved fixture HTML.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .sanitize import sanitize_concepts, sanitize_problem_field

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL
)

ANSWER_LETTERS = ("a", "b", "c", "d", "e")


class LiveParseError(ValueError):
    pass


@dataclass
class ParsedProblem:
    contest: str          # e.g. "amc10"
    year: int             # e.g. 2021
    label: str            # e.g. "A", "B", "C", "D" (Fall contests)
    position: int          # 1..25, contest order
    question: str
    choices: dict[str, str]     # {"a": "...", ..., "e": "..."}
    answer: str                  # correct letter, lowercased
    concepts: list[str]          # broad-taxonomy-ready raw concept strings
    seed_elo: float               # LIVE's crowd-sourced difficulty rating
    solve_time_seconds: float
    solution_text: str
    amc10_problem_number: int    # LIVE's cross-listing field (0 if N/A)
    amc12_problem_number: int
    amc8_problem_number: int
    aime_problem_number: int


@dataclass
class ParsedContest:
    contest: str
    year: int
    label: str
    problems: list[ParsedProblem]

    @property
    def contest_id(self) -> str:
        return f"{self.year}{self.label}"


def extract_next_data(html: str) -> dict:
    match = _NEXT_DATA_RE.search(html)
    if not match:
        raise LiveParseError("could not find __NEXT_DATA__ script tag in page")
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as e:
        raise LiveParseError(f"__NEXT_DATA__ was not valid JSON: {e}") from e


def _coerce_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_contest_page(html: str, year: int, label: str, contest: str = "amc10") -> ParsedContest:
    """Parse a single /past-contests/amc10/<YEAR><LABEL> page's HTML."""
    data = extract_next_data(html)
    try:
        page_props = data["props"]["pageProps"]
        base_questions = page_props["baseQuestions"]
    except KeyError as e:
        raise LiveParseError(f"unexpected page structure, missing key {e}") from e

    if not isinstance(base_questions, list) or not base_questions:
        raise LiveParseError("baseQuestions was empty or not a list")

    problems: list[ParsedProblem] = []
    for idx, q in enumerate(base_questions):
        position = idx + 1
        try:
            answer_letter = str(q["answer"]).strip().lower()
        except KeyError as e:
            raise LiveParseError(
                f"{contest} {year}{label} problem {position}: missing field {e}"
            ) from e
        if answer_letter not in ANSWER_LETTERS:
            raise LiveParseError(
                f"{contest} {year}{label} problem {position}: invalid answer "
                f"letter {answer_letter!r}"
            )
        choices = {}
        for letter in ANSWER_LETTERS:
            raw_choice = q.get(letter, "")
            choices[letter] = sanitize_problem_field(str(raw_choice))

        problems.append(
            ParsedProblem(
                contest=contest,
                year=year,
                label=label,
                position=position,
                question=sanitize_problem_field(str(q.get("question", ""))),
                choices=choices,
                answer=answer_letter,
                concepts=sanitize_concepts(str(q.get("concepts", ""))),
                seed_elo=_coerce_float(q.get("seedElo")),
                solve_time_seconds=_coerce_float(q.get("solveTimeSeconds")),
                solution_text=sanitize_problem_field(str(q.get("solutions", ""))),
                amc10_problem_number=_coerce_int(q.get("amc10ProblemNumber")),
                amc12_problem_number=_coerce_int(q.get("amc12ProblemNumber")),
                amc8_problem_number=_coerce_int(q.get("amc8ProblemNumber")),
                aime_problem_number=_coerce_int(q.get("aimeProblemNumber")),
            )
        )

    if len(problems) != 25:
        raise LiveParseError(
            f"{contest} {year}{label}: expected 25 problems, found {len(problems)}"
        )

    return ParsedContest(contest=contest, year=year, label=label, problems=problems)


def contest_url(year: int, label: str, contest: str = "amc10") -> str:
    return f"https://live.poshenloh.com/past-contests/{contest}/{year}{label}"
