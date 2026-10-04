import json

import pytest

from ingestion.live_parser import LiveParseError, contest_url, parse_contest_page


def _fake_html(base_questions) -> str:
    data = {"props": {"pageProps": {"baseQuestions": base_questions}}}
    return f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(data)}</script></html>'


def _valid_question(answer="d", concepts="order of operations"):
    return {
        "a": "1", "b": "2", "c": "5", "d": "8", "e": "12",
        "answer": answer, "answerFillIn": "8",
        "concepts": concepts,
        "question": "What is the value of X?",
        "seedElo": 450, "solveTimeSeconds": 20,
        "solutions": "Because reasons.",
        "amc10ProblemNumber": 0, "amc12ProblemNumber": 0,
        "amc8ProblemNumber": 2, "aimeProblemNumber": 0,
    }


def test_parse_valid_contest():
    questions = [_valid_question() for _ in range(25)]
    html = _fake_html(questions)
    contest = parse_contest_page(html, year=2021, label="A")
    assert contest.contest_id == "2021A"
    assert len(contest.problems) == 25
    assert contest.problems[0].position == 1
    assert contest.problems[0].answer == "d"
    assert contest.problems[0].choices["d"] == "8"


def test_missing_next_data_raises():
    with pytest.raises(LiveParseError):
        parse_contest_page("<html><body>nope</body></html>", year=2021, label="A")


def test_wrong_problem_count_raises():
    questions = [_valid_question() for _ in range(24)]
    html = _fake_html(questions)
    with pytest.raises(LiveParseError):
        parse_contest_page(html, year=2021, label="A")


def test_invalid_answer_letter_raises():
    questions = [_valid_question(answer="z")] + [_valid_question() for _ in range(24)]
    html = _fake_html(questions)
    with pytest.raises(LiveParseError):
        parse_contest_page(html, year=2021, label="A")


def test_contest_url():
    assert contest_url(2021, "A") == "https://live.poshenloh.com/past-contests/amc10/2021A"


def test_sanitizes_latex_thin_comma():
    q = _valid_question()
    q["a"] = "17{,}402"
    questions = [q] + [_valid_question() for _ in range(24)]
    html = _fake_html(questions)
    contest = parse_contest_page(html, year=2021, label="A")
    assert contest.problems[0].choices["a"] == "17,402"


def test_concepts_split_and_cleaned():
    q = _valid_question(concepts="ratio and proportion; linear equation")
    questions = [q] + [_valid_question() for _ in range(24)]
    html = _fake_html(questions)
    contest = parse_contest_page(html, year=2021, label="A")
    assert contest.problems[0].concepts == ["ratio and proportion", "linear equation"]


def test_image_only_choice_keeps_a_placeholder():
    q = _valid_question()
    q["a"] = '<img src="/amc10/2001/17a.svg" style="max-height: 8rem"/>'
    html = _fake_html([q] + [_valid_question() for _ in range(24)])
    contest = parse_contest_page(html, year=2001, label="")
    assert contest.contest_id == "2001"
    assert contest.problems[0].choices["a"] == "[image: /amc10/2001/17a.svg]"


def test_stored_contest_carries_no_problem_text():
    from ingestion.pipeline import IngestionError, strip_text

    contest = strip_text(parse_contest_page(_fake_html([_valid_question() for _ in range(25)]), 2021, "A"))
    p = contest.problems[0]
    assert p.question == "" and p.solution_text == "" and set(p.choices.values()) == {""}
    assert p.answer == "d" and p.seed_elo == 450 and p.concepts == ["order of operations"]

    empty = _valid_question(); empty["question"] = ""
    with pytest.raises(IngestionError):
        strip_text(parse_contest_page(_fake_html([empty] + [_valid_question() for _ in range(24)]), 2021, "A"))
