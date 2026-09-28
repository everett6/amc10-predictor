-- AMC 10 Future Score Predictor -- SQLite schema
-- One row per contest, one row per problem, one row per (problem, concept)
-- tag (a problem can carry multiple concepts), plus tables for a user's
-- historical responses (the predictor's input) and their resulting
-- predictions (cached so re-viewing a prediction doesn't re-simulate).

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS contests (
    contest_id      TEXT PRIMARY KEY,      -- e.g. "2021A", "2021C" (Fall)
    contest         TEXT NOT NULL,         -- e.g. "amc10"
    year            INTEGER NOT NULL,
    label           TEXT NOT NULL,         -- "A" | "B" | "C" | "D"
    num_problems    INTEGER NOT NULL,
    source_url      TEXT NOT NULL,
    fetched_at      TEXT NOT NULL          -- ISO 8601 UTC timestamp
);

CREATE TABLE IF NOT EXISTS problems (
    problem_id              TEXT PRIMARY KEY,   -- e.g. "2021A-13"
    contest_id              TEXT NOT NULL REFERENCES contests(contest_id) ON DELETE CASCADE,
    position                INTEGER NOT NULL,   -- 1..25, contest order
    question                TEXT NOT NULL,
    choice_a                TEXT NOT NULL,
    choice_b                TEXT NOT NULL,
    choice_c                TEXT NOT NULL,
    choice_d                TEXT NOT NULL,
    choice_e                TEXT NOT NULL,
    answer                  TEXT NOT NULL,      -- 'a'..'e'
    seed_elo                REAL NOT NULL,       -- LIVE's crowd-difficulty rating
    solve_time_seconds      REAL NOT NULL,
    solution_text           TEXT NOT NULL,
    amc10_problem_number    INTEGER NOT NULL,
    amc12_problem_number    INTEGER NOT NULL,
    amc8_problem_number     INTEGER NOT NULL,
    aime_problem_number     INTEGER NOT NULL,
    UNIQUE(contest_id, position)
);

CREATE TABLE IF NOT EXISTS problem_concepts (
    problem_id      TEXT NOT NULL REFERENCES problems(problem_id) ON DELETE CASCADE,
    concept         TEXT NOT NULL,       -- fine-grained LIVE concept tag
    category        TEXT NOT NULL,       -- one of the 15 broad categories
    PRIMARY KEY (problem_id, concept)
);

CREATE INDEX IF NOT EXISTS idx_problem_concepts_category ON problem_concepts(category);
CREATE INDEX IF NOT EXISTS idx_problems_contest ON problems(contest_id);

-- A "user" here is just an opaque identifier the API assigns per session;
-- we never require a login. Responses are the raw input to the predictor.
CREATE TABLE IF NOT EXISTS user_responses (
    user_id         TEXT NOT NULL,
    contest_id      TEXT NOT NULL REFERENCES contests(contest_id),
    position        INTEGER NOT NULL,
    response        TEXT,                -- 'a'..'e' or NULL for blank
    recorded_at     TEXT NOT NULL,
    PRIMARY KEY (user_id, contest_id, position)
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    input_contests  TEXT NOT NULL,        -- JSON list of contest_ids used
    result_json     TEXT NOT NULL         -- full PredictionResult, serialized
);
