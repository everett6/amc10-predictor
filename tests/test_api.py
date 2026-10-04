import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AMC10_USER_DATA_DIR", str(tmp_path))
    import importlib

    import api.app as app_module
    importlib.reload(app_module)
    if not app_module.DB_PATH.exists() and not app_module.SEED_DIR.exists():
        pytest.skip("no database or seed data")
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


SHEET = ["a", "b", "c", "d", "e"] * 4 + [None] * 5


def test_contests_and_answer_key(client):
    contests = client.get("/api/contests").json()
    assert len(contests) == 51 and contests[0]["contest_id"] == "2001"
    detail = client.get("/api/contests/2021A").json()
    assert len(detail["problems"]) == 25
    assert set(detail["problems"][0]) == {"position", "answer", "seed_elo", "categories"}   # no problem text
    assert client.get("/api/contests/1999").status_code == 404


def test_attempts_round_trip(client):
    assert client.get("/api/attempts").json() == {"attempts": []}
    sitting = {"id": "x1", "contest_id": "2021A", "responses": SHEET, "taken_on": "2026-09-01"}
    assert client.put("/api/attempts", json={"attempts": [sitting]}).json() == {"saved": 1}
    assert client.get("/api/attempts").json()["attempts"][0] == sitting


def test_predict_v2_and_v1(client):
    body = {"contests": [{"contest_id": "2021A", "responses": SHEET}], "n_simulations": 1000, "seed": 1}
    v2 = client.post("/api/predict", json=body)
    assert v2.status_code == 200 and v2.json()["engine"] == "v2"
    v1 = client.post("/api/predict", json={**body, "engine": "v1"})
    assert v1.status_code == 200 and v1.json()["engine"] == "v1"


def test_predict_rejects_bad_input(client):
    assert client.post("/api/predict", json={"contests": []}).status_code == 400
    assert client.post("/api/predict", json={"contests": [{"contest_id": "2021A", "responses": ["z"] * 25}]}).status_code == 422
    assert client.post("/api/predict", json={"contests": [{"contest_id": "2021A", "responses": ["a"] * 24}]}).status_code == 422
    assert client.post("/api/predict", json={"contests": [{"contest_id": "nope", "responses": SHEET}]}).status_code == 404
