import { useEffect, useMemo, useState } from "react";
import "./App.css";
import { AbilityChart } from "./components/AbilityChart";
import { AnswerGrid } from "./components/AnswerGrid";
import { ScoreDistributionChart } from "./components/ScoreDistributionChart";
import {
  type ContestDetail,
  type ContestSummary,
  type PredictionResult,
  getContest,
  listContests,
  predict,
} from "./api";

type LoadState = "idle" | "loading" | "error";

function App() {
  const [contests, setContests] = useState<ContestSummary[]>([]);
  const [selectedContestId, setSelectedContestId] = useState<string>("");
  const [contestDetail, setContestDetail] = useState<ContestDetail | null>(null);
  const [responses, setResponses] = useState<(string | null)[]>(Array(25).fill(null));
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [state, setState] = useState<LoadState>("idle");
  const [error, setError] = useState<string>("");

  useEffect(() => {
    listContests()
      .then((cs) => {
        setContests(cs);
        if (cs.length) setSelectedContestId(cs[cs.length - 1].contest_id);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!selectedContestId) return;
    getContest(selectedContestId)
      .then((d) => {
        setContestDetail(d);
        setResponses(Array(25).fill(null));
      })
      .catch((e) => setError(String(e)));
  }, [selectedContestId]);

  const answeredCount = useMemo(() => responses.filter((r) => r !== null).length, [responses]);

  function handleChange(position: number, letter: string | null) {
    setResponses((prev) => {
      const next = [...prev];
      next[position - 1] = letter;
      return next;
    });
  }

  function fillCorrect() {
    if (!contestDetail) return;
    setResponses(contestDetail.problems.map((p) => p.answer));
  }

  function clearAll() {
    setResponses(Array(25).fill(null));
  }

  async function handleSubmit() {
    if (!selectedContestId) return;
    setState("loading");
    setError("");
    try {
      const r = await predict([{ contest_id: selectedContestId, responses }]);
      setResult(r);
      setState("idle");
    } catch (e) {
      setError(String(e));
      setState("error");
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>AMC 10 Future Score Predictor</h1>
        <p className="subtitle">
          Enter your answers to a past AMC 10 contest and get a Monte Carlo-simulated projection of your
          2026 AMC 10 score, with a per-topic breakdown and improvement suggestions.
        </p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <section className="panel">
        <h2>1. Choose a past contest</h2>
        <select value={selectedContestId} onChange={(e) => setSelectedContestId(e.target.value)}>
          {contests.map((c) => (
            <option key={c.contest_id} value={c.contest_id}>
              AMC 10{c.label} {c.year}
            </option>
          ))}
        </select>
      </section>

      {contestDetail && (
        <section className="panel">
          <div className="panel-header-row">
            <h2>2. Enter your answers ({answeredCount}/25 answered)</h2>
            <div className="panel-actions">
              <button type="button" onClick={fillCorrect} className="secondary-btn">
                Fill all correct (demo)
              </button>
              <button type="button" onClick={clearAll} className="secondary-btn">
                Clear
              </button>
            </div>
          </div>
          <AnswerGrid problems={contestDetail.problems} responses={responses} onChange={handleChange} />
          <button type="button" className="primary-btn" onClick={handleSubmit} disabled={state === "loading"}>
            {state === "loading" ? "Simulating..." : "Predict my 2026 score"}
          </button>
        </section>
      )}

      {result && (
        <section className="panel results">
          <h2>Your 2026 Prediction</h2>

          <div className="score-summary">
            <div className="score-card">
              <span className="score-card-label">Median projected score</span>
              <span className="score-card-value">{result.simulation_summary.median_score.toFixed(1)}</span>
            </div>
            <div className="score-card">
              <span className="score-card-label">90% interval</span>
              <span className="score-card-value">
                {result.simulation_summary.percentiles.p5.toFixed(0)} - {result.simulation_summary.percentiles.p95.toFixed(0)}
              </span>
            </div>
            <div className="score-card">
              <span className="score-card-label">Expected correct / wrong / blank</span>
              <span className="score-card-value">
                {result.simulation_summary.expected_correct.toFixed(1)} / {result.simulation_summary.expected_wrong.toFixed(1)} /{" "}
                {result.simulation_summary.expected_blank.toFixed(1)}
              </span>
            </div>
          </div>

          <h3>Score distribution ({result.n_simulations.toLocaleString()} simulations)</h3>
          <ScoreDistributionChart histogram={result.histogram} medianScore={result.simulation_summary.median_score} />

          <h3>Ability by topic (Elo scale, global ability = {Math.round(result.ability_global)})</h3>
          <AbilityChart abilityByCategory={result.ability_by_category} globalAbility={result.ability_global} />

          <h3>Why this prediction?</h3>
          <p className="explanation">{result.explanation}</p>

          {result.counterfactuals.length > 0 && (
            <>
              <h3>If you improved your weakest topics...</h3>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Topic</th>
                    <th>+Elo</th>
                    <th>Baseline mean score</th>
                    <th>Improved mean score</th>
                    <th>Gain</th>
                  </tr>
                </thead>
                <tbody>
                  {result.counterfactuals.map((cf) => (
                    <tr key={cf.category}>
                      <td>{cf.category}</td>
                      <td>+{cf.delta_elo}</td>
                      <td>{cf.baseline_mean_score.toFixed(1)}</td>
                      <td>{cf.improved_mean_score.toFixed(1)}</td>
                      <td className="gain">+{cf.score_gain.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          <h3>Question-level detail for a typical 2026 contest</h3>
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                <th>P(correct)</th>
                <th>Typical difficulty</th>
                <th>Likely topics</th>
              </tr>
            </thead>
            <tbody>
              {result.question_level.map((q) => (
                <tr key={q.position}>
                  <td>{q.position}</td>
                  <td>{(q.expected_p_correct * 100).toFixed(0)}%</td>
                  <td>{Math.round(q.typical_difficulty)}</td>
                  <td>{q.likely_categories.join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}

export default App;
