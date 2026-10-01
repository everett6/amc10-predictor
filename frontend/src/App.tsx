import { useEffect, useMemo, useRef, useState } from "react";
import {
  contestName,
  forecast as requestForecast,
  getContest,
  listContests,
  loadSittings,
  saveSittings,
  scoreOf,
  type ContestDetail,
  type ContestSummary,
  type Forecast,
  type Response,
  type Sitting,
} from "./api";
import { AnswerSheet } from "./components/AnswerSheet";
import { ForecastView } from "./components/ForecastView";

const TARGETS = [84, 90, 96, 102, 108, 114, 120, 132];
const DEFAULT_TARGET_DATE = "2026-11-05";
const today = () => {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
};
const newId = () => Math.random().toString(36).slice(2, 10);

export default function App() {
  const [contests, setContests] = useState<ContestSummary[]>([]);
  const [details, setDetails] = useState<Record<string, ContestDetail>>({});
  const [sittings, setSittings] = useState<Sitting[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [view, setView] = useState<"sheet" | "forecast">("sheet");
  const [result, setResult] = useState<Forecast | null>(null);
  const [stale, setStale] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [target, setTarget] = useState(102);
  const [targetDate, setTargetDate] = useState(DEFAULT_TARGET_DATE);
  const loaded = useRef(false);

  useEffect(() => {
    Promise.all([listContests(), loadSittings()])
      .then(([cs, saved]) => {
        setContests(cs);
        setSittings(saved);
        setSelected(saved[0]?.id ?? null);
        loaded.current = true;
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  // Fetch answer keys for every paper in use.
  useEffect(() => {
    const missing = [...new Set(sittings.map((s) => s.contest_id))].filter((id) => !details[id]);
    missing.forEach((id) =>
      getContest(id)
        .then((d) => setDetails((prev) => ({ ...prev, [id]: d })))
        .catch((e: Error) => setError(e.message)),
    );
  }, [sittings, details]);

  // Save after every change (debounced).
  useEffect(() => {
    if (!loaded.current) return;
    const timer = setTimeout(() => {
      saveSittings(sittings).catch((e: Error) => setError(`Couldn't save your sittings: ${e.message}`));
    }, 400);
    return () => clearTimeout(timer);
  }, [sittings]);

  const current = sittings.find((s) => s.id === selected) ?? null;
  const currentDetail = current ? details[current.contest_id] : undefined;
  const sortedContests = useMemo(() => [...contests].reverse(), [contests]);

  function update(id: string, patch: Partial<Sitting>) {
    setSittings((prev) => prev.map((s) => (s.id === id ? { ...s, ...patch } : s)));
    setStale(true);
  }

  function addSitting() {
    const used = new Set(sittings.map((s) => s.contest_id));
    const next = sortedContests.find((c) => !used.has(c.contest_id)) ?? sortedContests[0];
    if (!next) return;
    const sitting: Sitting = { id: newId(), contest_id: next.contest_id, taken_on: today(), responses: Array<Response>(25).fill(null) };
    setSittings((prev) => [...prev, sitting]);
    setSelected(sitting.id);
    setView("sheet");
    setStale(true);
  }

  function removeSitting(id: string) {
    setSittings((prev) => prev.filter((s) => s.id !== id));
    if (selected === id) setSelected(sittings.find((s) => s.id !== id)?.id ?? null);
    setStale(true);
  }

  async function runForecast() {
    setBusy(true);
    setError("");
    try {
      const r = await requestForecast(sittings, TARGETS, targetDate);
      setResult(r);
      setStale(false);
      setView("forecast");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const answeredTotal = sittings.reduce((n, s) => n + s.responses.filter((r) => r !== null).length, 0);

  return (
    <div className="app">
      <header className="masthead">
        <h1>AMC 10 Predictor</h1>
        <p>Enter the papers you have sat. Get the range your 2026 score is likely to fall in, and why.</p>
      </header>

      {error && (
        <div className="error" role="alert">
          {error}
          <button type="button" className="link" onClick={() => setError("")}>Dismiss</button>
        </div>
      )}

      <div className="layout">
        <aside className="rail">
          <h2>Your sittings</h2>
          {sittings.length === 0 && <p className="note">No papers yet. Add one you have sat, as practice or for real.</p>}
          <ul className="sitting-list">
            {sittings.map((s) => {
              const d = details[s.contest_id];
              const sc = d ? scoreOf(s.responses, d.problems) : null;
              return (
                <li key={s.id}>
                  <button
                    type="button"
                    className={`sitting${s.id === selected && view === "sheet" ? " active" : ""}`}
                    onClick={() => { setSelected(s.id); setView("sheet"); }}
                  >
                    <span className="sitting-name">{contestName(s.contest_id)}</span>
                    <span className="sitting-meta">{s.taken_on ?? "no date"}</span>
                    <span className="sitting-score">{sc ? sc.score : "…"}</span>
                  </button>
                </li>
              );
            })}
          </ul>
          <button type="button" className="btn" onClick={addSitting} disabled={contests.length === 0}>Add a paper</button>

          <div className="rail-forecast">
            <label htmlFor="target-date">Contest date</label>
            <input id="target-date" type="date" value={targetDate} onChange={(e) => { setTargetDate(e.target.value || DEFAULT_TARGET_DATE); setStale(true); }} />
            <button type="button" className="btn primary" onClick={runForecast} disabled={busy || sittings.length === 0}>
              {busy ? "Simulating…" : result && !stale ? "Forecast again" : "Forecast my 2026 score"}
            </button>
            {sittings.length > 0 && answeredTotal < 15 && (
              <p className="note">Only {answeredTotal} answered problems so far. The forecast will lean heavily on assumptions.</p>
            )}
            {result && (
              <button type="button" className="link" onClick={() => setView("forecast")}>
                {stale ? "Show last forecast (out of date)" : "Show forecast"}
              </button>
            )}
          </div>
        </aside>

        <main>
          {view === "forecast" && result ? (
            <>
              {stale && <p className="stale">Your sittings changed since this forecast. Run it again to update.</p>}
              <ForecastView forecast={result} target={target} onTargetChange={setTarget} />
            </>
          ) : current ? (
            <div className="editor">
              <div className="editor-head">
                <div className="field">
                  <label htmlFor="paper">Paper</label>
                  <select id="paper" value={current.contest_id} onChange={(e) => update(current.id, { contest_id: e.target.value })}>
                    {sortedContests.map((c) => (
                      <option key={c.contest_id} value={c.contest_id}>{contestName(c.contest_id)}</option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="taken">Date you sat it</label>
                  <input id="taken" type="date" value={current.taken_on ?? ""} max={today()} onChange={(e) => update(current.id, { taken_on: e.target.value || null })} />
                </div>
                {currentDetail && (
                  <div className="tally">
                    {(() => {
                      const sc = scoreOf(current.responses, currentDetail.problems);
                      return (
                        <>
                          <span className="tally-score">{sc.score}</span>
                          <span>{sc.correct} right, {sc.wrong} wrong, {sc.blank} blank</span>
                        </>
                      );
                    })()}
                  </div>
                )}
              </div>
              <p className="note">
                The date lets the forecast see whether you are improving. Leave it empty if you don't remember.
                {currentDetail && (
                  <>
                    {" "}
                    <a href={currentDetail.live_url} target="_blank" rel="noreferrer">Read this paper on LIVE</a>
                  </>
                )}
              </p>
              {currentDetail ? (
                <AnswerSheet key={current.id + current.contest_id} contest={currentDetail} responses={current.responses} onChange={(responses) => update(current.id, { responses })} />
              ) : (
                <p className="note">Loading the answer key…</p>
              )}
              <div className="editor-foot">
                <button type="button" className="link danger" onClick={() => removeSitting(current.id)}>Remove this sitting</button>
              </div>
            </div>
          ) : (
            <div className="empty">
              <h2>Start with one paper</h2>
              <p>
                Pick any AMC 10 from 2001 to 2025 that you have worked through under test conditions and fill in
                what you answered. One paper gives a rough range. Three or more, with dates, give a much tighter one,
                and papers from the last three years tell the forecast the most.
              </p>
              <button type="button" className="btn primary" onClick={addSitting} disabled={contests.length === 0}>Add a paper</button>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
