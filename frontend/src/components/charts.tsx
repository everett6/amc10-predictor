import { useRef, useState, type ReactNode } from "react";
import type { Forecast } from "../api";

/* ---------- shared hover tooltip ---------- */
function useTip() {
  const box = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; body: ReactNode } | null>(null);
  const show = (e: React.MouseEvent, body: ReactNode) => {
    const rect = box.current?.getBoundingClientRect();
    if (!rect) return;
    setTip({ x: e.clientX - rect.left, y: e.clientY - rect.top, body });
  };
  const hide = () => setTip(null);
  const node = tip && (
    <div
      className="tip"
      style={{ left: Math.min(tip.x + 14, (box.current?.clientWidth ?? 600) - 190), top: Math.max(tip.y - 12, 0) }}
      role="status"
    >
      {tip.body}
    </div>
  );
  return { box, show, hide, node };
}

const scale = (d0: number, d1: number, r0: number, r1: number) => (v: number) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
const pct = (v: number, digits = 0) => `${(v * 100).toFixed(digits)}%`;

/* ---------- score distribution ---------- */
export function ScoreHistogram({ forecast, target }: { forecast: Forecast; target: number | null }) {
  const { box, show, hide, node } = useTip();
  const W = 760, H = 300, L = 44, R = 16, T = 34, B = 40;
  const bins = forecast.histogram;
  const maxP = Math.max(...bins.map((b) => b.probability));
  const yMax = Math.ceil(maxP * 20) / 20 || 0.05;
  const x = scale(0, 150, L, W - R);
  const y = scale(0, yMax, H - B, T);
  const s = forecast.simulation_summary;
  const p5 = s.percentiles.p5, p95 = s.percentiles.p95;
  const barW = Math.min(24, x(6) - x(0) - 2);
  const ticks = [0, 0.5, 1].map((f) => f * yMax);
  return (
    <div className="chart" ref={box} onMouseLeave={hide}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Simulated score distribution, median ${s.median_score}`}>
        <rect x={x(p5)} y={T} width={x(p95) - x(p5)} height={H - B - T} className="band" />
        <text x={(x(p5) + x(p95)) / 2} y={T - 8} className="axis-text" textAnchor="middle">
          9 in 10 simulated sittings land here
        </text>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} className="grid" />
            <text x={L - 8} y={y(t) + 4} className="axis-text" textAnchor="end">{pct(t)}</text>
          </g>
        ))}
        {bins.map((b) => {
          const cx = x((b.range_low + b.range_high) / 2);
          const h = H - B - y(b.probability);
          const body = (
            <>
              <strong>{b.range_low}–{b.range_high} points</strong>
              <span>{pct(b.probability, 1)} of sittings</span>
            </>
          );
          return (
            <g key={b.range_low} onMouseMove={(e) => show(e, body)}>
              <rect x={x(b.range_low)} y={T} width={x(b.range_high) - x(b.range_low)} height={H - B - T} fill="transparent" />
              {b.probability > 0 && (
                <path
                  className="bar"
                  d={`M${cx - barW / 2},${H - B} v${-Math.max(h - 4, 0)} q0,-4 4,-4 h${barW - 8} q4,0 4,4 v${Math.max(h - 4, 0)} z`}
                />
              )}
            </g>
          );
        })}
        <line x1={L} x2={W - R} y1={H - B} y2={H - B} className="axis" />
        {[0, 30, 60, 90, 120, 150].map((t) => (
          <text key={t} x={x(t)} y={H - B + 18} className="axis-text" textAnchor="middle">{t}</text>
        ))}
        <text x={(L + W - R) / 2} y={H - 4} className="axis-text" textAnchor="middle">Score</text>
        <line x1={x(s.median_score)} x2={x(s.median_score)} y1={T} y2={H - B} className="ref-line" />
        <text x={x(s.median_score) + 6} y={T + 12} className="label-text">median {s.median_score}</text>
        {target !== null && (
          <>
            <line x1={x(target)} x2={x(target)} y1={T + 18} y2={H - B} className="target-line" />
            <text x={x(target) + 6} y={T + 30} className="label-text">target {target}</text>
          </>
        )}
      </svg>
      {node}
    </div>
  );
}

/* ---------- ability by topic: dot + 90% interval ---------- */
export function TopicIntervals({ forecast }: { forecast: Forecast }) {
  const { box, show, hide, node } = useTip();
  const rows = forecast.topics;
  const W = 760, rowH = 26, L = 300, R = 96, T = 30, B = 30;
  const H = T + rows.length * rowH + B;
  const lo = Math.min(...rows.map((r) => r.ability_low)), hi = Math.max(...rows.map((r) => r.ability_high));
  const d0 = Math.floor((lo - 40) / 200) * 200, d1 = Math.ceil((hi + 40) / 200) * 200;
  const x = scale(d0, d1, L, W - R);
  const ticks: number[] = [];
  for (let t = d0; t <= d1; t += 200) ticks.push(t);
  return (
    <div className="chart" ref={box} onMouseLeave={hide}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Estimated ability by topic with 90% intervals">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={T} y2={H - B} className="grid" />
            <text x={x(t)} y={H - B + 16} className="axis-text" textAnchor="middle">{t}</text>
          </g>
        ))}
        <line x1={x(forecast.ability_global)} x2={x(forecast.ability_global)} y1={T - 6} y2={H - B} className="ref-line" />
        <text x={x(forecast.ability_global)} y={T - 12} className="label-text" textAnchor="middle">
          overall {Math.round(forecast.ability_global)}
        </text>
        {rows.map((r, i) => {
          const cy = T + i * rowH + rowH / 2;
          const seen = r.n_answered > 0;
          const body = (
            <>
              <strong>{r.category}</strong>
              <span>Rating {Math.round(r.ability)} (likely {Math.round(r.ability_low)}–{Math.round(r.ability_high)})</span>
              <span>{seen ? `${r.n_correct} right of ${r.n_answered} answered` : "No answered problems yet: this is the overall estimate"}</span>
            </>
          );
          return (
            <g key={r.category} onMouseMove={(e) => show(e, body)}>
              <rect x={0} y={cy - rowH / 2} width={W} height={rowH} fill="transparent" />
              <text x={L - 12} y={cy + 4} className="row-text" textAnchor="end">{r.category}</text>
              <line x1={x(r.ability_low)} x2={x(r.ability_high)} y1={cy} y2={cy} className={seen ? "whisker" : "whisker faint"} />
              <circle cx={x(r.ability)} cy={cy} r={5} className={seen ? "dot" : "dot hollow"} />
              <text x={W - R + 12} y={cy + 4} className="axis-text">{seen ? `${r.n_correct}/${r.n_answered} right` : "no data"}</text>
            </g>
          );
        })}
        <text x={(L + W - R) / 2} y={H - 2} className="axis-text" textAnchor="middle">Rating (same scale as problem difficulty)</text>
      </svg>
      {node}
    </div>
  );
}

/* ---------- accuracy against difficulty ---------- */
export function DifficultyCurve({ forecast }: { forecast: Forecast }) {
  const { box, show, hide, node } = useTip();
  const [hover, setHover] = useState<number | null>(null);
  const W = 760, H = 300, L = 44, R = 16, T = 16, B = 40;
  const curve = forecast.difficulty_curve;
  const x = scale(400, 2800, L, W - R);
  const y = scale(0, 1, H - B, T);
  const path = (key: "p_correct_if_answered" | "p_answer") =>
    curve.map((c, i) => `${i ? "L" : "M"}${x(c.difficulty).toFixed(1)},${y(c[key]).toFixed(1)}`).join(" ");
  const observed = forecast.difficulty_bucket_accuracy
    .filter((b) => b.attempted > 0 && b.accuracy_attempted !== null)
    .map((b) => ({ mid: Number(b.bucket.split("-")[0]) + 100, acc: b.accuracy_attempted as number, n: b.attempted, correct: b.correct }));
  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const d = 400 + ((e.clientX - rect.left) / rect.width) * 2400;
    let best = 0;
    curve.forEach((c, i) => {
      if (Math.abs(c.difficulty - d) < Math.abs(curve[best].difficulty - d)) best = i;
    });
    setHover(best);
    const c = curve[best];
    const obs = observed.find((o) => Math.abs(o.mid - c.difficulty) <= 100);
    show(e, (
      <>
        <strong>Problems rated about {c.difficulty}</strong>
        <span><i className="key s1" /> right when answered: {pct(c.p_correct_if_answered)}</span>
        <span><i className="key s2" /> chance you answer: {pct(c.p_answer)}</span>
        {obs && <span>Your record nearby: {obs.correct} right of {obs.n}</span>}
      </>
    ));
  };
  return (
    <div className="chart" ref={box} onMouseLeave={() => { hide(); setHover(null); }}>
      <div className="legend">
        <span><i className="key s1" /> Right, when you answer</span>
        <span><i className="key s2" /> Chance you answer at all</span>
        <span><i className="key dotkey" /> Your actual record (per 200-point band)</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Chance of a right answer against problem difficulty">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} className="grid" />
            <text x={L - 8} y={y(t) + 4} className="axis-text" textAnchor="end">{pct(t)}</text>
          </g>
        ))}
        {[400, 800, 1200, 1600, 2000, 2400, 2800].map((t) => (
          <text key={t} x={x(t)} y={H - B + 18} className="axis-text" textAnchor="middle">{t}</text>
        ))}
        <text x={(L + W - R) / 2} y={H - 4} className="axis-text" textAnchor="middle">Problem difficulty rating</text>
        <line x1={L} x2={W - R} y1={H - B} y2={H - B} className="axis" />
        <path d={path("p_answer")} className="line s2" />
        <path d={path("p_correct_if_answered")} className="line s1" />
        {observed.map((o) => (
          <circle key={o.mid} cx={x(o.mid)} cy={y(o.acc)} r={4 + Math.min(o.n, 12) / 4} className="dot ringed" />
        ))}
        {hover !== null && (
          <>
            <line x1={x(curve[hover].difficulty)} x2={x(curve[hover].difficulty)} y1={T} y2={H - B} className="crosshair" />
            <circle cx={x(curve[hover].difficulty)} cy={y(curve[hover].p_correct_if_answered)} r={4.5} className="dot ringed" />
            <circle cx={x(curve[hover].difficulty)} cy={y(curve[hover].p_answer)} r={4.5} className="dot ringed s2" />
          </>
        )}
        <rect x={L} y={T} width={W - R - L} height={H - B - T} fill="transparent" onMouseMove={onMove} />
      </svg>
      {node}
    </div>
  );
}

/* ---------- 25-problem strip ---------- */
export function QuestionStrip({ forecast }: { forecast: Forecast }) {
  const { box, show, hide, node } = useTip();
  return (
    <div className="chart" ref={box} onMouseLeave={hide}>
      <div className="strip" role="list" aria-label="Chance of a right answer for each of the 25 problems">
        {forecast.question_level.map((q) => {
          const p = q.expected_p_correct;
          const body = (
            <>
              <strong>Problem {q.position}</strong>
              <span>Right: {pct(p)} (answer it {pct(q.p_answer)} of the time, right {pct(q.p_correct_if_answered)} of those)</span>
              <span>Typical rating {Math.round(q.typical_difficulty)}</span>
              <span>Often: {q.likely_categories.slice(0, 2).join("; ")}</span>
            </>
          );
          return (
            <div
              key={q.position}
              role="listitem"
              className="strip-cell"
              style={{ ["--p" as string]: p }}
              onMouseMove={(e) => show(e, body)}
            >
              <span className="strip-fill" />
              <span className="strip-num">{q.position}</span>
              <span className="strip-val">{pct(p)}</span>
            </div>
          );
        })}
      </div>
      {node}
    </div>
  );
}
