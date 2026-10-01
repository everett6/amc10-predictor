import { contestName, type Forecast } from "../api";
import { DifficultyCurve, QuestionStrip, ScoreHistogram, TopicIntervals } from "./charts";

const MODEL_NAMES: Record<string, [string, string]> = {
  irt_hier: ["Topic-aware rating", "overall rating plus a strength per topic"],
  irt_guess: ["Rating with lucky guesses", "assumes one answer in five can be right by luck"],
  irt_flat: ["Gentler rating curve", "results depend less on rated difficulty"],
  irt_global: ["Single rating", "one overall rating, no topics"],
  logit_feat: ["Difficulty and position", "learns its own difficulty slope and a late-paper effect"],
  isotonic: ["Free-form curve", "accuracy against difficulty with no assumed shape"],
  gbt: ["Boosted trees", "difficulty, position and topics, combined freely"],
};

const pct = (v: number, d = 0) => `${(v * 100).toFixed(d)}%`;

interface Props {
  forecast: Forecast;
  target: number;
  onTargetChange: (t: number) => void;
}

export function ForecastView({ forecast, target, onTargetChange }: Props) {
  const s = forecast.simulation_summary;
  const targetProb = forecast.target_probabilities.find((t) => t.score === target);
  const check = forecast.self_check;
  const weights = Object.entries(forecast.model_weights).sort((a, b) => b[1] - a[1]);
  const seenTopics = forecast.topics.filter((t) => t.n_answered > 0);
  const spread = seenTopics.length ? seenTopics[0].ability - seenTopics[seenTopics.length - 1].ability : 0;
  const halfWidth = seenTopics.length
    ? seenTopics.reduce((sum, t) => sum + (t.ability_high - t.ability_low) / 2, 0) / seenTopics.length
    : 0;
  const topicsUnclear = spread < halfWidth;

  return (
    <div className="forecast">
      <section>
        <h2>
          Most likely around {s.median_score}, and nine times in ten between {s.percentiles.p5} and {s.percentiles.p95}
        </h2>
        <p className="lede">
          From {s.n_simulations.toLocaleString()} simulated sittings of a {forecast.contest_forecast.target_year} AMC 10:
          on average {s.expected_correct.toFixed(1)} right, {s.expected_wrong.toFixed(1)} wrong and{" "}
          {s.expected_blank.toFixed(1)} left blank.
        </p>
        <ScoreHistogram forecast={forecast} target={targetProb ? target : null} />
        <div className="target-row">
          <label htmlFor="target">Chance of scoring at least</label>
          <select id="target" value={target} onChange={(e) => onTargetChange(Number(e.target.value))}>
            {forecast.target_probabilities.map((t) => (
              <option key={t.score} value={t.score}>{t.score}</option>
            ))}
          </select>
          <strong>{targetProb ? pct(targetProb.probability, targetProb.probability < 0.1 ? 1 : 0) : "–"}</strong>
          <span className="target-all">
            {forecast.target_probabilities.map((t) => `${t.score}+: ${pct(t.probability, t.probability < 0.1 ? 1 : 0)}`).join("   ")}
          </span>
        </div>
      </section>

      <section>
        <h3>Your sittings</h3>
        <table>
          <thead>
            <tr><th>Paper</th><th>Taken</th><th className="num">Right</th><th className="num">Wrong</th><th className="num">Blank</th><th className="num">Score</th></tr>
          </thead>
          <tbody>
            {forecast.inputs.map((i, k) => (
              <tr key={k}>
                <td>{contestName(i.contest_id)}</td>
                <td>{i.taken_on ?? "no date"}</td>
                <td className="num">{i.correct}</td>
                <td className="num">{i.wrong}</td>
                <td className="num">{i.blank}</td>
                <td className="num">
                  {i.score_current_rules}
                  {i.score_rules_of_the_year !== i.score_current_rules && (
                    <span className="aside"> ({i.score_rules_of_the_year} under {i.year} rules, {i.blank_points_that_year} per blank)</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {forecast.growth_per_year !== null && (
          <p className="note">
            Trend across your dated sittings: {forecast.growth_per_year >= 0 ? "+" : ""}
            {Math.round(forecast.growth_per_year)} rating points a year (give or take {Math.round(forecast.growth_sd ?? 0)}),
            carried forward to {forecast.contest_forecast.target_date}.
          </p>
        )}
      </section>

      <section>
        <h3>Strength by topic</h3>
        <p className="note">
          Dot: best estimate. Line: where your true level probably sits (90% range). Topics with few answers
          stay close to your overall rating until there is evidence to move them.
        </p>
        <TopicIntervals forecast={forecast} />
        {topicsUnclear && (
          <p className="note">
            The gaps between your topics are smaller than the uncertainty in each one, so there is no clear
            strong or weak topic yet. More sittings will separate them.
          </p>
        )}
      </section>

      <section>
        <h3>How difficulty affects you</h3>
        <DifficultyCurve forecast={forecast} />
      </section>

      <section>
        <h3>Problem by problem on a typical {forecast.contest_forecast.target_year} paper</h3>
        <p className="note">Chance each problem ends up right. Hover a problem for the breakdown.</p>
        <QuestionStrip forecast={forecast} />
      </section>

      {forecast.counterfactuals.length > 0 && (
        <section>
          <h3>What 150 more rating points in one topic would buy</h3>
          <p className="note">
            Shown for your three lowest-rated topics{topicsUnclear ? " (which are not clearly weaker than the rest)" : ""}.
            The gain depends mostly on how often the topic comes up.
          </p>
          <table>
            <thead>
              <tr><th>Topic</th><th className="num">Rating now</th><th className="num">Average score now</th><th className="num">After</th><th className="num">Gain</th></tr>
            </thead>
            <tbody>
              {forecast.counterfactuals.map((c) => (
                <tr key={c.category}>
                  <td>{c.category}</td>
                  <td className="num">{Math.round(c.current_ability)}</td>
                  <td className="num">{c.baseline_mean_score.toFixed(1)}</td>
                  <td className="num">{c.improved_mean_score.toFixed(1)}</td>
                  <td className="num">+{c.score_gain.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section>
        <h3>How well the models predict your own answers</h3>
        <p className="note">
          {check.stacked
            ? `Each answered problem (${check.n_items}) was hidden in turn (${check.fold_type}), predicted by models fitted on the rest, then compared with what you actually did. Lower error is better; a model's share is how much it contributes to your forecast.`
            : check.note}
        </p>
        <table>
          <thead>
            <tr><th>Model</th><th className="num">Share</th><th className="num">Error (log loss)</th><th className="num">Calls right</th></tr>
          </thead>
          <tbody>
            {weights.map(([name, w]) => (
              <tr key={name}>
                <td>
                  {MODEL_NAMES[name]?.[0] ?? name}
                  <span className="aside"> {MODEL_NAMES[name]?.[1]}</span>
                </td>
                <td className="num">{pct(w)}</td>
                <td className="num">{check.members[name] ? check.members[name].log_loss.toFixed(3) : "–"}</td>
                <td className="num">{check.members[name] ? pct(check.members[name].accuracy) : "–"}</td>
              </tr>
            ))}
            {check.ensemble && (
              <tr className="total">
                <td>Combined forecast</td>
                <td className="num">100%</td>
                <td className="num">{check.ensemble.log_loss.toFixed(3)}</td>
                <td className="num">{pct(check.ensemble.accuracy)}</td>
              </tr>
            )}
            {check.base_rate && (
              <tr>
                <td>
                  Baseline<span className="aside"> same chance for every problem</span>
                </td>
                <td className="num">–</td>
                <td className="num">{check.base_rate.log_loss.toFixed(3)}</td>
                <td className="num">{pct(check.base_rate.accuracy)}</td>
              </tr>
            )}
          </tbody>
        </table>
      </section>

      <section>
        <h3>How this forecast was made</h3>
        <p className="explanation">{forecast.explanation}</p>
        <p className="note">
          Difficulty ratings come from LIVE by Po-Shen Loh. The forecast has been checked against the history
          of the papers themselves and against simulated students, not against real students' results, so
          treat the range as a guide.
        </p>
      </section>
    </div>
  );
}
