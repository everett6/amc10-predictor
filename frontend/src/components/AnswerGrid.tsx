import type { ContestProblem } from "../api";

const LETTERS = ["a", "b", "c", "d", "e"] as const;

interface Props {
  problems: ContestProblem[];
  responses: (string | null)[];
  onChange: (position: number, letter: string | null) => void;
}

export function AnswerGrid({ problems, responses, onChange }: Props) {
  return (
    <div className="answer-grid">
      {problems.map((p) => {
        const current = responses[p.position - 1];
        return (
          <div className="answer-row" key={p.problem_id}>
            <span className="answer-row-number">{p.position}</span>
            <div className="answer-row-choices">
              {LETTERS.map((letter) => (
                <button
                  key={letter}
                  type="button"
                  className={`choice-btn${current === letter ? " selected" : ""}`}
                  onClick={() => onChange(p.position, current === letter ? null : letter)}
                  title={p[`choice_${letter}` as const]}
                >
                  {letter.toUpperCase()}
                </button>
              ))}
              <button
                type="button"
                className={`choice-btn blank-btn${current === null ? " selected" : ""}`}
                onClick={() => onChange(p.position, null)}
              >
                Blank
              </button>
            </div>
            <span className="answer-row-elo" title="LIVE difficulty rating">
              {Math.round(p.seed_elo)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
