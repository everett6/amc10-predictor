import { useRef, useState } from "react";
import type { ContestDetail, Letter, Response } from "../api";

const LETTERS: Letter[] = ["a", "b", "c", "d", "e"];

interface Props {
  contest: ContestDetail;
  responses: Response[];
  onChange: (responses: Response[]) => void;
}

/** Parse "ABCDE-B A..." into responses: letters fill, "-", "_", "." or "x" leave a blank. */
export function parseAnswerString(text: string): Response[] {
  const out: Response[] = [];
  for (const ch of text.toLowerCase()) {
    if ("abcde".includes(ch)) out.push(ch as Letter);
    else if ("-_.x".includes(ch)) out.push(null);
    if (out.length === 25) break;
  }
  return out;
}

export function AnswerSheet({ contest, responses, onChange }: Props) {
  const [cursor, setCursor] = useState(0);
  const [paste, setPaste] = useState("");
  const rowRefs = useRef<(HTMLDivElement | null)[]>([]);

  function set(index: number, value: Response) {
    const next = [...responses];
    next[index] = value;
    onChange(next);
  }

  function focusRow(index: number) {
    const clamped = Math.max(0, Math.min(24, index));
    setCursor(clamped);
    rowRefs.current[clamped]?.focus();
  }

  function onKeyDown(e: React.KeyboardEvent, index: number) {
    const key = e.key.toLowerCase();
    if (LETTERS.includes(key as Letter)) {
      set(index, key as Letter);
      focusRow(index + 1);
      e.preventDefault();
    } else if (key === " " || key === "-" || key === "backspace" || key === "delete") {
      set(index, null);
      focusRow(key === "backspace" ? index - 1 : index + 1);
      e.preventDefault();
    } else if (key === "arrowdown" || key === "enter") {
      focusRow(index + 1);
      e.preventDefault();
    } else if (key === "arrowup") {
      focusRow(index - 1);
      e.preventDefault();
    }
  }

  function applyPaste() {
    const parsed = parseAnswerString(paste);
    if (parsed.length === 0) return;
    const next: Response[] = Array(25).fill(null);
    parsed.forEach((r, i) => (next[i] = r));
    onChange(next);
    setPaste("");
  }

  const parsedCount = parseAnswerString(paste).length;

  return (
    <div className="sheet">
      <div className="sheet-entry">
        <label htmlFor="paste-answers">Type or paste all 25 answers</label>
        <div className="sheet-entry-row">
          <input
            id="paste-answers"
            value={paste}
            onChange={(e) => setPaste(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && applyPaste()}
            placeholder="e.g. DCBAE CDB-A … use - for a blank"
            spellCheck={false}
          />
          <button type="button" className="btn" onClick={applyPaste} disabled={parsedCount === 0}>
            Fill sheet{paste ? ` (${parsedCount}/25)` : ""}
          </button>
        </div>
        <p className="hint">
          Or click a row and press A–E. Space leaves it blank, arrows move between rows.
        </p>
      </div>

      <div className="sheet-rows" role="group" aria-label="Answer sheet">
        {contest.problems.map((p, i) => {
          const r = responses[i];
          const state = r === null ? "blank" : r === p.answer ? "right" : "wrong";
          return (
            <div
              key={p.position}
              ref={(el) => {
                rowRefs.current[i] = el;
              }}
              className={`sheet-row${Math.floor(i / 5) % 2 === 0 ? " banded" : ""}${cursor === i ? " current" : ""}`}
              tabIndex={cursor === i ? 0 : -1}
              onFocus={() => setCursor(i)}
              onKeyDown={(e) => onKeyDown(e, i)}
              aria-label={`Problem ${p.position}, ${r ? `answered ${r.toUpperCase()}` : "blank"}`}
            >
              <span className="sheet-num">{p.position}</span>
              <span className="bubbles">
                {LETTERS.map((letter) => (
                  <button
                    key={letter}
                    type="button"
                    tabIndex={-1}
                    className={`bubble${r === letter ? " filled" : ""}${p.answer === letter && r !== null && r !== letter ? " key" : ""}`}
                    aria-pressed={r === letter}
                    aria-label={`Problem ${p.position} answer ${letter.toUpperCase()}`}
                    onClick={() => {
                      set(i, r === letter ? null : letter);
                      setCursor(i);
                    }}
                  >
                    {letter.toUpperCase()}
                  </button>
                ))}
              </span>
              <span className={`mark ${state}`}>{state === "right" ? "✓ right" : state === "wrong" ? "✗ wrong" : "blank"}</span>
              <span className="sheet-rating" title="LIVE difficulty rating">
                {Math.round(p.seed_elo)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
