const API_BASE: string = import.meta.env.VITE_API_BASE ?? "";

export type Letter = "a" | "b" | "c" | "d" | "e";
export type Response = Letter | null;

export interface ContestSummary {
  contest_id: string;
  year: number;
  label: string;
  mean_difficulty: number;
}

export interface ContestProblem {
  position: number;
  answer: Letter;
  seed_elo: number;
  categories: string[];
}

export interface ContestDetail {
  contest_id: string;
  live_url: string;
  problems: ContestProblem[];
}

export interface Sitting {
  id: string;
  contest_id: string;
  taken_on: string | null;
  responses: Response[];
}

export interface Metrics {
  log_loss: number;
  brier: number;
  accuracy: number;
}

export interface Forecast {
  engine: string;
  inputs: {
    contest_id: string;
    year: number;
    taken_on: string | null;
    correct: number;
    wrong: number;
    blank: number;
    score_current_rules: number;
    score_rules_of_the_year: number;
    blank_points_that_year: number;
  }[];
  ability_global: number;
  ability_global_sd: number;
  growth_per_year: number | null;
  growth_sd: number | null;
  topics: {
    category: string;
    ability: number;
    ability_low: number;
    ability_high: number;
    n_answered: number;
    n_correct: number;
  }[];
  difficulty_bucket_accuracy: { bucket: string; attempted: number; correct: number; blank: number; accuracy_attempted: number | null }[];
  difficulty_curve: { difficulty: number; p_correct_if_answered: number; p_answer: number; p_correct: number }[];
  model_weights: Record<string, number>;
  self_check: {
    n_items: number;
    stacked: boolean;
    fold_type?: string;
    note?: string;
    ensemble?: Metrics;
    base_rate?: Metrics;
    members: Record<string, Metrics>;
  };
  contest_forecast: { target_year: number; target_date: string; level_mean: number; level_sd: number; n_contests: number };
  simulation_summary: {
    n_simulations: number;
    mean_score: number;
    median_score: number;
    std_score: number;
    percentiles: Record<string, number>;
    expected_correct: number;
    expected_wrong: number;
    expected_blank: number;
  };
  histogram: { range_low: number; range_high: number; count: number; probability: number }[];
  target_probabilities: { score: number; probability: number }[];
  question_level: {
    position: number;
    typical_difficulty: number;
    p_answer: number;
    p_correct_if_answered: number;
    expected_p_correct: number;
    likely_categories: string[];
  }[];
  counterfactuals: { category: string; current_ability: number; delta_elo: number; baseline_mean_score: number; improved_mean_score: number; score_gain: number }[];
  explanation: string;
}

async function call<T>(path: string, options?: RequestInit): Promise<T> {
  let res: globalThis.Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { headers: { "Content-Type": "application/json" }, ...options });
  } catch {
    throw new Error("Can't reach the prediction engine. If you started the app by hand, check the backend is running.");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const listContests = () => call<ContestSummary[]>("/api/contests");
export const getContest = (id: string) => call<ContestDetail>(`/api/contests/${id}`);
export const loadSittings = () => call<{ attempts: Sitting[] }>("/api/attempts").then((r) => r.attempts);
export const saveSittings = (attempts: Sitting[]) =>
  call<{ saved: number }>("/api/attempts", { method: "PUT", body: JSON.stringify({ attempts }) });

export function forecast(sittings: Sitting[], targets: number[], targetDate: string, nSimulations = 20000) {
  return call<Forecast>("/api/predict", {
    method: "POST",
    body: JSON.stringify({ contests: sittings, n_simulations: nSimulations, targets, target_date: targetDate }),
  });
}

export function contestName(id: string): string {
  const year = id.slice(0, 4);
  const label = id.slice(4);
  if (year === "2021" && (label === "C" || label === "D")) return `Fall 2021 AMC 10${label === "C" ? "A" : "B"}`;
  if (year === "2021") return `Spring 2021 AMC 10${label}`;
  return `${year} AMC 10${label}`;
}

export function scoreOf(responses: Response[], problems: ContestProblem[]) {
  let correct = 0;
  let blank = 0;
  responses.forEach((r, i) => {
    if (r === null) blank += 1;
    else if (r === problems[i]?.answer) correct += 1;
  });
  return { correct, blank, wrong: 25 - correct - blank, score: correct * 6 + blank * 1.5 };
}
