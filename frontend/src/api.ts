export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export interface ContestSummary {
  contest_id: string;
  contest: string;
  year: number;
  label: string;
  num_problems: number;
}

export interface ContestProblem {
  problem_id: string;
  position: number;
  question: string;
  choice_a: string;
  choice_b: string;
  choice_c: string;
  choice_d: string;
  choice_e: string;
  answer: string;
  seed_elo: number;
  categories: string[];
}

export interface ContestDetail {
  contest_id: string;
  problems: ContestProblem[];
}

export interface HistogramBin {
  range_low: number;
  range_high: number;
  count: number;
  probability: number;
}

export interface QuestionLevelPrediction {
  position: number;
  expected_p_correct: number;
  typical_difficulty: number;
  likely_categories: string[];
}

export interface Counterfactual {
  category: string;
  current_ability: number;
  improved_ability: number;
  delta_elo: number;
  baseline_mean_score: number;
  improved_mean_score: number;
  score_gain: number;
}

export interface PredictionResult {
  input_scores: Record<string, number>;
  ability_global: number;
  ability_by_category: Record<string, number>;
  topic_accuracy: Record<string, unknown>[];
  difficulty_bucket_accuracy: Record<string, unknown>[];
  position_accuracy: Record<string, unknown>[];
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
  histogram: HistogramBin[];
  question_level: QuestionLevelPrediction[];
  counterfactuals: Counterfactual[];
  explanation: string;
  n_simulations: number;
  contests_used: string[];
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export function listContests(): Promise<ContestSummary[]> {
  return apiFetch("/api/contests");
}

export function getContest(contestId: string): Promise<ContestDetail> {
  return apiFetch(`/api/contests/${contestId}`);
}

export function predict(
  contests: { contest_id: string; responses: (string | null)[] }[],
  nSimulations = 20000
): Promise<PredictionResult> {
  return apiFetch("/api/predict", {
    method: "POST",
    body: JSON.stringify({ contests, n_simulations: nSimulations }),
  });
}
