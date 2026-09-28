#!/usr/bin/env python3
"""Compare baseline / logistic regression / gradient-boosted trees / our
IRT-hierarchical ability model on rolling-window held-out-contest
prediction, averaged over many synthetic students (see
src/analytics/calibration.py for why synthetic students are used)."""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

# A rolling-window backtest can see a category in the test contest that
# never appeared in the (smaller) training window yet; sklearn's
# MultiLabelBinarizer warns about this on every such fold, which is
# expected behavior (not a bug) but floods stdout, so we silence it here.
warnings.filterwarnings("ignore", message="unknown class.*will be ignored")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from analytics.calibration import generate_synthetic_student, simulate_student_on_contest  # noqa: E402
from analytics.data_access import ContestResponses, build_response_frame, load_problems  # noqa: E402
from models.compare import all_models, evaluate_model  # noqa: E402
from storage import schema  # noqa: E402


def main() -> int:
    conn = schema.connect(REPO_ROOT / "data" / "amc10.sqlite3")
    problems = load_problems(conn)
    categories = sorted({c for cats in problems["categories"] for c in cats})
    contest_ids = sorted(
        problems["contest_id"].unique(),
        key=lambda cid: tuple(problems[problems.contest_id == cid]["recency_rank"].iloc[0]),
    )

    rng = np.random.default_rng(123)
    n_students = 20
    all_results: dict[str, list[dict]] = defaultdict(list)

    for student_idx in range(n_students):
        skill_level = rng.uniform(-400, 400)
        student = generate_synthetic_student(rng, categories, skill_level=skill_level)
        responses = {
            cid: simulate_student_on_contest(rng, student, problems[problems.contest_id == cid])
            for cid in contest_ids
        }
        # rolling-window splits: train on contests[0:i], test on contests[i]
        for i in range(3, len(contest_ids)):  # need a few contests before testing
            train_ids, test_id = contest_ids[:i], contest_ids[i]
            train_df = build_response_frame(problems, [ContestResponses(c, responses[c]) for c in train_ids])
            test_df = build_response_frame(problems, [ContestResponses(test_id, responses[test_id])])
            train_df = train_df[train_df.outcome != "blank"]
            test_df = test_df[test_df.outcome != "blank"]
            if len(train_df) < 10 or len(test_df) < 3:
                continue
            for model in all_models():
                metrics = evaluate_model(model, train_df, test_df)
                all_results[model.name].append(metrics)

    print(f"{'model':25s} {'n_evals':>8s} {'mean_log_loss':>14s} {'mean_brier':>12s} {'mean_accuracy':>14s}")
    summary = {}
    for name, metrics_list in all_results.items():
        mean_ll = float(np.mean([m["log_loss"] for m in metrics_list]))
        mean_brier = float(np.mean([m["brier_score"] for m in metrics_list]))
        mean_acc = float(np.mean([m["accuracy"] for m in metrics_list]))
        summary[name] = {"n_evals": len(metrics_list), "mean_log_loss": mean_ll, "mean_brier_score": mean_brier, "mean_accuracy": mean_acc}
        print(f"{name:25s} {len(metrics_list):8d} {mean_ll:14.4f} {mean_brier:12.4f} {mean_acc:14.4f}")

    out_path = REPO_ROOT / "data" / "model_comparison_report.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFull report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
