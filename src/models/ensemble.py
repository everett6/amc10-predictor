"""Stacked ensemble of the correctness models in models/members.py.

Stacking procedure
------------------
1. Split the student's attempted problems into folds (whole sittings when
   there are four or more, otherwise five random folds).
2. For each fold, refit every member on the other folds and predict the
   held-out answers. That gives one out-of-fold probability per member
   per answered problem, on data the member did not see.
3. Choose non-negative weights summing to 1 that maximise the likelihood
   of the student's real right/wrong outcomes under the weighted average
   of the members (EM for a mixture), with a Dirichlet prior that favours
   the hierarchical IRT model. With one sitting the prior dominates; with
   several sittings the student's own data decides.

The out-of-fold predictions double as a self-check on real data: the
log-loss, Brier score and accuracy reported to the user are measured on
their own answers, each predicted by models that had not seen it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from models.members import (
    GBT_MIN_ITEMS, GbtMember, IrtMember, IsotonicMember, Items, LogitFeatureMember,
)
from simulation.contest_model import category_weights

PRIOR_WEIGHTS = {
    "irt_hier": 0.35, "irt_guess": 0.15, "irt_flat": 0.12, "irt_global": 0.13,
    "logit_feat": 0.15, "isotonic": 0.05, "gbt": 0.05,
}
PRIOR_STRENGTH = 20.0        # pseudo-observations behind the prior weights
MIN_ITEMS_FOR_STACKING = 15


def items_from_frame(response_df: pd.DataFrame, target_date=None, attempted_only: bool = True) -> Items:
    """Convert a response frame (analytics.data_access.build_response_frame)
    into model inputs. Rows with a blank are dropped unless attempted_only
    is False (the attempt model needs them)."""
    df = response_df[response_df["outcome"] != "blank"] if attempted_only else response_df
    dated = "taken_on" in df.columns and df["taken_on"].notna().all() and target_date is not None and len(df) > 0
    if dated:
        t_years = np.array([(d - target_date).days / 365.25 for d in df["taken_on"]], dtype=float)
    else:
        t_years = np.zeros(len(df))
    return Items(
        difficulty=df["seed_elo"].to_numpy(dtype=float),
        cat=np.vstack([category_weights(c) for c in df["categories"]]) if len(df) else np.zeros((0, 15)),
        position=df["position"].to_numpy(dtype=float),
        attempt=df["attempt_idx"].to_numpy(dtype=int) if "attempt_idx" in df.columns else np.zeros(len(df), dtype=int),
        t_years=t_years,
        y=(df["outcome"] == "correct").to_numpy(dtype=float),
    )


def default_members(n_items: int) -> list:
    members: list = [
        IrtMember("irt_hier"),
        IrtMember("irt_guess", guess_floor=0.2),
        IrtMember("irt_flat", discrimination=0.6),
        IrtMember("irt_global", use_categories=False),
        LogitFeatureMember(),
        IsotonicMember(),
    ]
    if n_items >= GBT_MIN_ITEMS:
        try:
            import sklearn  # noqa: F401
            members.append(GbtMember())
        except ImportError:          # the packaged desktop build may omit sklearn
            pass
    return members


def _folds(items: Items, rng: np.random.Generator) -> list[np.ndarray]:
    n = len(items)
    sittings = np.unique(items.attempt)
    if len(sittings) >= 4:
        groups = rng.permutation(sittings)
        k = min(len(groups), 8)
        assign = {int(g): i % k for i, g in enumerate(groups)}
        fold_of = np.array([assign[int(a)] for a in items.attempt])
    else:
        k = 5
        fold_of = rng.permutation(n) % k
    return [fold_of == i for i in range(k)]


def stack_weights(oof: np.ndarray, y: np.ndarray, prior: np.ndarray, strength: float = PRIOR_STRENGTH) -> np.ndarray:
    """MAP mixture weights by EM. oof is (n, M) member probabilities."""
    lik = np.where(y[:, None] == 1, oof, 1 - oof)
    lik = np.clip(lik, 1e-9, None)
    w = prior.copy()
    alpha = prior * strength
    for _ in range(500):
        resp = lik * w
        resp /= resp.sum(axis=1, keepdims=True)
        new = (resp.sum(axis=0) + alpha) / (len(y) + alpha.sum())
        if np.max(np.abs(new - w)) < 1e-8:
            w = new
            break
        w = new
    return w


def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
    }


@dataclass
class Ensemble:
    members: list
    weights: np.ndarray
    cv_report: dict = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.members]

    def member(self, name: str):
        return self.members[self.names.index(name)]

    def predict(self, items: Items) -> np.ndarray:
        return np.column_stack([m.predict(items) for m in self.members]) @ self.weights


def fit_ensemble(items: Items, rng: np.random.Generator | None = None, members: list | None = None) -> Ensemble:
    rng = rng or np.random.default_rng(0)
    members = members if members is not None else default_members(len(items))
    for m in members:
        if isinstance(m, (IsotonicMember, GbtMember)):
            m.fit(items, rng)
        else:
            m.fit(items)

    prior = np.array([PRIOR_WEIGHTS.get(m.name, 0.05) for m in members])
    prior = prior / prior.sum()
    y = items.y
    assert y is not None
    n = len(items)
    report: dict = {"n_items": n, "stacked": False, "members": {}}

    if n < MIN_ITEMS_FOR_STACKING or len(np.unique(y)) < 2:
        report["note"] = "Too few answered problems (or no mix of right and wrong) to cross-validate; using default weights."
        return Ensemble(members, prior, report)

    folds = _folds(items, rng)
    oof = np.full((n, len(members)), np.nan)
    base = np.full(n, np.nan)
    for held in folds:
        train = items.subset(~held)
        test = items.subset(held)
        if len(test) == 0 or len(train) < 5:
            continue
        base[held] = np.clip((train.y.sum() + 1.0) / (len(train) + 2.0), 0.02, 0.98)
        for j, m in enumerate(members):
            clone = m.frozen_copy()
            if isinstance(clone, (IsotonicMember, GbtMember)):
                clone.fit(train, rng)
            else:
                clone.fit(train)
            oof[held, j] = clone.predict(test)
    ok = ~np.isnan(oof).any(axis=1)
    weights = stack_weights(oof[ok], y[ok], prior)

    report.update({
        "stacked": True,
        "n_folds": len(folds),
        "fold_type": "by sitting" if len(np.unique(items.attempt)) >= 4 else "random 5-fold",
        "ensemble": _metrics(oof[ok] @ weights, y[ok]),
        "base_rate": _metrics(base[ok], y[ok]),
        "members": {m.name: _metrics(oof[ok, j], y[ok]) for j, m in enumerate(members)},
        "note": (
            "Measured on the student's own answers, each predicted by models fitted without it. "
            "The ensemble row is slightly optimistic because its weights were tuned on these same predictions."
        ),
    })
    return Ensemble(members, weights, report)
