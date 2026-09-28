"""A common fit/predict interface for comparing several correctness
models against each other on held-out contests: a naive baseline, plain
logistic regression, a gradient-boosted tree, and our IRT-inspired
hierarchical ability model.

All models predict P(correct) for each row of a problems-with-response
DataFrame (same shape as analytics.data_access.build_response_frame,
minus blanks) given a *training* set of the student's other attempts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

from models.ability_model import fit_ability_profile, p_correct


class CorrectnessModel(Protocol):
    name: str

    def fit(self, train_df: pd.DataFrame) -> None: ...
    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray: ...


def _binary_labels(df: pd.DataFrame) -> np.ndarray:
    return (df["outcome"] == "correct").astype(int).to_numpy()


@dataclass
class BaselineModel:
    """Predicts the student's overall historical accuracy rate for
    every problem, ignoring difficulty and topic entirely -- the floor
    any real model needs to beat."""

    name: str = "baseline_global_rate"
    rate: float = 0.5

    def fit(self, train_df: pd.DataFrame) -> None:
        y = _binary_labels(train_df)
        self.rate = float(y.mean()) if len(y) else 0.5

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        return np.full(len(test_df), self.rate)


def _feature_matrix(df: pd.DataFrame, mlb: MultiLabelBinarizer, fit_mlb: bool) -> np.ndarray:
    elo = df["seed_elo"].to_numpy().reshape(-1, 1) / 1000.0
    position = df["position"].to_numpy().reshape(-1, 1) / 25.0
    cats = df["categories"].apply(lambda v: v if isinstance(v, list) else [])
    cat_matrix = mlb.fit_transform(cats) if fit_mlb else mlb.transform(cats)
    return np.hstack([elo, position, cat_matrix])


@dataclass
class LogisticModel:
    name: str = "logistic_regression"

    def __post_init__(self) -> None:
        self.mlb = MultiLabelBinarizer()
        self.clf = LogisticRegression(max_iter=1000, C=1.0)

    def fit(self, train_df: pd.DataFrame) -> None:
        X = _feature_matrix(train_df, self.mlb, fit_mlb=True)
        y = _binary_labels(train_df)
        if len(np.unique(y)) < 2:
            self._constant = float(y.mean()) if len(y) else 0.5
        else:
            self._constant = None
            self.clf.fit(X, y)

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        if self._constant is not None:
            return np.full(len(test_df), self._constant)
        X = _feature_matrix(test_df, self.mlb, fit_mlb=False)
        return self.clf.predict_proba(X)[:, 1]


@dataclass
class GradientBoostedTreeModel:
    name: str = "gradient_boosted_trees"

    def __post_init__(self) -> None:
        self.mlb = MultiLabelBinarizer()
        self.clf = GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.1)

    def fit(self, train_df: pd.DataFrame) -> None:
        X = _feature_matrix(train_df, self.mlb, fit_mlb=True)
        y = _binary_labels(train_df)
        if len(np.unique(y)) < 2:
            self._constant = float(y.mean()) if len(y) else 0.5
        else:
            self._constant = None
            self.clf.fit(X, y)

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        if self._constant is not None:
            return np.full(len(test_df), self._constant)
        X = _feature_matrix(test_df, self.mlb, fit_mlb=False)
        return self.clf.predict_proba(X)[:, 1]


@dataclass
class IrtHierarchicalModel:
    name: str = "irt_hierarchical"
    shrinkage_k: float = 6.0
    recency_half_life: float = 4.0

    def fit(self, train_df: pd.DataFrame) -> None:
        self.profile = fit_ability_profile(
            train_df, shrinkage_k=self.shrinkage_k, recency_half_life=self.recency_half_life
        )

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        probs = []
        for _, row in test_df.iterrows():
            cats = row["categories"] or []
            ability = (
                float(np.mean([self.profile.ability_for(c) for c in cats]))
                if cats
                else self.profile.theta_global
            )
            probs.append(p_correct(ability, row["seed_elo"]))
        return np.array(probs)


def all_models() -> list:
    return [BaselineModel(), LogisticModel(), GradientBoostedTreeModel(), IrtHierarchicalModel()]


def evaluate_model(model, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Fit on train_df, predict on test_df (both non-blank rows), return
    log-loss, Brier score, and accuracy against actual outcomes."""
    model.fit(train_df)
    p = np.clip(model.predict_proba(test_df), 1e-6, 1 - 1e-6)
    y = _binary_labels(test_df)
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    accuracy = float(np.mean((p >= 0.5).astype(int) == y))
    return {"model": model.name, "log_loss": log_loss, "brier_score": brier, "accuracy": accuracy, "n_test": len(test_df)}
