"""Bayesian binary GLM with independent Gaussian priors, fitted by Fisher
scoring to the posterior mode (MAP) and summarised by a Laplace
approximation (Gaussian centred on the mode).

Link, with an optional guessing floor c:

    P(y = 1) = c + (1 - c) * sigmoid(scale * (X @ beta + offset))

c = 0 is ordinary logistic regression. The Elo item-response curve is the
special case scale = ln(10)/400, offset = -difficulty, X @ beta = ability.

Everything the predictor needs comes out of one fit: the mode, a
posterior covariance to draw parameter samples from (so simulations carry
parameter uncertainty, not just coin-flip noise), and an approximate log
evidence for choosing hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


@dataclass
class GlmFit:
    beta: np.ndarray        # (P,) posterior mode
    cov: np.ndarray         # (P,P) Laplace covariance
    log_evidence: float
    converged: bool

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.multivariate_normal(self.beta, self.cov, size=n, method="cholesky")


def predict_proba(
    X: np.ndarray, offset: np.ndarray, beta: np.ndarray, scale: float = 1.0, guess_floor: float = 0.0
) -> np.ndarray:
    return guess_floor + (1.0 - guess_floor) * _sigmoid(scale * (X @ beta + offset))


def fit_map(
    X: np.ndarray,
    offset: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_var: np.ndarray,
    weights: np.ndarray | None = None,
    scale: float = 1.0,
    guess_floor: float = 0.0,
    max_iter: int = 60,
    tol: float = 1e-7,
) -> GlmFit:
    n, p = X.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    y = np.asarray(y, dtype=float)
    prior_prec = 1.0 / prior_var
    c = guess_floor

    def log_post(beta: np.ndarray) -> float:
        prob = np.clip(predict_proba(X, offset, beta, scale, c), 1e-12, 1 - 1e-12)
        ll = float(np.sum(w * (y * np.log(prob) + (1 - y) * np.log(1 - prob))))
        lp = float(-0.5 * np.sum(prior_prec * (beta - prior_mean) ** 2))
        return ll + lp

    def grad_info(beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = _sigmoid(scale * (X @ beta + offset))
        prob = np.clip(c + (1 - c) * s, 1e-12, 1 - 1e-12)
        dp = (1 - c) * s * (1 - s) * scale          # d prob / d (X@beta)
        g = X.T @ (w * (y - prob) / (prob * (1 - prob)) * dp) - prior_prec * (beta - prior_mean)
        fisher = (X * (w * dp**2 / (prob * (1 - prob)))[:, None]).T @ X + np.diag(prior_prec)
        return g, fisher

    beta = prior_mean.astype(float).copy()
    current = log_post(beta)
    converged = False
    for _ in range(max_iter):
        g, fisher = grad_info(beta)
        step = np.linalg.solve(fisher, g)
        t = 1.0
        while t > 1e-6:                              # step-halving line search
            candidate = beta + t * step
            value = log_post(candidate)
            if value >= current - 1e-12:
                break
            t *= 0.5
        beta, improvement = candidate, value - current
        current = value
        if improvement < tol:
            converged = True
            break

    _, fisher = grad_info(beta)
    cov = np.linalg.inv(fisher)
    cov = (cov + cov.T) / 2.0
    sign, logdet = np.linalg.slogdet(fisher)
    log_prior_norm = float(-0.5 * np.sum(np.log(2 * np.pi * prior_var)))
    log_evidence = current + log_prior_norm + 0.5 * p * np.log(2 * np.pi) - 0.5 * logdet
    return GlmFit(beta=beta, cov=cov, log_evidence=float(log_evidence), converged=converged)
