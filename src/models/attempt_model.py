"""Will the student answer a problem at all, or leave it blank?

The correctness models are fitted on answered problems only, so they
describe P(correct | answered). To simulate a whole paper we also need
P(answered), which depends on how far the problem is above the student's
level and on how late in the paper it sits (time runs out):

    logit P(answered) = b0 + b1 * (difficulty - ability) / 400 + b2 * (position - 13) / 12

fitted as a Bayesian logistic regression on every problem in the
student's history. The priors encode a generic test-taker (answers most
things, less so when far out of depth or late in the paper) and matter
only while the student's own blank pattern is thin; they are stated
assumptions, not population estimates.

(v1 instead drew "correct" for every problem first and then split the
misses into blank/wrong, which let the student score on problems they
would in practice have skipped.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.bayes_glm import GlmFit, _sigmoid, fit_map

PRIOR_MEAN = np.array([2.0, -1.5, -0.5])
PRIOR_VAR = np.array([1.5, 1.5, 1.0]) ** 2


@dataclass
class AttemptModel:
    fit_: GlmFit
    n_blank: int
    n_total: int

    @staticmethod
    def design(gap: np.ndarray, position: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones_like(gap), gap / 400.0, (position - 13.0) / 12.0])

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.fit_.draw(n, rng)

    def predict_with(self, gamma: np.ndarray, gap: np.ndarray, position: np.ndarray) -> np.ndarray:
        """gamma (n,3), gap (n,25) -> P(answered) (n,25)."""
        pos = (np.broadcast_to(position, gap.shape) - 13.0) / 12.0
        return _sigmoid(gamma[:, [0]] + gamma[:, [1]] * gap / 400.0 + gamma[:, [2]] * pos)


def fit_attempt_model(difficulty: np.ndarray, ability: np.ndarray, position: np.ndarray, answered: np.ndarray) -> AttemptModel:
    X = AttemptModel.design(difficulty - ability, position)
    fit = fit_map(X, np.zeros(len(X)), answered.astype(float), PRIOR_MEAN, PRIOR_VAR)
    return AttemptModel(fit, int((answered == 0).sum()), int(len(answered)))
