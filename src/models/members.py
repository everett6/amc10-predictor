"""The individual correctness models that the ensemble combines.

Each member answers one question: given that the student ATTEMPTS a
problem with this difficulty, topic mix and position, how likely is a
correct answer? Blanks are a separate decision (models/attempt_model.py).

Members
-------
irt_hier     Bayesian hierarchical Elo/IRT: overall ability + one offset
             per topic category (shrunk toward zero), a per-sitting
             "good day / bad day" effect, optional growth over time.
irt_guess    Same, with a guessing floor: an attempted multiple-choice
             answer is right at least 20% of the time, so one lucky hit on
             a very hard problem moves the ability estimate much less.
irt_flat     Same, with a shallower response curve (discrimination 0.6),
             for students whose results depend less on rated difficulty
             than the Elo curve assumes.
irt_global   No topic offsets at all: the bias-free, lowest-variance choice
             when there is little data.
logit_feat   Logistic regression that learns its own difficulty slope and
             a position effect (fatigue / time pressure late in the paper).
isotonic     Non-parametric monotone curve of accuracy against difficulty
             (pool-adjacent-violators), blended with irt_global.
gbt          Gradient-boosted trees on difficulty, position and topics.
             Only used with at least 60 attempted problems.

Every member exposes the same three operations: fit, predict (a point
probability, used for cross-validation and stacking) and draw/predict_with
(posterior or bootstrap parameter draws, used by the simulation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from models.bayes_glm import GlmFit, _sigmoid, fit_map
from simulation.contest_model import N_CATEGORIES

ELO_K = np.log(10.0) / 400.0          # logit units per Elo point
ABILITY_PRIOR_MEAN = 1400.0           # centre of AMC 10 problem ratings
ABILITY_PRIOR_SD = 500.0              # deliberately wide
GROWTH_PRIOR_SD = 150.0               # Elo per year
DEFAULT_DAY_SD = 60.0                 # sitting-to-sitting swing when it can't be estimated
TAU_GRID = (40.0, 100.0, 160.0, 240.0)   # candidate sd of topic offsets
DAY_SD_GRID = (20.0, 60.0, 100.0)        # candidate sd of per-sitting effects


@dataclass
class Items:
    """A flat batch of (student, problem) rows."""

    difficulty: np.ndarray              # (n,)
    cat: np.ndarray                     # (n, 15) category weights, rows sum to 1 (or 0)
    position: np.ndarray                # (n,) 1..25
    attempt: np.ndarray                 # (n,) sitting index
    t_years: np.ndarray                 # (n,) sitting time minus target date, in years (<= 0); 0 if unknown
    y: np.ndarray | None = None         # (n,) 1 = correct, 0 = wrong

    def __len__(self) -> int:
        return len(self.difficulty)

    def subset(self, mask: np.ndarray) -> "Items":
        return Items(
            self.difficulty[mask], self.cat[mask], self.position[mask], self.attempt[mask],
            self.t_years[mask], None if self.y is None else self.y[mask],
        )


def has_usable_dates(items: Items) -> bool:
    """Growth is only fitted when sittings span at least 30 days."""
    return float(np.ptp(items.t_years)) >= 30.0 / 365.25 if len(items) else False


# --------------------------------------------------------------------------
# Bayesian IRT family
# --------------------------------------------------------------------------
@dataclass
class IrtMember:
    name: str
    guess_floor: float = 0.0
    discrimination: float = 1.0
    use_categories: bool = True

    fit_: GlmFit | None = field(default=None, repr=False)
    tau_: float = 0.0
    day_sd_: float = DEFAULT_DAY_SD
    growth_: bool = False
    attempts_: list[int] = field(default_factory=list)
    hyper_: tuple[float, float] | None = None     # frozen (tau, day_sd) for CV refits

    @property
    def scale(self) -> float:
        return ELO_K * self.discrimination

    # column layout: [global][growth?][15 categories?][one per training sitting if >= 2]
    def _layout(self) -> tuple[int | None, slice | None, slice | None, int]:
        col = 1
        growth_col = None
        if self.growth_:
            growth_col, col = col, col + 1
        cat_cols = None
        if self.use_categories:
            cat_cols, col = slice(col, col + N_CATEGORIES), col + N_CATEGORIES
        att_cols = None
        if len(self.attempts_) >= 2:
            att_cols, col = slice(col, col + len(self.attempts_)), col + len(self.attempts_)
        return growth_col, cat_cols, att_cols, col

    def _design(self, items: Items) -> np.ndarray:
        growth_col, cat_cols, att_cols, p = self._layout()
        X = np.zeros((len(items), p))
        X[:, 0] = 1.0
        if growth_col is not None:
            X[:, growth_col] = items.t_years
        if cat_cols is not None:
            X[:, cat_cols] = items.cat
        if att_cols is not None:
            for j, a in enumerate(self.attempts_):
                X[items.attempt == a, att_cols.start + j] = 1.0
        return X

    def _prior(self, tau: float, day_sd: float) -> tuple[np.ndarray, np.ndarray]:
        growth_col, cat_cols, att_cols, p = self._layout()
        mean, var = np.zeros(p), np.zeros(p)
        mean[0], var[0] = ABILITY_PRIOR_MEAN, ABILITY_PRIOR_SD**2
        if growth_col is not None:
            var[growth_col] = GROWTH_PRIOR_SD**2
        if cat_cols is not None:
            var[cat_cols] = tau**2
        if att_cols is not None:
            var[att_cols] = day_sd**2
        return mean, var

    def fit(self, items: Items) -> None:
        self.attempts_ = sorted(int(a) for a in np.unique(items.attempt))
        self.growth_ = has_usable_dates(items)
        X = self._design(items)
        offset = -items.difficulty
        if self.hyper_ is not None:
            grid = [self.hyper_]
        else:
            taus = TAU_GRID if self.use_categories else (0.0,)
            # The sitting-to-sitting sd is only identifiable with 3+ sittings.
            days = DAY_SD_GRID if len(self.attempts_) >= 3 else (DEFAULT_DAY_SD,)
            grid = [(t, d) for t in taus for d in days]
        best: tuple[float, GlmFit, float, float] | None = None
        for tau, day_sd in grid:
            mean, var = self._prior(tau, day_sd)
            fit = fit_map(X, offset, items.y, mean, var, scale=self.scale, guess_floor=self.guess_floor)
            if best is None or fit.log_evidence > best[0]:
                best = (fit.log_evidence, fit, tau, day_sd)
        assert best is not None
        _, self.fit_, self.tau_, self.day_sd_ = best

    def frozen_copy(self) -> "IrtMember":
        """Unfitted copy that reuses this fit's hyperparameters (cheap CV refits)."""
        return IrtMember(self.name, self.guess_floor, self.discrimination, self.use_categories,
                         hyper_=(self.tau_, self.day_sd_))

    def _prob(self, theta: np.ndarray, difficulty: np.ndarray) -> np.ndarray:
        return self.guess_floor + (1 - self.guess_floor) * _sigmoid(self.scale * (theta - difficulty))

    def predict(self, items: Items) -> np.ndarray:
        assert self.fit_ is not None
        X = self._design(items)
        mu = X @ self.fit_.beta
        var = np.einsum("ij,jk,ik->i", X, self.fit_.cov, X)
        unseen = ~np.isin(items.attempt, self.attempts_) | (len(self.attempts_) < 2)
        var = var + np.where(unseen, self.day_sd_**2, 0.0)
        # Logistic-normal mean: shrink the logit by the predictive sd.
        kappa = 1.0 / np.sqrt(1.0 + np.pi * (self.scale**2) * var / 8.0)
        return self.guess_floor + (1 - self.guess_floor) * _sigmoid(kappa * self.scale * (mu - items.difficulty))

    def draw(self, n: int, rng: np.random.Generator) -> dict:
        assert self.fit_ is not None
        return {"beta": self.fit_.draw(n, rng), "day": rng.normal(0.0, self.day_sd_, size=n)}

    def predict_with(self, params: dict, difficulty: np.ndarray, cat: np.ndarray, position: np.ndarray) -> np.ndarray:
        """difficulty (n,25), cat (n,25,15) -> P(correct | attempted) (n,25),
        for a sitting at the target date (so the growth term is zero)."""
        _, cat_cols, _, _ = self._layout()
        beta = params["beta"]
        theta = beta[:, [0]] + params["day"][:, None]
        if cat_cols is not None:
            theta = theta + np.einsum("npc,nc->np", cat, beta[:, cat_cols])
        return self._prob(theta, difficulty)

    # reporting helpers ------------------------------------------------------
    def ability_summary(self) -> dict:
        assert self.fit_ is not None
        growth_col, cat_cols, _, _ = self._layout()
        beta, cov = self.fit_.beta, self.fit_.cov
        out = {
            "global_mean": float(beta[0]), "global_sd": float(np.sqrt(cov[0, 0])),
            "growth_per_year": None if growth_col is None else float(beta[growth_col]),
            "growth_sd": None if growth_col is None else float(np.sqrt(cov[growth_col, growth_col])),
            "tau": self.tau_, "day_sd": self.day_sd_, "categories": {},
        }
        if cat_cols is not None:
            for i in range(N_CATEGORIES):
                j = cat_cols.start + i
                var = cov[0, 0] + cov[j, j] + 2 * cov[0, j]
                out["categories"][i] = (float(beta[0] + beta[j]), float(np.sqrt(max(var, 0.0))))
        return out


# --------------------------------------------------------------------------
# Feature logistic regression (learned difficulty slope + position effect)
# --------------------------------------------------------------------------
@dataclass
class LogitFeatureMember:
    name: str = "logit_feat"
    fit_: GlmFit | None = field(default=None, repr=False)

    @staticmethod
    def _design_flat(difficulty: np.ndarray, cat: np.ndarray, position: np.ndarray) -> np.ndarray:
        return np.column_stack([
            np.ones_like(difficulty), (difficulty - 1400.0) / 400.0, (position - 13.0) / 12.0, cat,
        ])

    def fit(self, items: Items) -> None:
        X = self._design_flat(items.difficulty, items.cat, items.position)
        # Slope prior is centred on the Elo curve's own slope (ln 10 per 400 points).
        mean = np.concatenate([[0.0, -np.log(10.0), 0.0], np.zeros(N_CATEGORIES)])
        var = np.concatenate([[3.0**2, 1.5**2, 1.0**2], np.full(N_CATEGORIES, 0.75**2)])
        self.fit_ = fit_map(X, np.zeros(len(items)), items.y, mean, var)

    def frozen_copy(self) -> "LogitFeatureMember":
        return LogitFeatureMember(self.name)

    def predict(self, items: Items) -> np.ndarray:
        assert self.fit_ is not None
        X = self._design_flat(items.difficulty, items.cat, items.position)
        mu = X @ self.fit_.beta
        var = np.einsum("ij,jk,ik->i", X, self.fit_.cov, X)
        return _sigmoid(mu / np.sqrt(1.0 + np.pi * var / 8.0))

    def draw(self, n: int, rng: np.random.Generator) -> dict:
        assert self.fit_ is not None
        return {"beta": self.fit_.draw(n, rng)}

    def predict_with(self, params: dict, difficulty: np.ndarray, cat: np.ndarray, position: np.ndarray) -> np.ndarray:
        b = params["beta"]
        pos = (np.broadcast_to(position, difficulty.shape) - 13.0) / 12.0
        eta = b[:, [0]] + b[:, [1]] * (difficulty - 1400.0) / 400.0 + b[:, [2]] * pos
        eta = eta + np.einsum("npc,nc->np", cat, b[:, 3:])
        return _sigmoid(eta)


# --------------------------------------------------------------------------
# Isotonic regression in difficulty
# --------------------------------------------------------------------------
def pava_decreasing(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators fit of a non-increasing step function.
    Returns (sorted x, fitted values at those x)."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order].astype(float)
    values: list[float] = []
    counts: list[int] = []
    for v in ys:
        values.append(float(v)); counts.append(1)
        while len(values) > 1 and values[-2] < values[-1]:     # violates "non-increasing"
            total = counts[-2] + counts[-1]
            merged = (values[-2] * counts[-2] + values[-1] * counts[-1]) / total
            values[-2:] = [merged]; counts[-2:] = [total]
    return xs, np.repeat(values, counts)


@dataclass
class IsotonicMember:
    name: str = "isotonic"
    n_boot: int = 30
    blend: float = 0.25                 # weight on the parametric irt_global curve
    grid_: np.ndarray = field(default_factory=lambda: np.arange(200.0, 3001.0, 25.0), repr=False)
    curve_: np.ndarray | None = field(default=None, repr=False)
    boot_curves_: np.ndarray | None = field(default=None, repr=False)

    def _curve(self, items: Items) -> np.ndarray:
        base = IrtMember("irt_global", use_categories=False)
        base.fit(items)
        assert base.fit_ is not None
        parametric = _sigmoid(ELO_K * (base.fit_.beta[0] - self.grid_))
        xs, fitted = pava_decreasing(items.difficulty, items.y)
        # Step function: carry the nearest fitted block (constant beyond the data range).
        iso = np.interp(self.grid_, xs, fitted)
        return np.clip((1 - self.blend) * iso + self.blend * parametric, 0.02, 0.98)

    def fit(self, items: Items, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng(0)
        self.curve_ = self._curve(items)
        n = len(items)
        curves = []
        for _ in range(self.n_boot):
            mask = rng.integers(0, n, size=n)
            curves.append(self._curve(items.subset(mask)))
        self.boot_curves_ = np.vstack(curves)

    def frozen_copy(self) -> "IsotonicMember":
        return IsotonicMember(self.name, n_boot=1, blend=self.blend)

    def predict(self, items: Items) -> np.ndarray:
        assert self.curve_ is not None
        return np.interp(items.difficulty, self.grid_, self.curve_)

    def draw(self, n: int, rng: np.random.Generator) -> dict:
        assert self.boot_curves_ is not None
        return {"curve": rng.integers(0, len(self.boot_curves_), size=n)}

    def predict_with(self, params: dict, difficulty: np.ndarray, cat: np.ndarray, position: np.ndarray) -> np.ndarray:
        assert self.boot_curves_ is not None
        out = np.empty_like(difficulty)
        for k in np.unique(params["curve"]):
            rows = params["curve"] == k
            out[rows] = np.interp(difficulty[rows], self.grid_, self.boot_curves_[k])
        return out


# --------------------------------------------------------------------------
# Gradient-boosted trees
# --------------------------------------------------------------------------
GBT_MIN_ITEMS = 60


def _gbt_features(difficulty: np.ndarray, cat: np.ndarray, position: np.ndarray) -> np.ndarray:
    return np.column_stack([difficulty / 1000.0, position / 25.0, cat])


@dataclass
class GbtMember:
    name: str = "gbt"
    n_boot: int = 6
    models_: list = field(default_factory=list, repr=False)
    constant_: float | None = None

    def _fit_one(self, items: Items, seed: int):
        from sklearn.ensemble import GradientBoostingClassifier

        clf = GradientBoostingClassifier(
            n_estimators=40, max_depth=2, learning_rate=0.08, subsample=0.8, random_state=seed
        )
        clf.fit(_gbt_features(items.difficulty, items.cat, items.position), items.y.astype(int))
        return clf

    def fit(self, items: Items, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng(0)
        self.models_, self.constant_ = [], None
        if len(np.unique(items.y)) < 2:
            self.constant_ = float(np.clip(items.y.mean(), 0.02, 0.98))
            return
        self.models_.append(self._fit_one(items, 0))
        n = len(items)
        for b in range(self.n_boot):
            sub = items.subset(rng.integers(0, n, size=n))
            if len(np.unique(sub.y)) == 2:
                self.models_.append(self._fit_one(sub, b + 1))

    def frozen_copy(self) -> "GbtMember":
        return GbtMember(self.name, n_boot=0)

    def _proba(self, model, feats: np.ndarray) -> np.ndarray:
        return np.clip(model.predict_proba(feats)[:, 1], 0.02, 0.98)

    def predict(self, items: Items) -> np.ndarray:
        if self.constant_ is not None:
            return np.full(len(items), self.constant_)
        return self._proba(self.models_[0], _gbt_features(items.difficulty, items.cat, items.position))

    def draw(self, n: int, rng: np.random.Generator) -> dict:
        return {"model": rng.integers(0, max(len(self.models_), 1), size=n)}

    def predict_with(self, params: dict, difficulty: np.ndarray, cat: np.ndarray, position: np.ndarray) -> np.ndarray:
        if self.constant_ is not None:
            return np.full(difficulty.shape, self.constant_)
        out = np.empty_like(difficulty)
        pos = np.broadcast_to(position, difficulty.shape)
        for k in np.unique(params["model"]):
            rows = params["model"] == k
            feats = _gbt_features(difficulty[rows].ravel(), cat[rows].reshape(-1, N_CATEGORIES), pos[rows].ravel())
            out[rows] = self._proba(self.models_[k], feats).reshape(difficulty[rows].shape)
        return out
