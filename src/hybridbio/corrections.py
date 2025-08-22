"""The data-driven half of the hybrid model, behind a stable interface.

The whole point of this module is that `hybridbio.hybrid` does not know or
care *what* is producing the correction. A gradient-boosted tree, an MLP, a
Gaussian process and a constant `1.0` are all valid `CorrectionModel`s and are
swapped without touching a line of the hybrid or mechanistic code.

That is not architectural vanity. In practice the modelling scientist wants to
try six things by Thursday, and every one of those six should be a new class
here and nothing else, anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Array = NDArray[np.float64]


@runtime_checkable
class CorrectionModel(Protocol):
    """A multiplicative correction on the mechanistic specific growth rate.

    Contract, and it is a *contract*:
      * `predict` returns a strictly positive multiplier, one per row.
      * `predict` is pure: no fitting, no mutation, no I/O.
      * A model that has not been fitted must raise, never silently return 1.0
        -- a silent identity correction is the single easiest way to ship a
        "working" hybrid model that is quietly just the mechanistic one.
    """

    def fit(self, X: Array, y: Array) -> CorrectionModel: ...

    def predict(self, X: Array) -> Array: ...

    @property
    def is_fitted(self) -> bool: ...


class NullCorrection:
    """Identity correction. The mechanistic model, wearing a hybrid coat.

    Exists so that "mechanistic only" is a *configuration*, not a code path.
    It is also the honest baseline every hybrid model must beat before anyone
    is allowed to be excited about it.
    """

    def fit(self, X: Array, y: Array) -> NullCorrection:  # noqa: ARG002
        return self

    def predict(self, X: Array) -> Array:
        return np.ones(len(X), dtype=np.float64)

    @property
    def is_fitted(self) -> bool:
        return True


class SklearnCorrection:
    """Wraps any scikit-learn regressor as a bounded multiplicative correction.

    The bounding is the interesting part. A raw regressor is free to predict a
    growth-rate multiplier of -4.2, which is not a bad prediction so much as a
    meaningless one. We clip into `bounds`, which encodes a modelling
    assumption we are willing to defend out loud: the mechanistic model is
    wrong, but it is not wrong by more than a factor of ~2.
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        bounds: tuple[float, float] = (0.5, 1.8),
    ) -> None:
        if bounds[0] <= 0.0:
            raise ValueError(
                "lower bound must be > 0: a non-positive growth multiplier is unphysical"
            )
        if bounds[0] >= bounds[1]:
            raise ValueError(f"invalid bounds: {bounds}")
        self.estimator = estimator if estimator is not None else _default_estimator()
        self.bounds = bounds
        self._fitted = False

    def fit(self, X: Array, y: Array) -> SklearnCorrection:
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
        if len(X) == 0:
            raise ValueError("refusing to fit a correction model on zero rows")
        self.estimator.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: Array) -> Array:
        if not self._fitted:
            raise RuntimeError(
                "SklearnCorrection.predict called before fit. Refusing to return "
                "an identity correction, which would silently degrade this to a "
                "mechanistic model while still reporting as hybrid."
            )
        raw = np.asarray(self.estimator.predict(X), dtype=np.float64)
        return np.clip(raw, *self.bounds)

    @property
    def is_fitted(self) -> bool:
        return self._fitted


def _default_estimator() -> Pipeline:
    """The default correction is an MLP, and the reason is not taste. See below."""
    return mlp_estimator()


def mlp_estimator(hidden: tuple[int, ...] = (16,), alpha: float = 1e-2, seed: int = 0) -> Pipeline:
    """A *smooth* MLP correction -- the drop-in slot where a Torch module goes.

    Smoothness is a hard requirement, not a preference, and this cost me an
    afternoon to learn properly. See `tree_estimator` below.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden,
                    alpha=alpha,
                    max_iter=3000,
                    random_state=seed,
                ),
            ),
        ]
    )


def tree_estimator(n_estimators: int = 180, max_depth: int = 3, seed: int = 0) -> Pipeline:
    """A gradient-boosted tree correction. **Do not put this inside the ODE.**

    Hard-won, and my favourite thing in this repo:

    A tree ensemble is piecewise-constant. Embed one in an ODE right-hand side
    and the derivative becomes *discontinuous* at every split boundary. LSODA,
    being an adaptive solver, then does exactly what it is designed to do --
    detects the discontinuity, shrinks its step size to chase it, and shrinks
    it again, and again. A 14-day fed-batch that integrates in 0.7 s with a
    smooth MLP took over 100 s with this estimator, and it was not a
    performance bug. It was the wrong function class for the job.

    The lesson generalises well beyond this repo: when a learned component
    lives inside a numerical integrator, the *smoothness* of the model class is
    a functional requirement, on exactly the same footing as its accuracy. No
    validation metric on a held-out set will ever tell you this. You find out
    when the solver hangs, and then you have to know why.

    Kept here on purpose, as a usable estimator for offline analysis (feature
    importances, exploratory fitting) and as a standing warning.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "gbr",
                GradientBoostingRegressor(
                    random_state=seed, n_estimators=n_estimators, max_depth=max_depth
                ),
            ),
        ]
    )


def save(model: CorrectionModel, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load(path: Path) -> CorrectionModel:
    model = joblib.load(path)
    if not isinstance(model, CorrectionModel):
        raise TypeError(f"{path} does not contain a CorrectionModel")
    return model
