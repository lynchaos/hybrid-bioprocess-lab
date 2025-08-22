"""Training the correction model, and tracking what happened.

The training target deserves a word. We do not train the correction model to
predict the states directly -- we train it to predict the *ratio* between the
growth rate reality implies and the growth rate the mechanism predicts:

    target = mu_observed / mu_mechanistic

This keeps the learned object small, bounded, physically interpretable, and
structurally incapable of breaking a mass balance. A scientist can plot it and
say whether it is plausible. You cannot do that with a black box that emits
dX/dt directly, and the ability to have that conversation is worth more than a
few points of RMSE.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .corrections import CorrectionModel, SklearnCorrection
from .data import Batch
from .evaluation import EvaluationReport, evaluate
from .features import FEATURE_VERSION, build_features
from .hybrid import HybridModel
from .mechanistic import FeedProfile, KineticParameters, specific_growth_rate

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    t_end_h: float = 336.0
    dt_h: float = 6.0
    bounds: tuple[float, float] = (0.5, 1.8)
    seed: int = 0
    smooth_window: int = 9
    smooth_polyorder: int = 2


def _smooth(series: Array, window: int = 9, polyorder: int = 2) -> Array:
    """Savitzky-Golay smoothing, degrading gracefully on short series."""
    from scipy.signal import savgol_filter

    n = len(series)
    if n < 5:
        return series
    w = min(window, n if n % 2 == 1 else n - 1)
    if w <= polyorder:
        return series
    return np.asarray(savgol_filter(series, w, polyorder), dtype=np.float64)


def build_training_matrix(
    batches: list[Batch], p: KineticParameters, cfg: TrainingConfig
) -> tuple[Array, Array]:
    """Assemble (X, y) for the correction model from batch records.

    The observed growth rate is estimated from the cell-density trajectory by
    finite difference, corrected for death and dilution -- i.e. we invert the
    mass balance we *do* trust, to isolate the term we *don't*.
    """
    X_parts: list[Array] = []
    y_parts: list[Array] = []

    for batch in batches:
        t, Y = batch.t, batch.Y
        Xv, V = Y[:, 0], Y[:, 4]

        # Smooth log(Xv) BEFORE differentiating.
        #
        # This single line is the difference between a hybrid model that beats
        # the mechanistic baseline and one that loses to it. The training label
        # is a *derivative* of a noisy measurement, and differentiation is a
        # high-pass filter: 3% assay noise on Xv becomes a swamping fraction of
        # mu_max once you finite-difference it at 6-hour spacing. Fit to that
        # and the correction model faithfully learns the noise, then injects it
        # back into the ODE.
        #
        # Savitzky-Golay preserves the shape of the growth curve while killing
        # the high-frequency assay noise -- which is exactly what a process
        # engineer does by eye before reading a slope off a plot.
        log_Xv = _smooth(np.log(np.maximum(Xv, 1e-9)))
        with np.errstate(divide="ignore", invalid="ignore"):
            dlnXv = np.gradient(log_Xv, t)
        F = np.array([batch.feed.flow(float(ti)) for ti in t])
        dilution = F / np.maximum(V, 1e-9)
        mu_obs = dlnXv + p.kd + dilution

        mu_mech = np.array(
            [
                specific_growth_rate(float(s), float(lac), p)
                for s, lac in zip(Y[:, 1], Y[:, 2], strict=True)
            ]
        )

        # Two filters, and both are load-bearing.
        #
        # (1) The ratio mu_obs/mu_mech is only well-conditioned where mu_mech is
        #     meaningfully non-zero. Divide by a mu_mech of 1e-9 and the label
        #     becomes 4e7, which the regressor will cheerfully chase off a cliff.
        #
        # (2) Labels outside the physical bounds are DROPPED, not clipped.
        #     Clipping them looks harmless and is not: it piles probability mass
        #     up against the boundary, and the model learns that 0.5 is a
        #     popular answer. It then over-suppresses growth, starves the
        #     substrate balance, and -- measured this -- makes nrmse_S three
        #     times worse while still passing every constraint check. A noisy
        #     label is not evidence; it is an absence of evidence, and it should
        #     be discarded rather than flattened into a confident lie.
        well_conditioned = mu_mech > 0.15 * p.mu_max
        ratio = np.divide(mu_obs, mu_mech, out=np.ones_like(mu_obs), where=well_conditioned)
        in_bounds = (ratio >= cfg.bounds[0]) & (ratio <= cfg.bounds[1])
        valid = well_conditioned & in_bounds

        X = build_features(t, Y, p, cfg.t_end_h)
        X_parts.append(X[valid])
        y_parts.append(ratio[valid])

    if not X_parts:
        raise ValueError("no valid training rows were produced")
    return np.vstack(X_parts), np.concatenate(y_parts)


def train_correction(
    batches: list[Batch],
    p: KineticParameters | None = None,
    cfg: TrainingConfig | None = None,
    estimator: Any | None = None,
) -> CorrectionModel:
    p = p or KineticParameters()
    cfg = cfg or TrainingConfig()
    X, y = build_training_matrix(batches, p, cfg)
    model = SklearnCorrection(estimator=estimator, bounds=cfg.bounds)
    return model.fit(X, y)


def train_and_evaluate(
    train_batches: list[Batch],
    test_batches: list[Batch],
    p: KineticParameters | None = None,
    cfg: TrainingConfig | None = None,
    estimator: Any | None = None,
) -> tuple[HybridModel, EvaluationReport, EvaluationReport]:
    """Train, then evaluate against BOTH the hybrid and the mechanistic baseline.

    Returning the baseline report alongside the candidate is not a courtesy. It
    is the only way to answer the question that actually matters -- "is the ML
    layer earning its keep?" -- and the answer is embarrassingly often "no".
    """
    p = p or KineticParameters()
    cfg = cfg or TrainingConfig()

    correction = train_correction(train_batches, p, cfg, estimator)
    hybrid = HybridModel(
        params=p, feed=FeedProfile(), correction=correction, t_end_h=cfg.t_end_h, dt_h=cfg.dt_h
    )
    baseline = HybridModel.mechanistic_only(p)

    return hybrid, evaluate(hybrid, test_batches), evaluate(baseline, test_batches)


# --------------------------------------------------------------------------
# Experiment tracking
# --------------------------------------------------------------------------


@contextlib.contextmanager
def track_run(run_name: str, experiment: str = "hybrid-bioprocess") -> Iterator[RunLogger]:
    """MLflow run context that degrades gracefully to stdout if MLflow is absent.

    Deliberate: a training script that crashes on a laptop because a tracking
    server is unreachable is a training script people stop running. Tracking is
    important. It is not more important than the science.
    """
    try:
        import mlflow

        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            yield RunLogger(mlflow)
    except ImportError:
        yield RunLogger(None)


class RunLogger:
    def __init__(self, mlflow_module: Any | None) -> None:
        self._mlflow = mlflow_module

    def params(self, **kwargs: Any) -> None:
        if self._mlflow:
            self._mlflow.log_params(kwargs)
        else:
            for k, v in kwargs.items():
                print(f"  param  {k}={v}")

    def metrics(self, metrics: dict[str, float]) -> None:
        if self._mlflow:
            self._mlflow.log_metrics(metrics)
        else:
            for k, v in metrics.items():
                print(f"  metric {k}={v:.5f}")

    def tag(self, key: str, value: str) -> None:
        if self._mlflow:
            self._mlflow.set_tag(key, value)
        else:
            print(f"  tag    {key}={value}")

    def feature_contract(self) -> None:
        """Always log the feature version. Always. This is the cheapest
        insurance policy in applied ML."""
        self.tag("feature_version", FEATURE_VERSION)
