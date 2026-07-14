"""Scientist-facing diagnostics for the learned growth correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Batch
from .features import FEATURE_NAMES, build_features
from .hybrid import HybridModel


@dataclass(frozen=True, slots=True)
class CorrectionAudit:
    """Descriptive correction range and local feature-perturbation effects."""

    n_observations: int
    feature_names: tuple[str, ...]
    correction_min: float
    correction_max: float
    correction_mean: float
    feature_effects: dict[str, float]


def audit_correction(model: HybridModel, batches: list[Batch]) -> CorrectionAudit:
    """Summarise correction behavior over observed states.

    Each feature effect is the absolute change in mean multiplier after that
    feature is independently replaced by its observed 10th versus 90th
    percentile. It is an audit aid for correlated process variables, not a
    causal feature-importance claim.
    """
    if not batches:
        raise ValueError("correction audit requires at least one observed batch")

    X = np.vstack(
        [build_features(batch.t, batch.Y, model.params, model.t_end_h) for batch in batches]
    )
    multipliers = np.asarray(model.correction.predict(X), dtype=np.float64)
    if not np.all(np.isfinite(multipliers)):
        raise ValueError("correction audit encountered non-finite multipliers")

    effects: dict[str, float] = {}
    for index, name in enumerate(FEATURE_NAMES):
        low, high = np.quantile(X[:, index], [0.1, 0.9])
        low_inputs = X.copy()
        high_inputs = X.copy()
        low_inputs[:, index] = low
        high_inputs[:, index] = high
        effects[name] = float(
            abs(
                np.mean(np.asarray(model.correction.predict(high_inputs), dtype=np.float64))
                - np.mean(np.asarray(model.correction.predict(low_inputs), dtype=np.float64))
            )
        )

    return CorrectionAudit(
        n_observations=len(X),
        feature_names=FEATURE_NAMES,
        correction_min=float(np.min(multipliers)),
        correction_max=float(np.max(multipliers)),
        correction_mean=float(np.mean(multipliers)),
        feature_effects=effects,
    )
