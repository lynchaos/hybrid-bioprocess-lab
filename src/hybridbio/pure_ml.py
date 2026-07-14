"""A direct trajectory baseline with no mechanistic structure.

This model deliberately receives only each batch's initial condition, feed
settings, and requested time. It predicts the complete state directly, so it
is a useful comparator for the hybrid model: it has access to the same
historical training batches, but none of the ODE or mass-balance assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor

from .data import Batch
from .mechanistic import KineticParameters

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PureMLConfig:
    """Configuration for the direct trajectory benchmark."""

    n_estimators: int = 200
    max_depth: int = 10
    min_samples_leaf: int = 2
    seed: int = 0


def build_trajectory_matrix(batches: list[Batch], t_end_h: float) -> tuple[Array, Array]:
    """Create direct-prediction rows without exposing any future batch state."""
    if not batches:
        raise ValueError("pure-ML training requires at least one batch")

    feature_parts: list[Array] = []
    target_parts: list[Array] = []
    for batch in batches:
        n_times = len(batch)
        initial_state = np.repeat(batch.y0[None, :], n_times, axis=0)
        feed = np.tile(
            np.array([batch.feed.rate, batch.feed.start_h, batch.feed.S_feed], dtype=np.float64),
            (n_times, 1),
        )
        time_fraction = (batch.t / t_end_h)[:, None]
        feature_parts.append(np.hstack((time_fraction, initial_state, feed)))
        target_parts.append(batch.Y)
    return np.vstack(feature_parts), np.vstack(target_parts)


@dataclass(slots=True)
class PureMLTrajectoryModel:
    """Direct multi-output regressor evaluated as a transparent benchmark."""

    params: KineticParameters
    estimator: Any
    t_end_h: float = 336.0

    def simulate_batch(self, batch: Batch) -> tuple[Array, Array]:
        """Predict a batch trajectory at its observed time grid."""
        features, _ = build_trajectory_matrix([batch], self.t_end_h)
        prediction = np.asarray(self.estimator.predict(features), dtype=np.float64)
        if prediction.shape != batch.Y.shape:
            raise RuntimeError(
                "pure-ML trajectory prediction shape does not match the requested batch grid"
            )
        if not np.all(np.isfinite(prediction)):
            raise RuntimeError("pure-ML trajectory prediction contains non-finite values")
        return batch.t, prediction


def train_pure_ml_trajectory(
    batches: list[Batch],
    params: KineticParameters | None = None,
    config: PureMLConfig | None = None,
    t_end_h: float = 336.0,
) -> PureMLTrajectoryModel:
    """Fit the direct benchmark on complete historical batches only."""
    config = config or PureMLConfig()
    features, targets = build_trajectory_matrix(batches, t_end_h)
    estimator = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.seed,
        n_jobs=1,
    )
    estimator.fit(features, targets)
    return PureMLTrajectoryModel(
        params=params or KineticParameters(),
        estimator=estimator,
        t_end_h=t_end_h,
    )
