"""Rollout training for the correction model.

The standard trainer learns from *observed* states but deploys the model on its
own *predicted* states. That is a train/serve distribution shift: a small error
in Xv feeds back into the feature vector, which feeds back into the growth rate,
which changes the next Xv. Over a 14-day fed-batch these errors accumulate.

Rollout training unrolls the hybrid model and fits the correction on simulated
trajectories, so the model learns to be robust to its own predictions. It is
slower and noisier than one-step training, but it closes the feedback loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .constraints import check_trajectory
from .corrections import CorrectionModel, SklearnCorrection
from .data import Batch
from .evaluation import EvaluationReport, evaluate
from .features import build_features
from .hybrid import HybridModel
from .mechanistic import FeedProfile, KineticParameters
from .training import TrainingConfig

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    """Configuration for rollout training."""

    t_end_h: float = 336.0
    dt_h: float = 6.0
    bounds: tuple[float, float] = (0.5, 1.8)
    seed: int = 0
    smooth_window: int = 9
    smooth_polyorder: int = 2
    n_rollout_steps: int = 4
    rollout_weight: float = 0.5


def _build_targets(
    t: Array,
    Y: Array,
    p: KineticParameters,
    cfg: RolloutConfig,
    feed: FeedProfile,
) -> tuple[Array, Array]:
    """Build (X, y) targets from a single trajectory.

    X comes from the trajectory states; y comes from inverting the trusted mass
    balance to recover the growth-rate correction that produced these states.
    This is the same label construction as the one-step trainer, but the states
    can be either observed or simulated.
    """
    from .training import _smooth

    Xv, V = Y[:, 0], Y[:, 4]
    log_Xv = _smooth(np.log(np.maximum(Xv, 1e-9)), cfg.smooth_window, cfg.smooth_polyorder)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlnXv = np.gradient(log_Xv, t)

    F = np.array([feed.flow(float(ti)) for ti in t])
    dilution = F / np.maximum(V, 1e-9)
    mu_obs = dlnXv + p.kd + dilution

    from .mechanistic import specific_growth_rate

    mu_mech = np.array(
        [
            specific_growth_rate(float(s), float(lac), p)
            for s, lac in zip(Y[:, 1], Y[:, 2], strict=True)
        ],
        dtype=np.float64,
    )

    well_conditioned = mu_mech > 0.15 * p.mu_max
    ratio = np.divide(mu_obs, mu_mech, out=np.ones_like(mu_obs), where=well_conditioned)
    in_bounds = (ratio >= cfg.bounds[0]) & (ratio <= cfg.bounds[1])
    valid = well_conditioned & in_bounds

    X = build_features(t, Y, p, cfg.t_end_h)
    return X[valid], ratio[valid]


def _rollout_once(
    model: HybridModel,
    batch: Batch,
    cfg: RolloutConfig,
) -> tuple[Array, Array]:
    """Unroll the model over one batch and return training rows from the simulation."""
    t, Y = model.simulate_batch(batch)
    report = check_trajectory(t, Y, model.params)
    if not report.ok:
        # An inadmissible rollout is not a training example. It is a broken model.
        return np.empty((0, 0)), np.empty(0)
    return _build_targets(t, Y, model.params, cfg, batch.feed)


def train_correction_rollout(
    batches: list[Batch],
    p: KineticParameters | None = None,
    cfg: RolloutConfig | None = None,
    estimator: Any | None = None,
) -> CorrectionModel:
    """Train a correction model on a mix of observed and rollout targets.

    The algorithm:
      1. Fit a one-step correction on observed data (warm start).
      2. Build a hybrid model with that correction.
      3. For each batch, unroll the hybrid model to get simulated targets.
      4. Mix observed targets and simulated targets by `rollout_weight`.
      5. Refit the correction on the mixed dataset.

    Step 5 can be iterated `n_rollout_steps` times.
    """
    from .training import train_correction

    p = p or KineticParameters()
    cfg = cfg or RolloutConfig()

    one_step_cfg = TrainingConfig(
        t_end_h=cfg.t_end_h,
        dt_h=cfg.dt_h,
        bounds=cfg.bounds,
        seed=cfg.seed,
        smooth_window=cfg.smooth_window,
        smooth_polyorder=cfg.smooth_polyorder,
    )

    correction = train_correction(batches, p, one_step_cfg, estimator)

    for _ in range(cfg.n_rollout_steps):
        hybrid = HybridModel(
            params=p,
            feed=FeedProfile(),
            correction=correction,
            t_end_h=cfg.t_end_h,
            dt_h=cfg.dt_h,
        )

        X_obs: list[Array] = []
        y_obs: list[Array] = []
        X_roll: list[Array] = []
        y_roll: list[Array] = []

        for batch in batches:
            Xb, yb = _build_targets(batch.t, batch.Y, p, cfg, batch.feed)
            X_obs.append(Xb)
            y_obs.append(yb)

            Xr, yr = _rollout_once(hybrid, batch, cfg)
            if len(Xr) > 0:
                X_roll.append(Xr)
                y_roll.append(yr)

        X_parts = []
        y_parts = []
        if X_obs:
            X_parts.append(np.vstack(X_obs))
            y_parts.append(np.concatenate(y_obs))
        if X_roll and cfg.rollout_weight > 0:
            X_parts.append(np.vstack(X_roll))
            y_parts.append(np.concatenate(y_roll))
            weights = [1.0 - cfg.rollout_weight, cfg.rollout_weight]
        else:
            weights = [1.0]

        if not X_parts:
            raise ValueError("no valid training rows were produced during rollout")

        # Mix by duplication rather than sample_weight, because sklearn
        # Pipeline.fit rejects a top-level sample_weight parameter. The
        # mixing is still correct: each source appears proportional to its
        # configured weight.
        duplicated_parts = []
        duplicated_y = []
        for X_part, y_part, weight in zip(X_parts, y_parts, weights, strict=True):
            n_copies = max(1, int(round(weight * 10)))  # weight 0.5 -> 5 copies
            duplicated_parts.extend([X_part] * n_copies)
            duplicated_y.extend([y_part] * n_copies)

        X_mix = np.vstack(duplicated_parts)
        y_mix = np.concatenate(duplicated_y)

        model = SklearnCorrection(estimator=estimator, bounds=cfg.bounds)
        correction = model.fit(X_mix, y_mix)

    return correction


def train_and_evaluate_rollout(
    train_batches: list[Batch],
    test_batches: list[Batch],
    p: KineticParameters | None = None,
    cfg: RolloutConfig | None = None,
    estimator: Any | None = None,
) -> tuple[HybridModel, EvaluationReport, EvaluationReport]:
    """Rollout train, then evaluate against the mechanistic baseline."""
    p = p or KineticParameters()
    cfg = cfg or RolloutConfig()

    correction = train_correction_rollout(train_batches, p, cfg, estimator)
    hybrid = HybridModel(
        params=p, feed=FeedProfile(), correction=correction, t_end_h=cfg.t_end_h, dt_h=cfg.dt_h
    )
    baseline = HybridModel.mechanistic_only(p)

    return hybrid, evaluate(hybrid, test_batches), evaluate(baseline, test_batches)
