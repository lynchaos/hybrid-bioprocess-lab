"""Bootstrap ensemble uncertainty for hybrid trajectory predictions.

Intervals quantify variation caused by resampling the available training
batches. They are not process guarantees and must be calibrated on external
held-out batches before operational use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .data import Batch
from .hybrid import HybridModel
from .mechanistic import FeedProfile, KineticParameters
from .training import TrainingConfig, train_correction

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    n_members: int = 8
    seed: int = 0
    lower_quantile: float = 0.05
    upper_quantile: float = 0.95


@dataclass(frozen=True, slots=True)
class TrajectoryInterval:
    t: Array
    median: Array
    lower: Array
    upper: Array

    def empirical_coverage(self, observed: Array) -> float:
        """Fraction of measured values lying inside the predicted interval."""
        if observed.shape != self.median.shape:
            raise ValueError("observed trajectory shape must match interval shape")
        return float(np.mean((observed >= self.lower) & (observed <= self.upper)))


class HybridEnsemble:
    """A fitted collection of admissible hybrid members."""

    def __init__(self, members: list[HybridModel], config: EnsembleConfig) -> None:
        if not members:
            raise ValueError("an ensemble requires at least one member")
        self.members = members
        self.config = config

    def predict(
        self,
        y0: Array,
        feed: FeedProfile | None = None,
    ) -> TrajectoryInterval:
        trajectories: list[Array] = []
        t_reference: Array | None = None
        for member in self.members:
            model = HybridModel(
                params=member.params,
                feed=feed or member.feed,
                correction=member.correction,
                t_end_h=member.t_end_h,
                dt_h=member.dt_h,
            )
            t, trajectory = model.simulate(y0)
            t_reference = t if t_reference is None else t_reference
            trajectories.append(trajectory)
        values = np.stack(trajectories)
        assert t_reference is not None
        return TrajectoryInterval(
            t=t_reference,
            median=np.median(values, axis=0),
            lower=np.quantile(values, self.config.lower_quantile, axis=0),
            upper=np.quantile(values, self.config.upper_quantile, axis=0),
        )


def train_bootstrap_ensemble(
    batches: list[Batch],
    params: KineticParameters | None = None,
    training_config: TrainingConfig | None = None,
    config: EnsembleConfig | None = None,
) -> HybridEnsemble:
    """Fit members on independently resampled *batches*, never timepoints."""
    if not batches:
        raise ValueError("cannot train an ensemble without batches")
    params = params or KineticParameters()
    training_config = training_config or TrainingConfig()
    config = config or EnsembleConfig()
    if config.n_members < 1:
        raise ValueError("n_members must be positive")
    if not 0.0 <= config.lower_quantile < config.upper_quantile <= 1.0:
        raise ValueError("ensemble quantiles must satisfy 0 <= lower < upper <= 1")
    rng = np.random.default_rng(config.seed)
    members: list[HybridModel] = []
    for _ in range(config.n_members):
        sample = [batches[index] for index in rng.integers(0, len(batches), size=len(batches))]
        correction = train_correction(sample, params, training_config)
        members.append(
            HybridModel(
                params=params,
                feed=FeedProfile(),
                correction=correction,
                t_end_h=training_config.t_end_h,
                dt_h=training_config.dt_h,
            )
        )
    return HybridEnsemble(members, config)
