"""Synthetic batch data, standing in for a real historical batch record set.

The "plant" here is deliberately *not* the mechanistic model. It contains an
effect the mechanistic model does not know about -- a progressive decline in
growth rate with culture age, of the sort you would get from an unmodelled
metabolic shift or ammonia accumulation.

This matters. If the plant and the model were the same equations, the
correction model would have nothing to learn and every result in this repo
would be a beautiful, meaningless tautology. The gap between mechanism and
reality *is* the object of study.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .mechanistic import (
    FeedProfile,
    KineticParameters,
    default_initial_state,
    simulate,
)

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Batch:
    """One batch record: times, states, and the conditions that produced it."""

    batch_id: str
    t: Array
    Y: Array
    feed: FeedProfile
    y0: Array

    def __len__(self) -> int:
        return len(self.t)


def _true_plant_correction(
    decay_onset_h: float, severity: float
) -> Callable[[float, Array], float]:
    """The 'reality' the mechanistic model is missing.

    A sigmoidal decline in growth capability with culture age. The mechanistic
    model has no term for this; the correction model must discover it.
    """

    def correction(t: float, y: Array) -> float:  # noqa: ARG001
        z = (t - decay_onset_h) / 24.0
        return float(1.0 - severity / (1.0 + np.exp(-z)))

    return correction


def generate_batch(
    batch_id: str,
    rng: np.random.Generator,
    p: KineticParameters | None = None,
    t_end_h: float = 336.0,
    dt_h: float = 6.0,
    noise_frac: float = 0.03,
) -> Batch:
    """Simulate one 'real' batch, with batch-to-batch variability and noise."""
    p = p or KineticParameters()

    feed = FeedProfile(
        rate=float(rng.normal(0.004, 0.0004)),
        start_h=float(rng.normal(72.0, 6.0)),
        S_feed=320.0,
    )
    y0 = default_initial_state().copy()
    y0[0] *= float(rng.normal(1.0, 0.08))  # seeding density varies
    y0[1] *= float(rng.normal(1.0, 0.05))  # initial glucose varies

    # The unmodelled effect must bite EARLY and HARD enough to matter.
    #
    # My first version decayed growth only after ~168 h, by which point the
    # culture was already substrate-limited and the mechanistic model's growth
    # term was small anyway. Result: a plant-model gap of 3.8% against 3% assay
    # noise -- a signal-to-noise ratio of roughly 1.3. No correction model on
    # earth could have learned anything from that, and for an afternoon I
    # blamed the model for failing to extract a signal I had not put there.
    #
    # Diagnosing the *dataset* before blaming the *model* is the whole lesson.
    plant = _true_plant_correction(
        decay_onset_h=float(rng.normal(108.0, 12.0)),
        severity=float(np.clip(rng.normal(0.55, 0.06), 0.35, 0.72)),
    )

    t, Y = simulate(p, feed, y0, t_end_h=t_end_h, dt_h=dt_h, mu_correction=plant)

    # Multiplicative measurement noise, floored at zero: assays are noisy, but
    # they never report negative cell density.
    Y_noisy = Y * rng.normal(1.0, noise_frac, size=Y.shape)
    Y_noisy = np.maximum(Y_noisy, 0.0)
    Y_noisy[:, 4] = Y[:, 4]  # volume is known exactly -- it is a pump, not an assay

    return Batch(batch_id=batch_id, t=t, Y=Y_noisy, feed=feed, y0=y0)


def generate_dataset(n_batches: int = 12, seed: int = 7) -> list[Batch]:
    rng = np.random.default_rng(seed)
    return [generate_batch(f"B{i:03d}", rng) for i in range(n_batches)]


def train_test_split_batches(
    batches: list[Batch], n_test: int = 4
) -> tuple[list[Batch], list[Batch]]:
    """Split *by batch*, never by timepoint.

    Splitting timepoints randomly would leak: a point at t=120h in the train
    set tells you almost everything about t=126h in the test set from the same
    batch. The resulting metrics would be spectacular and entirely fictional.
    Batch-level splitting is the only honest option, and it is the one thing
    most people get wrong first.
    """
    if n_test >= len(batches):
        raise ValueError(f"n_test={n_test} leaves no training batches")
    return batches[:-n_test], batches[-n_test:]
