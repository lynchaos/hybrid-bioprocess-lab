"""Tests for bootstrap trajectory intervals."""

from __future__ import annotations

import numpy as np
import pytest

from hybridbio import generate_dataset, train_test_split_batches
from hybridbio.uncertainty import EnsembleConfig, train_bootstrap_ensemble


def test_bootstrap_ensemble_returns_ordered_trajectory_interval() -> None:
    train, test = train_test_split_batches(generate_dataset(n_batches=8, seed=7), n_test=2)
    ensemble = train_bootstrap_ensemble(train, config=EnsembleConfig(n_members=3, seed=2))
    interval = ensemble.predict(test[0].y0, test[0].feed)
    assert interval.median.shape == test[0].Y.shape
    assert np.all(interval.lower <= interval.median)
    assert np.all(interval.median <= interval.upper)
    assert 0.0 <= interval.empirical_coverage(test[0].Y) <= 1.0


def test_interval_rejects_shape_mismatch() -> None:
    train, test = train_test_split_batches(generate_dataset(n_batches=6, seed=7), n_test=2)
    ensemble = train_bootstrap_ensemble(train, config=EnsembleConfig(n_members=2))
    interval = ensemble.predict(test[0].y0)
    with pytest.raises(ValueError, match="shape"):
        interval.empirical_coverage(np.ones((1, 1)))