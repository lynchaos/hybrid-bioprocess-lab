"""Tests for the pure data-driven trajectory comparator."""

from __future__ import annotations

import numpy as np

from hybridbio import evaluate, train_pure_ml_trajectory
from hybridbio.pure_ml import build_trajectory_matrix


def test_pure_ml_uses_complete_training_batches_without_leaking_test_states(split, params) -> None:
    train_batches, test_batches = split
    features, targets = build_trajectory_matrix(train_batches, t_end_h=336.0)
    model = train_pure_ml_trajectory(train_batches, params=params)
    report = evaluate(model, test_batches)

    assert features.shape[0] == sum(len(batch) for batch in train_batches)
    assert targets.shape == (features.shape[0], 5)
    assert model.simulate_batch(test_batches[0])[1].shape == test_batches[0].Y.shape
    assert np.isfinite(report.metrics["nrmse_mean"])
