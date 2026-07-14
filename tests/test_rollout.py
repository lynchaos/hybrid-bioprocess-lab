"""Tests for rollout training."""

from __future__ import annotations

from hybridbio import generate_dataset
from hybridbio import train_test_split_batches as split_batches
from hybridbio.rollout import RolloutConfig, train_and_evaluate_rollout, train_correction_rollout


def test_rollout_train_produces_fitted_correction() -> None:
    batches = generate_dataset(n_batches=6, seed=7)
    train, _ = split_batches(batches, n_test=2)
    correction = train_correction_rollout(
        train,
        cfg=RolloutConfig(n_rollout_steps=1, seed=0),
    )
    assert correction.is_fitted


def test_rollout_model_beats_baseline() -> None:
    batches = generate_dataset(n_batches=8, seed=7)
    train, test = split_batches(batches, n_test=2)
    hybrid, candidate, baseline = train_and_evaluate_rollout(
        train,
        test,
        cfg=RolloutConfig(n_rollout_steps=1, seed=0),
    )
    assert candidate.constraints_ok, candidate.summary()
    assert candidate.metrics["nrmse_mean"] < baseline.metrics["nrmse_mean"]
    # Rollout model must also be admissible on its own rollout.
    _, _, report = hybrid.simulate_checked()
    assert report.ok, report.summary()
