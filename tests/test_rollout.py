"""Tests for rollout training."""

from __future__ import annotations

import numpy as np

from hybridbio import generate_dataset
from hybridbio import train_test_split_batches as split_batches
from hybridbio.mechanistic import FeedProfile, KineticParameters
from hybridbio.rollout import (
    RolloutConfig,
    _build_targets,
    _rollout_once,
    train_and_evaluate_rollout,
    train_correction_rollout,
)


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


def test_rollout_targets_use_batch_feed() -> None:
    batch = generate_dataset(n_batches=1, seed=7)[0]
    params = KineticParameters()
    config = RolloutConfig(n_rollout_steps=1)
    zero_feed = FeedProfile(rate=0.0, start_h=batch.feed.start_h, S_feed=batch.feed.S_feed)

    _, targets_with_batch_feed = _build_targets(batch.t, batch.Y, params, config, batch.feed)
    _, targets_with_zero_feed = _build_targets(batch.t, batch.Y, params, config, zero_feed)

    assert not np.array_equal(targets_with_batch_feed, targets_with_zero_feed)


def test_rollout_simulation_uses_batch_feed(monkeypatch) -> None:
    batch = generate_dataset(n_batches=1, seed=7)[0]
    correction = train_correction_rollout([batch], cfg=RolloutConfig(n_rollout_steps=0))
    from hybridbio import HybridModel

    model = HybridModel(
        params=KineticParameters(),
        feed=FeedProfile(rate=0.0),
        correction=correction,
    )
    captured: dict[str, FeedProfile] = {}
    original = HybridModel.simulate_batch

    def capture_simulation(self, observed_batch):
        captured["feed"] = observed_batch.feed
        return original(self, observed_batch)

    monkeypatch.setattr(HybridModel, "simulate_batch", capture_simulation)
    _rollout_once(model, batch, RolloutConfig(n_rollout_steps=1))

    assert captured["feed"] == batch.feed
