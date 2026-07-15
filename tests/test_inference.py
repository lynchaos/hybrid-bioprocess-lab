"""Tests for the inference / production loading path."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hybridbio import (
    FeedProfile,
    HybridModel,
    KineticParameters,
    generate_dataset,
    train_correction,
)
from hybridbio.inference import HybridPredictor, InferenceError


def _trained_model_dir(tmp_path):
    from hybridbio import FeedProfile

    batches = generate_dataset(n_batches=4, seed=7)
    correction = train_correction(batches)
    model = HybridModel(params=KineticParameters(), feed=FeedProfile(), correction=correction)
    out = tmp_path / "model"
    model.save(out)
    return out


def test_predictor_loads_saved_model(tmp_path) -> None:
    model_dir = _trained_model_dir(tmp_path)
    predictor = HybridPredictor.load(model_dir)
    prediction = predictor.predict()
    assert prediction.Y.shape[1] == 5
    assert prediction.constraint_report.ok


def test_predictor_uses_custom_feed(tmp_path) -> None:
    model_dir = _trained_model_dir(tmp_path)
    predictor = HybridPredictor.load(model_dir)
    from hybridbio import FeedProfile

    prediction = predictor.predict(feed=FeedProfile(rate=0.0))
    # No feed means constant volume.
    assert np.allclose(prediction.Y[:, 4], prediction.Y[0, 4])


def test_saved_model_preserves_training_feed(tmp_path) -> None:
    batches = generate_dataset(n_batches=4, seed=7)
    correction = train_correction(batches)
    feed = FeedProfile(rate=0.006, start_h=54.0, S_feed=280.0)
    model = HybridModel(params=KineticParameters(), feed=feed, correction=correction)
    model_dir = tmp_path / "custom-feed-model"
    model.save(model_dir)

    predictor = HybridPredictor.load(model_dir)

    assert predictor.model.feed == feed


def test_predictor_rejects_incompatible_feature_contract(tmp_path) -> None:
    model_dir = _trained_model_dir(tmp_path)
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["feature_version"] = "v999"
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(InferenceError, match="feature version"):
        HybridPredictor.load(model_dir)


def test_predictor_rejects_missing_directory(tmp_path) -> None:
    with pytest.raises(InferenceError, match="not found"):
        HybridPredictor.load(tmp_path / "does-not-exist")


def test_predictor_rejects_corrupt_metadata(tmp_path) -> None:
    model_dir = _trained_model_dir(tmp_path)
    (model_dir / "metadata.json").write_text("not json")
    with pytest.raises(InferenceError, match="corrupt metadata"):
        HybridPredictor.load(model_dir)


def test_predictor_returns_trajectory_with_true_titre(tmp_path) -> None:
    model_dir = _trained_model_dir(tmp_path)
    predictor = HybridPredictor.load(model_dir)
    prediction = predictor.predict(true_final_titre=1000.0)
    assert prediction.final_titre_rel_err is not None
    assert prediction.final_titre_rel_err >= 0.0
