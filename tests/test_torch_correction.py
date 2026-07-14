"""Tests for the Torch correction model."""

from __future__ import annotations

import numpy as np
import pytest

from hybridbio import FeedProfile, HybridModel, KineticParameters, build_features, generate_dataset
from hybridbio.torch_correction import TorchCorrection, torch_estimator


def test_torch_correction_fits_and_predicts() -> None:
    batches = generate_dataset(n_batches=4, seed=7)
    X = np.vstack([build_features(b.t, b.Y, KineticParameters()) for b in batches])
    y = np.ones(len(X))  # trivial target

    model = TorchCorrection(epochs=10, seed=0)
    fitted = model.fit(X, y)
    assert fitted.is_fitted
    preds = fitted.predict(X[:5])
    assert preds.shape == (5,)
    assert np.all((preds >= 0.5) & (preds <= 1.8))


def test_torch_correction_refuses_prediction_before_fit() -> None:
    model = TorchCorrection(epochs=10)
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(np.ones((3, 7)))


def test_torch_correction_saves_and_loads(tmp_path) -> None:
    batches = generate_dataset(n_batches=4, seed=7)
    X = np.vstack([build_features(b.t, b.Y, KineticParameters()) for b in batches])
    y = np.ones(len(X))

    model = TorchCorrection(epochs=10, seed=0).fit(X, y)
    path = tmp_path / "torch.joblib"
    model.save(path)

    loaded = TorchCorrection.load(path)
    assert loaded.is_fitted
    np.testing.assert_allclose(loaded.predict(X[:5]), model.predict(X[:5]), rtol=1e-5)


def test_torch_correction_replaces_sklearn_in_hybrid() -> None:
    from hybridbio import evaluate, train_correction, train_test_split_batches

    batches = generate_dataset(n_batches=8, seed=7)
    train, test = train_test_split_batches(batches, n_test=2)
    correction = train_correction(train, estimator=torch_estimator(epochs=200, seed=0))
    hybrid = HybridModel(params=KineticParameters(), feed=FeedProfile(), correction=correction)
    report = evaluate(hybrid, test)
    assert report.constraints_ok, report.summary()


def test_torch_factory_returns_fitted_model() -> None:
    batches = generate_dataset(n_batches=4, seed=7)
    X = np.vstack([build_features(b.t, b.Y, KineticParameters()) for b in batches])
    y = np.ones(len(X))
    model = torch_estimator(epochs=10, seed=0).fit(X, y)
    assert model.is_fitted
