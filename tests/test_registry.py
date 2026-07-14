"""Tests for MLflow registry integration.

These tests use a local temporary MLflow tracking URI so they do not need a
running server.
"""

from __future__ import annotations

import pytest

from hybridbio import (
    FeedProfile,
    HybridModel,
    KineticParameters,
    generate_dataset,
    train_correction,
)
from hybridbio.evaluation import evaluate
from hybridbio.registry import RegistryError, log_and_register


@pytest.fixture
def mlflow_tmp_uri(tmp_path, monkeypatch):
    """Point MLflow at a temporary SQLite backend for the duration of the test."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


def _trained_and_evaluated_model_dir(tmp_path):
    batches = generate_dataset(n_batches=6, seed=7)
    correction = train_correction(batches)
    hybrid = HybridModel(params=KineticParameters(), feed=FeedProfile(), correction=correction)
    out = tmp_path / "model"
    hybrid.save(out)

    # Use a held-out batch for evaluation.
    test_batches = generate_dataset(n_batches=2, seed=8)
    candidate_report = evaluate(hybrid, test_batches)
    baseline_report = evaluate(HybridModel.mechanistic_only(KineticParameters()), test_batches)
    return out, candidate_report, baseline_report


def test_register_passes_and_returns_version(tmp_path, mlflow_tmp_uri) -> None:
    model_dir, candidate, baseline = _trained_and_evaluated_model_dir(tmp_path)
    version = log_and_register(
        model_dir,
        candidate_report=candidate,
        baseline_report=baseline,
        registered_model_name="test-hybrid-model",
    )
    assert version is not None and str(version) != ""


def test_register_refuses_failed_candidate(tmp_path, mlflow_tmp_uri) -> None:
    model_dir, _, baseline = _trained_and_evaluated_model_dir(tmp_path)
    from hybridbio.evaluation import EvaluationReport

    failed = EvaluationReport()
    with pytest.raises(RegistryError, match="refusing to register"):
        log_and_register(model_dir, candidate_report=failed, baseline_report=baseline)
