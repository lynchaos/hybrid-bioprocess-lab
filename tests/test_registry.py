"""Tests for MLflow registry integration.

These tests use a local temporary MLflow tracking URI so they do not need a
running server.
"""

from __future__ import annotations

import json

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


def test_register_refuses_rejected_manifest(tmp_path, mlflow_tmp_uri) -> None:
    model_dir, candidate, baseline = _trained_and_evaluated_model_dir(tmp_path)
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "created_at_utc": "2026-01-01T00:00:00Z",
                "data_source": "synthetic-fed-batch",
                "dataset_id": "test",
                "train_batch_ids": ["B000"],
                "test_batch_ids": ["B001"],
                "feature_version": "v1",
                "kinetic_parameters": {},
                "training_config": {},
                "candidate_metrics": candidate.metrics,
                "baseline_metrics": baseline.metrics,
                "candidate_constraints_ok": False,
                "candidate_violations": 1,
                "promotion_decision": "rejected",
                "promotion_reason": "scientific constraints failed",
                "git_sha": "test",
                "python_version": "3.11",
                "package_version": "test",
            }
        )
    )

    with pytest.raises(RegistryError, match="manifest rejects promotion"):
        log_and_register(model_dir, candidate_report=candidate, baseline_report=baseline)
