"""Tests for reproducible experiment lineage artifacts."""

from __future__ import annotations

from hybridbio import HybridModel, KineticParameters, generate_dataset, train_correction
from hybridbio.evaluation import evaluate
from hybridbio.lineage import build_manifest, load_manifest, write_manifest


def test_manifest_records_split_reports_and_environment(tmp_path) -> None:
    batches = generate_dataset(n_batches=6, seed=7)
    train_batches, test_batches = batches[:4], batches[4:]
    params = KineticParameters()
    hybrid = HybridModel(
        params=params, feed=train_batches[0].feed, correction=train_correction(train_batches)
    )
    candidate = evaluate(hybrid, test_batches)
    baseline = evaluate(HybridModel.mechanistic_only(params), test_batches)

    manifest = build_manifest(
        data_source="synthetic-fed-batch",
        dataset_id="seed-7",
        train_batches=train_batches,
        test_batches=test_batches,
        params=params,
        candidate_report=candidate,
        baseline_report=baseline,
        training_config={"seed": 7, "backend": "sklearn"},
        git_sha="abc123",
        created_at_utc="2026-07-14T00:00:00Z",
    )
    path = write_manifest(tmp_path / "manifest.json", manifest)
    restored = load_manifest(path)

    assert restored.train_batch_ids == tuple(batch.batch_id for batch in train_batches)
    assert restored.test_batch_ids == tuple(batch.batch_id for batch in test_batches)
    assert restored.feature_version == hybrid.feature_version
    assert restored.candidate_constraints_ok == candidate.constraints_ok
    assert restored.git_sha == "abc123"
    assert restored.training_config["backend"] == "sklearn"
