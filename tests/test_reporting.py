"""Tests for the reporting utility."""

from __future__ import annotations

from hybridbio import HybridModel, KineticParameters, evaluate, generate_dataset
from hybridbio.lineage import ExperimentManifest
from hybridbio.reporting import render_html, render_markdown, write_report


def test_render_markdown_contains_metrics() -> None:
    batches = generate_dataset(n_batches=4, seed=7)
    model = HybridModel.mechanistic_only(KineticParameters())
    report = evaluate(model, batches)
    md = render_markdown(hybrid=model, candidate_report=report)
    assert "nrmse_mean" in md
    assert "violations" in md
    assert "Hybrid Bioprocess Model Report" in md


def test_write_report_creates_file(tmp_path) -> None:
    batches = generate_dataset(n_batches=4, seed=7)
    model = HybridModel.mechanistic_only(KineticParameters())
    report = evaluate(model, batches)
    path = tmp_path / "report.md"
    written = write_report(path, hybrid=model, candidate_report=report)
    assert written.exists()
    assert "nrmse_mean" in written.read_text()


def test_render_html_contains_report_title() -> None:
    batches = generate_dataset(n_batches=4, seed=7)
    model = HybridModel.mechanistic_only(KineticParameters())
    report = evaluate(model, batches)
    html = render_html(hybrid=model, candidate_report=report)
    assert "<title>Hybrid Bioprocess Model Report</title>" in html
    assert "<h1>Hybrid Bioprocess Model Report</h1>" in html
    assert "<table>" in html
    assert "# Hybrid Bioprocess Model Report" not in html
    assert "nrmse_mean" in html


def test_write_html_includes_experiment_lineage(tmp_path) -> None:
    model = HybridModel.mechanistic_only(KineticParameters())
    manifest = ExperimentManifest(
        schema_version="v1",
        created_at_utc="2025-01-01T00:00:00Z",
        dataset_id="dataset-123",
        data_source="synthetic",
        train_batch_ids=("B001",),
        test_batch_ids=("B002",),
        feature_version=model.feature_version,
        kinetic_parameters=model.params.as_dict(),
        training_config={},
        candidate_metrics={},
        baseline_metrics={},
        candidate_constraints_ok=True,
        candidate_violations=0,
        promotion_decision="eligible_for_registration",
        promotion_reason="candidate passed",
        git_sha="abc123",
        python_version="3.11.0",
        package_version="0.1.0",
    )

    path = write_report(
        tmp_path / "report.html",
        hybrid=model,
        manifest=manifest,
        format="html",
    )
    html = path.read_text()

    assert "Experiment lineage" in html
    assert "dataset-123" in html
    assert "abc123" in html
