"""Tests for the reporting utility."""

from __future__ import annotations

from hybridbio import HybridModel, KineticParameters, evaluate, generate_dataset
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
    assert "nrmse_mean" in html
