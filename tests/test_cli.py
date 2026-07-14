"""Tests for CLI artifacts that users rely on outside the notebook."""

from __future__ import annotations

from hybridbio.cli import main
from hybridbio.lineage import load_manifest


def test_train_writes_manifest_and_auditable_report(tmp_path) -> None:
    model_dir = tmp_path / "model"
    report_path = tmp_path / "report.md"

    exit_code = main(
        [
            "train",
            "--out-dir",
            str(model_dir),
            "--n-batches",
            "6",
            "--n-test",
            "2",
            "--report",
            str(report_path),
        ]
    )

    manifest = load_manifest(model_dir / "manifest.json")
    assert exit_code == 0
    assert manifest.train_batch_ids == ("B000", "B001", "B002", "B003")
    assert manifest.test_batch_ids == ("B004", "B005")
    assert "Experiment lineage" in report_path.read_text()
    assert "Correction audit" in report_path.read_text()
