"""Tests for the raw-data boundary before records reach model code."""

from __future__ import annotations

import csv

import pytest

from hybridbio.ingestion import REQUIRED_COLUMNS, BatchDataError, load_batches_csv


def _write_rows(tmp_path, rows: list[dict[str, str]]) -> str:
    path = tmp_path / "batches.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def _row(time_h: str, **overrides: str) -> dict[str, str]:
    row = {
        "batch_id": "B001",
        "time_h": time_h,
        "Xv_1e6_cells_per_mL": "0.3",
        "S_mM": "20.0",
        "L_mM": "1.0",
        "P_mg_per_L": "0.0",
        "V_L": "2.0",
        "feed_rate_L_per_h": "0.004",
        "feed_start_h": "72",
        "feed_S_mM": "320",
    }
    row.update(overrides)
    return row


def test_load_batches_csv_builds_batch_contract(tmp_path) -> None:
    batches = load_batches_csv(_write_rows(tmp_path, [_row("0"), _row("6", S_mM="18.0")]))
    assert len(batches) == 1
    assert batches[0].batch_id == "B001"
    assert batches[0].Y.shape == (2, 5)
    assert batches[0].feed.rate == pytest.approx(0.004)


def test_load_batches_csv_rejects_duplicate_time(tmp_path) -> None:
    path = _write_rows(tmp_path, [_row("0"), _row("0")])
    with pytest.raises(BatchDataError, match="strictly increasing"):
        load_batches_csv(path)


def test_load_batches_csv_rejects_inconsistent_feed(tmp_path) -> None:
    path = _write_rows(tmp_path, [_row("0"), _row("6", feed_rate_L_per_h="0.006")])
    with pytest.raises(BatchDataError, match="feed settings must be constant"):
        load_batches_csv(path)