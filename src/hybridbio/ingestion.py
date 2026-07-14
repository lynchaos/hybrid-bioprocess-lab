"""Validated ingestion of scientist-provided batch records.

Model code operates on :class:`Batch`, never directly on CSV rows. This module
is the boundary that turns an exported historian file into that stable internal
contract and fails early when units, timestamps, or physical values are wrong.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import Batch
from .mechanistic import FeedProfile

REQUIRED_COLUMNS: tuple[str, ...] = (
    "batch_id",
    "time_h",
    "Xv_1e6_cells_per_mL",
    "S_mM",
    "L_mM",
    "P_mg_per_L",
    "V_L",
    "feed_rate_L_per_h",
    "feed_start_h",
    "feed_S_mM",
)


class BatchDataError(ValueError):
    """Raised when a batch record cannot safely enter the modeling workflow."""


def load_batches_csv(path: str | Path) -> list[Batch]:
    """Load a unit-explicit CSV export into validated batch trajectories.

    Each row is one measurement. Feed settings must remain constant within a
    batch because a :class:`FeedProfile` represents one planned feed regime.
    """
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise BatchDataError("CSV is missing a header row")
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames))
        if missing:
            raise BatchDataError(f"CSV is missing required columns: {', '.join(missing)}")
        rows_by_batch: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row_number, row in enumerate(reader, start=2):
            batch_id = (row.get("batch_id") or "").strip()
            if not batch_id:
                raise BatchDataError(f"row {row_number}: batch_id is required")
            rows_by_batch[batch_id].append(row)

    if not rows_by_batch:
        raise BatchDataError("CSV contains no measurement rows")
    return [_to_batch(batch_id, rows) for batch_id, rows in sorted(rows_by_batch.items())]


def _to_batch(batch_id: str, rows: list[dict[str, str]]) -> Batch:
    measurements = np.array(
        [
            _finite(row, column, batch_id)
            for row in rows
            for column in (
                "time_h",
                "Xv_1e6_cells_per_mL",
                "S_mM",
                "L_mM",
                "P_mg_per_L",
                "V_L",
            )
        ],
        dtype=np.float64,
    ).reshape(len(rows), 6)
    t = measurements[:, 0]
    Y = measurements[:, 1:]
    if len(t) < 2:
        raise BatchDataError(f"batch {batch_id}: at least two measurement rows are required")
    if np.any(np.diff(t) <= 0.0):
        raise BatchDataError(f"batch {batch_id}: time_h must be strictly increasing")
    if np.any(Y < 0.0):
        raise BatchDataError(f"batch {batch_id}: measured states must be non-negative")
    if np.any(Y[:, 4] <= 0.0):
        raise BatchDataError(f"batch {batch_id}: V_L must be strictly positive")

    feed_values = {
        column: {_finite(row, column, batch_id) for row in rows}
        for column in ("feed_rate_L_per_h", "feed_start_h", "feed_S_mM")
    }
    inconsistent = [column for column, values in feed_values.items() if len(values) != 1]
    if inconsistent:
        columns = ", ".join(inconsistent)
        raise BatchDataError(
            f"batch {batch_id}: feed settings must be constant; inconsistent {columns}"
        )
    feed = FeedProfile(
        rate=feed_values["feed_rate_L_per_h"].pop(),
        start_h=feed_values["feed_start_h"].pop(),
        S_feed=feed_values["feed_S_mM"].pop(),
    )
    return Batch(batch_id=batch_id, t=t, Y=Y, feed=feed, y0=Y[0].copy())


def _finite(row: dict[str, str], column: str, batch_id: str) -> float:
    raw = row.get(column)
    try:
        value = float(raw) if raw is not None else float("nan")
    except ValueError as error:
        raise BatchDataError(f"batch {batch_id}: {column} must be numeric") from error
    if not math.isfinite(value):
        raise BatchDataError(f"batch {batch_id}: {column} must be finite")
    return value
