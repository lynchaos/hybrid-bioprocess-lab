"""Versioned provenance records for every train/evaluate/promote decision."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .data import Batch
from .evaluation import EvaluationReport
from .features import FEATURE_VERSION
from .mechanistic import KineticParameters

MANIFEST_SCHEMA_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """The reproducibility record attached to a saved model artifact."""

    schema_version: str
    created_at_utc: str
    data_source: str
    dataset_id: str
    train_batch_ids: tuple[str, ...]
    test_batch_ids: tuple[str, ...]
    feature_version: str
    kinetic_parameters: dict[str, float]
    training_config: dict[str, Any]
    candidate_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    candidate_constraints_ok: bool
    candidate_violations: int
    promotion_decision: str
    promotion_reason: str
    git_sha: str
    python_version: str
    package_version: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation with stable key names."""
        return asdict(self)


def build_manifest(
    *,
    data_source: str,
    dataset_id: str,
    train_batches: list[Batch],
    test_batches: list[Batch],
    params: KineticParameters,
    candidate_report: EvaluationReport,
    baseline_report: EvaluationReport,
    training_config: dict[str, Any],
    git_sha: str | None = None,
    created_at_utc: str | None = None,
) -> ExperimentManifest:
    """Capture the evidence required to reproduce a model-selection decision."""
    if not train_batches or not test_batches:
        raise ValueError("manifest requires non-empty train and test batch partitions")

    candidate_nrmse = candidate_report.metrics.get("nrmse_mean", float("inf"))
    baseline_nrmse = baseline_report.metrics.get("nrmse_mean", float("inf"))
    if not candidate_report.constraints_ok:
        decision = "rejected"
        reason = f"scientific constraints failed ({candidate_report.n_violations()} violation(s))"
    elif candidate_nrmse >= baseline_nrmse:
        decision = "rejected"
        reason = "candidate did not improve the mechanistic baseline"
    else:
        decision = "eligible_for_registration"
        reason = "scientific constraints passed and candidate improved the baseline"

    return ExperimentManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        created_at_utc=created_at_utc or _utc_now(),
        data_source=data_source,
        dataset_id=dataset_id,
        train_batch_ids=tuple(batch.batch_id for batch in train_batches),
        test_batch_ids=tuple(batch.batch_id for batch in test_batches),
        feature_version=FEATURE_VERSION,
        kinetic_parameters=params.as_dict(),
        training_config=training_config,
        candidate_metrics=dict(candidate_report.metrics),
        baseline_metrics=dict(baseline_report.metrics),
        candidate_constraints_ok=candidate_report.constraints_ok,
        candidate_violations=candidate_report.n_violations(),
        promotion_decision=decision,
        promotion_reason=reason,
        git_sha=git_sha or _git_sha(),
        python_version=platform.python_version(),
        package_version=_package_version(),
    )


def write_manifest(path: str | Path, manifest: ExperimentManifest) -> Path:
    """Persist a model-selection record alongside the model artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def load_manifest(path: str | Path) -> ExperimentManifest:
    """Load and validate the stable manifest schema."""
    raw = json.loads(Path(path).read_text())
    if raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema: {raw.get('schema_version')!r}")
    raw["train_batch_ids"] = tuple(raw["train_batch_ids"])
    raw["test_batch_ids"] = tuple(raw["test_batch_ids"])
    return ExperimentManifest(**raw)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_sha() -> str:
    configured = os.environ.get("GITHUB_SHA")
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _package_version() -> str:
    try:
        return version("hybridbio")
    except PackageNotFoundError:
        return "unknown"
