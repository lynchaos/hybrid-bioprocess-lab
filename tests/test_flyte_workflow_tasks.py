"""Tests that exercise the Flyte workflow tasks themselves, not just the gate.

`workflows/flyte_training.py` is deliberately written to remain importable
(and testable) without flytekit installed -- see the fallback `task`/`workflow`
decorators at the top of that module. That comment is only worth anything if a
test actually calls the plain functions those decorators produce, including the
failure path a real Flyte run would report as a failed DAG. This file is that
test.

`workflows/` is not an installable package (there is no `__init__.py`, on
purpose -- these are example scripts, not library code), so the module is
loaded by file path rather than by `import workflows...`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

WORKFLOW_PATH = Path(__file__).resolve().parent.parent / "workflows" / "flyte_training.py"


def _load_workflow_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("flyte_training_under_test", WORKFLOW_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def flyte_wf() -> ModuleType:
    return _load_workflow_module()


def _train_and_score(flyte_wf: ModuleType, tmp_path: Path):
    return flyte_wf.train_and_score(seed=7, n_batches=8, n_test=2, model_output_dir=str(tmp_path))


def test_train_and_score_writes_model_and_manifest(tmp_path, flyte_wf) -> None:
    """The task Flyte would run first must leave a saved model + lineage manifest."""
    result = _train_and_score(flyte_wf, tmp_path)

    assert result.model_dir == str(tmp_path / "seed-7")
    assert (Path(result.model_dir) / "correction.joblib").exists()
    manifest_path = Path(result.model_dir) / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["feature_version"]
    assert result.n_violations == 0
    assert result.passed


def test_validation_gate_passes_an_admissible_result(tmp_path, flyte_wf) -> None:
    result = _train_and_score(flyte_wf, tmp_path)
    gated = flyte_wf.validation_gate(result=result)
    assert gated is result


def test_validation_gate_fails_the_dag_on_scientific_violation(tmp_path, flyte_wf) -> None:
    """A broken candidate must fail at the gate task, before registration is attempted.

    This is the scenario the job description calls "a failed Flyte run related
    to model code": the gate raises, the DAG fails, and the failure message
    names the violated constraint rather than a stack trace three layers down
    in the registry client.
    """
    result = _train_and_score(flyte_wf, tmp_path)
    broken = flyte_wf.TrainingResult(
        nrmse_mean=result.nrmse_mean,
        final_titre_rel_err=result.final_titre_rel_err,
        baseline_nrmse_mean=result.baseline_nrmse_mean,
        n_violations=1,
        passed=False,
        model_dir=result.model_dir,
        candidate_metrics=result.candidate_metrics,
        baseline_metrics=result.baseline_metrics,
    )
    with pytest.raises(ValueError, match="SCIENTIFIC VALIDATION FAILED"):
        flyte_wf.validation_gate(result=broken)


def test_validation_gate_fails_the_dag_on_regression(tmp_path, flyte_wf) -> None:
    """A candidate that regresses against the mechanistic baseline must also be vetoed."""
    result = _train_and_score(flyte_wf, tmp_path)
    non_improving = flyte_wf.TrainingResult(
        nrmse_mean=result.baseline_nrmse_mean,
        final_titre_rel_err=result.final_titre_rel_err,
        baseline_nrmse_mean=result.baseline_nrmse_mean,
        n_violations=0,
        passed=True,
        model_dir=result.model_dir,
        candidate_metrics=result.candidate_metrics,
        baseline_metrics=result.baseline_metrics,
    )
    with pytest.raises(ValueError, match="REGRESSION"):
        flyte_wf.validation_gate(result=non_improving)


def test_register_model_promotes_a_gated_result(tmp_path, flyte_wf, monkeypatch) -> None:
    """End to end: train, gate, and register against a local MLflow backend."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    result = _train_and_score(flyte_wf, tmp_path)
    gated = flyte_wf.validation_gate(result=result)
    version = flyte_wf.register_model(result=gated)
    assert version is not None and str(version) != ""
