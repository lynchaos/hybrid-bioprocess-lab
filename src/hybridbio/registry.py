"""MLflow model registry integration.

This is the promotion stage: a trained and validated model is logged as an
MLflow model and, if it passes the scientific validation gate, registered under
a named model. The validation gate is the promotion criterion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import PythonModel  # type: ignore[attr-defined]
from mlflow.tracking import MlflowClient

from .evaluation import EvaluationReport
from .inference import HybridPredictor


class RegistryError(Exception):
    """Raised when registry interaction fails."""


def log_and_register(
    model_dir: str | Path,
    candidate_report: EvaluationReport,
    baseline_report: EvaluationReport,
    experiment: str = "hybrid-bioprocess",
    registered_model_name: str = "hybrid-bioprocess-correction",
    run_name: str = "train-and-register",
) -> str:
    """Log a saved model to MLflow and promote it to the model registry.

    Parameters
    ----------
    model_dir
        Directory containing the saved HybridModel artifacts.
    candidate_report
        Evaluation report for the trained hybrid model.
    baseline_report
        Evaluation report for the mechanistic baseline.
    experiment
        MLflow experiment name.
    registered_model_name
        Name under which to register the model.
    run_name
        Name for the MLflow run.

    Returns
    -------
    str
        The version string of the registered model.

    Raises
    ------
    RegistryError
        If the candidate fails the validation gate or MLflow interaction fails.
    """
    model_dir = Path(model_dir)
    if not candidate_report.passed:
        raise RegistryError(
            f"refusing to register: candidate did not pass validation. "
            f"violations={candidate_report.n_violations()}, "
            f"nrmse_mean={candidate_report.metrics.get('nrmse_mean')}"
        )

    baseline_nrmse = baseline_report.metrics.get("nrmse_mean")
    candidate_nrmse = candidate_report.metrics.get("nrmse_mean")
    if (
        baseline_nrmse is not None
        and candidate_nrmse is not None
        and candidate_nrmse >= baseline_nrmse
    ):
        raise RegistryError(
            f"refusing to register: regression against baseline "
            f"({candidate_nrmse:.4f} >= {baseline_nrmse:.4f})"
        )

    try:
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({f"param.{k}": v for k, v in _load_params(model_dir).items()})
            mlflow.log_metrics(candidate_report.metrics)
            mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_report.metrics.items()})
            mlflow.set_tag("feature_version", _load_feature_version(model_dir))
            mlflow.set_tag("constraints_ok", str(candidate_report.constraints_ok))

            predictor = HybridPredictor.load(model_dir)
            example = predictor.predict(check=False)
            signature = infer_signature(
                example.Y[:-1],  # rough input shape
                example.Y[1:],  # rough output shape
            )

            mlflow.log_artifacts(str(model_dir), artifact_path="model")
            model_info = mlflow.pyfunc.log_model(
                artifact_path="hybrid_model",
                python_model=_HybridPredictorPyfunc(),
                artifacts={"model_dir": str(model_dir)},
                signature=signature,
            )
            registered = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=registered_model_name,
            )
            return registered.version
    except Exception as e:
        raise RegistryError(f"MLflow registration failed: {e}") from e


def stage_exists(
    registered_model_name: str,
    stage: str = "Production",
) -> bool:
    """Check whether a model version is currently in the given stage."""
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(registered_model_name, stages=[stage])
        return len(versions) > 0
    except Exception:
        return False


def _load_params(model_dir: Path) -> dict[str, Any]:
    import json

    metadata = json.loads((model_dir / "metadata.json").read_text())
    return {k[len("param.") :]: v for k, v in metadata.items() if k.startswith("param.")}


def _load_feature_version(model_dir: Path) -> str:
    import json

    metadata = json.loads((model_dir / "metadata.json").read_text())
    return str(metadata.get("feature_version", "unknown"))


class _HybridPredictorPyfunc(PythonModel):
    """Thin MLflow pyfunc wrapper around HybridPredictor."""

    def load_context(self, context: Any) -> None:  # noqa: ANN401
        """MLflow calls this once when the model is loaded."""
        model_dir = context.artifacts.get("model_dir", "")
        self._predictor = HybridPredictor.load(model_dir)

    def predict(
        self,
        context: Any,  # noqa: ARG002
        model_input: Any,
        params: Any | None = None,  # noqa: ARG002
    ) -> Any:  # noqa: ANN401
        """MLflow pyfunc predict entry point."""
        import numpy as np

        if isinstance(model_input, dict):
            y0 = np.array(model_input.get("y0", []), dtype=float)
        else:
            y0 = np.asarray(model_input, dtype=float)

        if y0.ndim == 1:
            y0 = y0.reshape(1, -1)
        prediction = self._predictor.predict(y0=y0[0])
        return {"t": prediction.t.tolist(), "Y": prediction.Y.tolist()}
