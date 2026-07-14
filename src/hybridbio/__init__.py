"""hybridbio -- a small lab for practising production-grade hybrid bioprocess ML.

Public surface, deliberately narrow.
"""

from .audit import CorrectionAudit, audit_correction
from .constraints import ConstraintReport, ScientificConstraintError, check_trajectory
from .corrections import (
    CorrectionModel,
    NullCorrection,
    SklearnCorrection,
    mlp_estimator,
    tree_estimator,
)
from .data import Batch, generate_dataset, train_test_split_batches
from .evaluation import EvaluationReport, compare, evaluate
from .features import FEATURE_NAMES, FEATURE_VERSION, build_features
from .hybrid import HybridModel
from .inference import HybridPredictor, InferenceError, TrajectoryPrediction
from .lineage import ExperimentManifest, build_manifest, load_manifest, write_manifest
from .mechanistic import FeedProfile, KineticParameters, simulate
from .promotion import validate_promotion
from .pure_ml import PureMLConfig, PureMLTrajectoryModel, train_pure_ml_trajectory
from .reporting import render_html, render_markdown, write_report
from .study import (
    ConfidenceInterval,
    StudyConfig,
    StudyResult,
    paired_bootstrap_ci,
    run_repeated_study,
)
from .training import TrainingConfig, track_run, train_and_evaluate, train_correction
from .uncertainty import (
    EnsembleConfig,
    HybridEnsemble,
    TrajectoryInterval,
    train_bootstrap_ensemble,
)

try:
    from .torch_correction import TorchCorrection, torch_estimator
except ImportError:  # pragma: no cover - torch is optional
    TorchCorrection = None  # type: ignore
    torch_estimator = None  # type: ignore

__version__ = "0.4.0"

__all__ = [
    "Batch",
    "CorrectionAudit",
    "ConstraintReport",
    "ConfidenceInterval",
    "CorrectionModel",
    "EvaluationReport",
    "EnsembleConfig",
    "ExperimentManifest",
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "FeedProfile",
    "HybridModel",
    "HybridEnsemble",
    "HybridPredictor",
    "InferenceError",
    "KineticParameters",
    "NullCorrection",
    "PureMLConfig",
    "PureMLTrajectoryModel",
    "ScientificConstraintError",
    "SklearnCorrection",
    "StudyConfig",
    "StudyResult",
    "TorchCorrection",
    "TrajectoryPrediction",
    "TrajectoryInterval",
    "TrainingConfig",
    "audit_correction",
    "build_manifest",
    "build_features",
    "check_trajectory",
    "compare",
    "evaluate",
    "generate_dataset",
    "mlp_estimator",
    "load_manifest",
    "paired_bootstrap_ci",
    "render_html",
    "render_markdown",
    "run_repeated_study",
    "simulate",
    "torch_estimator",
    "track_run",
    "train_pure_ml_trajectory",
    "train_and_evaluate",
    "train_bootstrap_ensemble",
    "train_correction",
    "train_test_split_batches",
    "tree_estimator",
    "validate_promotion",
    "write_manifest",
    "write_report",
]
