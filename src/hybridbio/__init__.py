"""hybridbio -- a small lab for practising production-grade hybrid bioprocess ML.

Public surface, deliberately narrow.
"""

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
from .mechanistic import FeedProfile, KineticParameters, simulate
from .reporting import render_html, render_markdown, write_report
from .training import TrainingConfig, track_run, train_and_evaluate, train_correction

try:
    from .torch_correction import TorchCorrection, torch_estimator
except ImportError:  # pragma: no cover - torch is optional
    TorchCorrection = None  # type: ignore
    torch_estimator = None  # type: ignore

__version__ = "0.4.0"

__all__ = [
    "Batch",
    "ConstraintReport",
    "CorrectionModel",
    "EvaluationReport",
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "FeedProfile",
    "HybridModel",
    "HybridPredictor",
    "InferenceError",
    "KineticParameters",
    "NullCorrection",
    "ScientificConstraintError",
    "SklearnCorrection",
    "TorchCorrection",
    "TrajectoryPrediction",
    "TrainingConfig",
    "build_features",
    "check_trajectory",
    "compare",
    "evaluate",
    "generate_dataset",
    "mlp_estimator",
    "render_html",
    "render_markdown",
    "simulate",
    "torch_estimator",
    "track_run",
    "train_and_evaluate",
    "train_correction",
    "train_test_split_batches",
    "tree_estimator",
    "write_report",
]
