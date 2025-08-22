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
from .mechanistic import FeedProfile, KineticParameters, simulate
from .training import TrainingConfig, track_run, train_and_evaluate, train_correction

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
    "KineticParameters",
    "NullCorrection",
    "ScientificConstraintError",
    "SklearnCorrection",
    "TrainingConfig",
    "build_features",
    "check_trajectory",
    "compare",
    "evaluate",
    "generate_dataset",
    "mlp_estimator",
    "tree_estimator",
    "simulate",
    "track_run",
    "train_and_evaluate",
    "train_correction",
    "train_test_split_batches",
]
