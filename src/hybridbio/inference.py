"""Production inference path for a saved hybrid model.

Training produces a directory. This module loads that directory and gives a
single-method object that can predict a trajectory from a seed state and a feed
profile. The scientist or downstream service should not need to know how the
correction was implemented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray

from .constraints import ConstraintReport, check_trajectory
from .corrections import CorrectionModel
from .features import FEATURE_VERSION
from .hybrid import HybridModel
from .mechanistic import FeedProfile, KineticParameters

Array = NDArray[np.float64]


class InferenceError(Exception):
    """Raised when a model artifact cannot be loaded or used."""


@dataclass(frozen=True, slots=True)
class TrajectoryPrediction:
    """The result of a single prediction call."""

    t: Array
    Y: Array
    constraint_report: ConstraintReport
    final_titre: float
    final_titre_rel_err: float | None


class HybridPredictor:
    """Load a saved HybridModel and predict trajectories.

    The saved artifact layout is:
        model_dir/
          correction.joblib
          metadata.json
    """

    def __init__(self, model: HybridModel) -> None:
        self._model = model

    @classmethod
    def load(cls, directory: str | Path) -> HybridPredictor:
        """Load a previously saved model directory."""
        directory = Path(directory)
        if not directory.is_dir():
            raise InferenceError(f"model directory not found: {directory}")

        metadata_path = directory / "metadata.json"
        correction_path = directory / "correction.joblib"
        if not metadata_path.exists():
            raise InferenceError(f"missing metadata: {metadata_path}")
        if not correction_path.exists():
            raise InferenceError(f"missing correction artifact: {correction_path}")

        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as e:
            raise InferenceError(f"corrupt metadata: {e}") from e

        feature_version = str(metadata.get("feature_version", ""))
        if feature_version != FEATURE_VERSION:
            raise InferenceError(
                f"incompatible feature version: artifact={feature_version!r}, "
                f"runtime={FEATURE_VERSION!r}"
            )

        params = _params_from_metadata(metadata)
        feed = _feed_from_metadata(metadata)
        correction = _load_correction(correction_path)

        model = HybridModel(
            params=params,
            feed=feed,
            correction=correction,
            t_end_h=float(metadata.get("t_end_h", 336.0)),
            dt_h=float(metadata.get("dt_h", 6.0)),
            feature_version=feature_version,
        )
        return cls(model)

    def predict(
        self,
        y0: Array | None = None,
        feed: FeedProfile | None = None,
        t_end_h: float | None = None,
        check: bool = True,
        true_final_titre: float | None = None,
    ) -> TrajectoryPrediction:
        """Predict a trajectory and optionally validate it."""
        model = self._model
        if feed is not None or t_end_h is not None:
            model = HybridModel(
                params=model.params,
                feed=feed if feed is not None else model.feed,
                correction=model.correction,
                t_end_h=t_end_h if t_end_h is not None else model.t_end_h,
                dt_h=model.dt_h,
            )

        t, Y = model.simulate(y0)
        report = check_trajectory(t, Y, model.params) if check else ConstraintReport()
        final_titre = float(Y[-1, 3])
        rel_err = None
        if true_final_titre is not None and true_final_titre > 1e-12:
            rel_err = abs(final_titre - true_final_titre) / true_final_titre

        return TrajectoryPrediction(
            t=t, Y=Y, constraint_report=report, final_titre=final_titre, final_titre_rel_err=rel_err
        )

    @property
    def model(self) -> HybridModel:
        return self._model


def _params_from_metadata(metadata: dict[str, Any]) -> KineticParameters:
    """Reconstruct KineticParameters from flattened metadata keys."""
    prefix = "param."
    raw = {k[len(prefix) :]: v for k, v in metadata.items() if k.startswith(prefix)}
    try:
        return KineticParameters(**{k: float(v) for k, v in raw.items()})
    except TypeError as e:
        raise InferenceError(f"metadata does not contain valid kinetic parameters: {e}") from e


def _feed_from_metadata(metadata: dict[str, Any]) -> FeedProfile:
    """Reconstruct FeedProfile from metadata, with sensible defaults."""
    return FeedProfile(
        rate=float(metadata.get("feed.rate", FeedProfile().rate)),
        start_h=float(metadata.get("feed.start_h", FeedProfile().start_h)),
        S_feed=float(metadata.get("feed.S_feed", FeedProfile().S_feed)),
    )


def _load_correction(path: Path) -> CorrectionModel:
    """Load a correction artifact, trying Torch first if available."""
    try:
        from .torch_correction import TorchCorrection

        candidate = joblib.load(path)
        if isinstance(candidate, TorchCorrection):
            return candidate
        if isinstance(candidate, dict) and "state_dict" in candidate:
            return TorchCorrection.load(path)
    except ImportError:
        pass

    correction = joblib.load(path)
    if isinstance(correction, CorrectionModel):
        return correction
    raise InferenceError(f"{path} does not contain a recognised CorrectionModel")
