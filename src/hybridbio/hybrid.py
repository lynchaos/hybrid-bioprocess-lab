"""The hybrid model: mechanistic core + data-driven correction.

Composition, not inheritance. The mechanistic model knows nothing about ML,
the correction model knows nothing about biology, and this module is the only
place that knows about both. Each can be tested in isolation, which is the
entire reason to draw the boundary here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .constraints import ConstraintReport, check_trajectory
from .corrections import CorrectionModel, NullCorrection
from .corrections import save as save_correction
from .features import FEATURE_VERSION, features_at_point
from .mechanistic import FeedProfile, KineticParameters, default_initial_state, simulate

Array = NDArray[np.float64]


@dataclass(slots=True)
class HybridModel:
    """A mechanistic ODE whose growth rate is corrected by a learned model."""

    params: KineticParameters
    feed: FeedProfile
    correction: CorrectionModel
    t_end_h: float = 336.0
    dt_h: float = 6.0
    feature_version: str = FEATURE_VERSION

    @classmethod
    def mechanistic_only(
        cls,
        params: KineticParameters | None = None,
        feed: FeedProfile | None = None,
    ) -> HybridModel:
        """The honest baseline. Every hybrid model must beat this one."""
        return cls(
            params=params or KineticParameters(),
            feed=feed or FeedProfile(),
            correction=NullCorrection(),
        )

    def _mu_correction(self, t: float, y: Array) -> float:
        x = features_at_point(t, y, self.params, self.t_end_h)
        return float(self.correction.predict(x)[0])

    def simulate(self, y0: Array | None = None) -> tuple[Array, Array]:
        """Run the hybrid model forward. Returns (t, Y)."""
        if not self.correction.is_fitted:
            raise RuntimeError(
                "HybridModel.simulate called with an unfitted correction model. "
                "Fit it, or use HybridModel.mechanistic_only() if that is what you meant."
            )
        y0 = default_initial_state() if y0 is None else y0
        return simulate(
            p=self.params,
            feed=self.feed,
            y0=y0,
            t_end_h=self.t_end_h,
            dt_h=self.dt_h,
            mu_correction=self._mu_correction,
        )

    def simulate_checked(self, y0: Array | None = None) -> tuple[Array, Array, ConstraintReport]:
        """Simulate, then check the result against scientific constraints.

        This is the method you should reach for by default. `simulate` gives
        you a trajectory; `simulate_checked` gives you a trajectory *and tells
        you whether to believe it*.
        """
        t, Y = self.simulate(y0)
        report = check_trajectory(t, Y, self.params)
        return t, Y, report

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_correction(self.correction, directory / "correction.joblib")
        meta = {
            "feature_version": self.feature_version,
            "t_end_h": self.t_end_h,
            "dt_h": self.dt_h,
            **{f"param.{k}": v for k, v in self.params.as_dict().items()},
        }
        (directory / "metadata.json").write_text(_to_json(meta))
        return directory


def _to_json(d: dict[str, object]) -> str:
    import json

    return json.dumps(d, indent=2, sort_keys=True)


def growth_multiplier_curve(model: HybridModel, t: Array, Y: Array) -> Array:
    """The correction the model actually applied, over a trajectory.

    Purely diagnostic, and worth its weight in gold in a review meeting: a
    scientist can look at this curve and say "no, it should not be doing that
    at day 9", which is feedback no validation metric will ever give you.
    """
    from .features import build_features

    X = build_features(t, Y, model.params, model.t_end_h)
    return np.asarray(model.correction.predict(X), dtype=np.float64)
