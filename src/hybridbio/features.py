"""Feature transformations for the correction model.

Kept in its own module, as pure functions of state, for one reason: features
are the thing the Process Model Scientist will want to change most often and
argue about most fiercely. They should be editable without opening the model,
the trainer, or the workflow.

The feature contract is versioned. If you change what `build_features`
produces, you bump FEATURE_VERSION, and the regression tests will fail loudly
until someone consciously blesses the new golden values. Silent feature drift
between training and inference is one of the great unforced errors in applied
ML, and it is entirely preventable with a string constant and some discipline.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from .mechanistic import KineticParameters, specific_growth_rate

Array = NDArray[np.float64]

FEATURE_VERSION: Final[str] = "v2"

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "S",  # substrate
    "L",  # lactate
    "Xv",  # viable cell density
    "S_over_Ks",  # dimensionless substrate saturation
    "L_over_Ki",  # dimensionless inhibition load
    "mu_mech",  # what the mechanistic model currently believes
    "t_norm",  # normalised culture age -- proxies for unmodelled ageing
)
N_FEATURES: Final[int] = len(FEATURE_NAMES)


def build_features(
    t: Array,
    Y: Array,
    p: KineticParameters,
    t_end_h: float = 336.0,
) -> Array:
    """Map a trajectory to the correction model's feature matrix.

    Returns an array of shape (len(t), N_FEATURES).

    Note the inclusion of `mu_mech`: we explicitly hand the data-driven layer
    the mechanistic model's own opinion. It is not competing with the
    mechanism, it is *correcting* it, and it cannot correct what it cannot see.
    """
    if Y.ndim != 2 or Y.shape[1] != 5:
        raise ValueError(f"expected trajectory of shape (n, 5), got {Y.shape}")
    if len(t) != len(Y):
        raise ValueError(f"t/Y length mismatch: {len(t)} vs {len(Y)}")

    Xv, S, L = Y[:, 0], Y[:, 1], Y[:, 2]
    mu_mech = np.array(
        [specific_growth_rate(s, lac, p) for s, lac in zip(S, L, strict=True)], dtype=np.float64
    )

    feats = np.column_stack(
        [
            S,
            L,
            Xv,
            S / p.Ks,
            L / p.Ki_lac,
            mu_mech,
            t / t_end_h,
        ]
    ).astype(np.float64)

    if feats.shape[1] != N_FEATURES:
        raise AssertionError(
            f"feature matrix has {feats.shape[1]} columns but FEATURE_NAMES declares "
            f"{N_FEATURES}. These must not drift apart."
        )
    return feats


def features_at_point(t: float, y: Array, p: KineticParameters, t_end_h: float = 336.0) -> Array:
    """Single-row feature vector, for use inside the ODE right-hand side.

    Shares the definition above rather than reimplementing it -- because a
    train/inference feature skew introduced by two near-identical functions is
    a bug that takes a fortnight to find and ten seconds to cause.
    """
    return build_features(np.array([t]), y.reshape(1, -1), p, t_end_h)
