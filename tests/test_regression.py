"""Regression tests: the model may improve, but it may not silently change.

Two distinct jobs here, often conflated:

  1. **Golden-value regression.** The mechanistic model is a deterministic
     function. If someone refactors it and the day-14 titre moves by 4%, that
     is either a bug or a decision -- and either way a human must look at it.
     The test failing is the *feature*, not the nuisance.

  2. **Contract regression.** Feature names, feature count, feature version,
     and the public API surface are contracts. Breaking them silently is how
     you get a model that trains on seven features and infers on six, scores
     beautifully in CI, and is wrong in production.
"""

from __future__ import annotations

import numpy as np
import pytest

from hybridbio import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    FeedProfile,
    HybridModel,
    KineticParameters,
    build_features,
    evaluate,
    simulate,
    train_correction,
)
from hybridbio.features import N_FEATURES
from hybridbio.mechanistic import STATE_NAMES, default_initial_state

# --------------------------------------------------------------------------
# 1. Golden values
# --------------------------------------------------------------------------
#
# Regenerate deliberately, never reflexively:
#     python -m scripts.refresh_golden --justify "why this changed"
#
GOLDEN_FINAL_STATE: dict[str, float] = {
    "Xv": 4.716463,
    "S": 0.585165,
    "L": 122.285742,
    "P": 1105.987914,
    "V": 2.656000,
}
GOLDEN_RTOL = 1e-4


def test_mechanistic_final_state_matches_golden(params: KineticParameters) -> None:
    """If this fails, the mechanistic model changed. Was that intentional?"""
    _, Y = simulate(params, FeedProfile(), default_initial_state(), t_end_h=336.0, dt_h=6.0)
    final = Y[-1]
    for name, expected in GOLDEN_FINAL_STATE.items():
        actual = float(final[STATE_NAMES.index(name)])
        assert actual == pytest.approx(expected, rel=GOLDEN_RTOL), (
            f"{name}: expected {expected:.6f}, got {actual:.6f}. "
            "The mechanistic model has changed behaviour. If this was deliberate, "
            "update the golden values and say why in the commit message."
        )


def test_simulation_is_deterministic(params: KineticParameters) -> None:
    a = simulate(params, FeedProfile(), default_initial_state(), t_end_h=200.0)[1]
    b = simulate(params, FeedProfile(), default_initial_state(), t_end_h=200.0)[1]
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# 2. Contracts
# --------------------------------------------------------------------------


def test_feature_contract_is_pinned() -> None:
    """Feature drift between training and inference is a silent killer.

    Changing features is allowed. Changing them *without noticing* is not.
    """
    assert FEATURE_VERSION == "v2"
    assert FEATURE_NAMES == (
        "S",
        "L",
        "Xv",
        "S_over_Ks",
        "L_over_Ki",
        "mu_mech",
        "t_norm",
    )
    assert len(FEATURE_NAMES) == N_FEATURES


def test_feature_matrix_shape_matches_contract(params: KineticParameters) -> None:
    t, Y = simulate(params, FeedProfile(), default_initial_state(), t_end_h=120.0)
    X = build_features(t, Y, params)
    assert X.shape == (len(t), N_FEATURES)
    assert np.all(np.isfinite(X))


def test_features_are_pure(params: KineticParameters) -> None:
    """build_features must not mutate its inputs. Ever."""
    t, Y = simulate(params, FeedProfile(), default_initial_state(), t_end_h=120.0)
    Y_before = Y.copy()
    build_features(t, Y, params)
    assert np.array_equal(Y, Y_before)


def test_point_features_agree_with_batch_features(params: KineticParameters) -> None:
    """The single-point path used inside the ODE must agree exactly with the
    vectorised path used in training. This is the train/serve skew test."""
    from hybridbio.features import features_at_point

    t, Y = simulate(params, FeedProfile(), default_initial_state(), t_end_h=120.0)
    X_batch = build_features(t, Y, params)
    for i in (0, 5, 10, len(t) - 1):
        X_point = features_at_point(float(t[i]), Y[i], params)
        assert np.allclose(X_point[0], X_batch[i], rtol=1e-12)


# --------------------------------------------------------------------------
# 3. The hybrid layer must earn its keep
# --------------------------------------------------------------------------


def test_hybrid_beats_mechanistic_baseline(split, params: KineticParameters) -> None:
    """The whole justification for the ML layer, expressed as an assertion.

    If a data-driven correction cannot beat the mechanistic model it corrects,
    it is not a hybrid model. It is a liability with a training loop.
    """
    train_batches, test_batches = split
    correction = train_correction(train_batches, params)

    hybrid = HybridModel(params=params, feed=FeedProfile(), correction=correction)
    baseline = HybridModel.mechanistic_only(params)

    hybrid_report = evaluate(hybrid, test_batches)
    baseline_report = evaluate(baseline, test_batches)

    assert hybrid_report.constraints_ok, hybrid_report.summary()
    assert hybrid_report.metrics["nrmse_mean"] < baseline_report.metrics["nrmse_mean"], (
        f"hybrid nrmse {hybrid_report.metrics['nrmse_mean']:.4f} did not improve on "
        f"mechanistic baseline {baseline_report.metrics['nrmse_mean']:.4f}"
    )


def test_trained_hybrid_stays_admissible(split, params: KineticParameters) -> None:
    """Accuracy is not a licence to violate biology."""
    train_batches, _ = split
    correction = train_correction(train_batches, params)
    model = HybridModel(params=params, feed=FeedProfile(), correction=correction)
    _, _, report = model.simulate_checked()
    assert report.ok, report.summary()


def test_unfitted_correction_refuses_to_simulate(params: KineticParameters) -> None:
    """No silent degradation to a mechanistic model wearing a hybrid label."""
    from hybridbio import SklearnCorrection

    model = HybridModel(params=params, feed=FeedProfile(), correction=SklearnCorrection())
    with pytest.raises(RuntimeError, match="unfitted"):
        model.simulate()
