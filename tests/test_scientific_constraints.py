"""Scientific constraint tests.

If you read one file in this repository, read this one.

These tests do not check that the model is *accurate*. They check that it is
*possible*. Every test below corresponds to a statement about biology or
thermodynamics that must hold regardless of what the loss function says, and
each one is a real failure mode I have either caused or watched someone cause:

  * a correction model that learned to grow cells faster than mu_max
  * a feature change that let glucose go negative and nobody noticed for a week
  * a "better" model whose lactate came from nowhere at all

An ML regression suite that only asserts `rmse < threshold` would have passed
all three.
"""

from __future__ import annotations

import numpy as np
import pytest

from hybridbio import (
    FeedProfile,
    HybridModel,
    KineticParameters,
    ScientificConstraintError,
    check_trajectory,
    simulate,
)
from hybridbio.constraints import ConstraintReport
from hybridbio.mechanistic import default_initial_state


class _RogueCorrection:
    """A correction model that has learned something biologically obscene.

    We inject this deliberately. A guardrail that has never been shown to catch
    anything is not a guardrail, it is a comment.
    """

    def __init__(self, multiplier: float) -> None:
        self.multiplier = multiplier

    def fit(self, X, y):  # noqa: ANN001, ARG002
        return self

    def predict(self, X):  # noqa: ANN001
        return np.full(len(X), self.multiplier, dtype=np.float64)

    @property
    def is_fitted(self) -> bool:
        return True


# --------------------------------------------------------------------------
# The mechanistic model must itself be admissible. If the floor is broken,
# nothing built on top of it can be trusted.
# --------------------------------------------------------------------------


def test_mechanistic_trajectory_is_scientifically_admissible(params: KineticParameters) -> None:
    t, Y = simulate(params, FeedProfile(), default_initial_state(), t_end_h=336.0)
    report = check_trajectory(t, Y, params)
    assert report.ok, report.summary()


def test_states_never_go_negative(params: KineticParameters) -> None:
    t, Y = simulate(params, FeedProfile(), default_initial_state(), t_end_h=336.0)
    assert np.all(Y >= -1e-6), "a state variable went negative: this is not a small numerical issue"


def test_product_mass_is_non_decreasing(params: KineticParameters) -> None:
    """Antibody does not spontaneously un-secrete itself."""
    t, Y = simulate(params, FeedProfile(), default_initial_state(), t_end_h=336.0)
    product_mass = Y[:, 3] * Y[:, 4]
    assert np.all(np.diff(product_mass) >= -1e-6)


def test_apparent_growth_never_exceeds_mu_max(params: KineticParameters) -> None:
    t, Y = simulate(params, FeedProfile(), default_initial_state(), t_end_h=336.0)
    mu_apparent = np.gradient(np.log(np.maximum(Y[:, 0], 1e-9)), t)
    assert float(np.max(mu_apparent)) <= params.mu_max * 1.25


# --------------------------------------------------------------------------
# The checker must actually catch things. Negative tests, which is where most
# validation frameworks quietly fall over.
# --------------------------------------------------------------------------


def test_checker_catches_negative_states(params: KineticParameters) -> None:
    t = np.linspace(0.0, 100.0, 20)
    Y = np.ones((20, 5))
    Y[:, 4] = 2.0
    Y[10, 1] = -5.0  # glucose goes impossibly negative
    report = check_trajectory(t, Y, params)
    assert not report.ok
    assert any(v.name == "negative_S" for v in report.violations)


def test_checker_catches_destroyed_product(params: KineticParameters) -> None:
    t = np.linspace(0.0, 100.0, 20)
    Y = np.ones((20, 5))
    Y[:, 4] = 2.0
    Y[:, 3] = np.linspace(100.0, 10.0, 20)  # titre falls: product vanished
    report = check_trajectory(t, Y, params)
    assert not report.ok
    assert any(v.name == "product_destroyed" for v in report.violations)


def test_checker_catches_shrinking_volume(params: KineticParameters) -> None:
    t = np.linspace(0.0, 100.0, 20)
    Y = np.ones((20, 5))
    Y[:, 4] = np.linspace(2.0, 1.0, 20)  # feed-only process, yet volume drops
    report = check_trajectory(t, Y, params)
    assert not report.ok
    assert any(v.name == "volume_decrease" for v in report.violations)


def test_rogue_correction_is_caught_not_rewarded(params: KineticParameters) -> None:
    """The central scenario this repo exists to defend against.

    A correction model that triples the growth rate will fit some training
    curves beautifully. It is also biologically impossible, and the pipeline
    must refuse it -- loudly, and before it reaches anyone's slide deck.
    """
    model = HybridModel(
        params=params,
        feed=FeedProfile(),
        correction=_RogueCorrection(multiplier=3.0),
    )
    _, _, report = model.simulate_checked()
    assert not report.ok, "a 3x growth-rate correction was accepted as admissible"
    assert any(v.name == "superluminal_growth" for v in report.violations)


def test_report_raises_on_violation() -> None:
    from hybridbio.constraints import Violation

    report = ConstraintReport(violations=[Violation("x", "broken", 1.0)])
    with pytest.raises(ScientificConstraintError, match="constraint violation"):
        report.raise_if_violated()


def test_clean_report_is_truthy_and_silent() -> None:
    report = ConstraintReport()
    assert report
    report.raise_if_violated()  # must not raise
