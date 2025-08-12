"""Unit tests for the mechanistic core -- the ordinary, unglamorous kind."""

from __future__ import annotations

import numpy as np
import pytest

from hybridbio import FeedProfile, KineticParameters, simulate
from hybridbio.mechanistic import default_initial_state, specific_growth_rate


def test_growth_rate_is_monotonic_in_substrate(params: KineticParameters) -> None:
    rates = [specific_growth_rate(S, 0.0, params) for S in (0.0, 1.0, 5.0, 20.0, 100.0)]
    assert rates == sorted(rates)


def test_growth_rate_saturates_at_mu_max(params: KineticParameters) -> None:
    assert specific_growth_rate(1e6, 0.0, params) == pytest.approx(params.mu_max, rel=1e-4)


def test_growth_rate_never_exceeds_mu_max(params: KineticParameters) -> None:
    for S in (0.0, 0.1, 10.0, 1e9):
        for L in (0.0, 10.0, 500.0):
            assert specific_growth_rate(S, L, params) <= params.mu_max + 1e-12


def test_lactate_inhibits_growth(params: KineticParameters) -> None:
    assert specific_growth_rate(20.0, 60.0, params) < specific_growth_rate(20.0, 0.0, params)


def test_negative_substrate_is_clipped_not_propagated(params: KineticParameters) -> None:
    """Solver undershoot must not produce a negative growth rate."""
    assert specific_growth_rate(-1e-9, 0.0, params) == 0.0


def test_zero_substrate_means_zero_growth(params: KineticParameters) -> None:
    assert specific_growth_rate(0.0, 0.0, params) == 0.0


def test_simulation_shape_and_finiteness(params: KineticParameters) -> None:
    t, Y = simulate(params, FeedProfile(), default_initial_state(), t_end_h=240.0, dt_h=6.0)
    assert Y.shape == (len(t), 5)
    assert np.all(np.isfinite(Y))


def test_no_feed_means_constant_volume(params: KineticParameters) -> None:
    feed = FeedProfile(rate=0.0)
    _, Y = simulate(params, feed, default_initial_state(), t_end_h=120.0)
    assert np.allclose(Y[:, 4], Y[0, 4])


def test_feed_increases_volume(params: KineticParameters) -> None:
    feed = FeedProfile(rate=0.005, start_h=24.0)
    _, Y = simulate(params, feed, default_initial_state(), t_end_h=240.0)
    assert Y[-1, 4] > Y[0, 4]


def test_parameters_are_immutable(params: KineticParameters) -> None:
    with pytest.raises((AttributeError, TypeError)):
        params.mu_max = 99.0  # type: ignore[misc]


def test_parameters_replace_returns_a_copy(params: KineticParameters) -> None:
    faster = params.replace(mu_max=0.09)
    assert faster.mu_max == pytest.approx(0.09)
    assert params.mu_max != pytest.approx(0.09)
