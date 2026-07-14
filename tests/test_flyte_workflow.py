"""Tests for Flyte promotion gates without requiring a remote Flyte cluster."""

from __future__ import annotations

import pytest

from hybridbio.promotion import validate_promotion


def test_validation_gate_accepts_admissible_improvement() -> None:
    assert validate_promotion(nrmse_mean=0.1, baseline_nrmse_mean=0.2, n_violations=0) is None


def test_validation_gate_rejects_scientific_violation() -> None:
    with pytest.raises(ValueError, match="SCIENTIFIC VALIDATION FAILED"):
        validate_promotion(nrmse_mean=0.1, baseline_nrmse_mean=0.2, n_violations=1)


def test_validation_gate_rejects_non_improving_candidate() -> None:
    with pytest.raises(ValueError, match="REGRESSION"):
        validate_promotion(nrmse_mean=0.2, baseline_nrmse_mean=0.2, n_violations=0)
