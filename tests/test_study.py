"""Tests for reproducible, uncertainty-aware synthetic studies."""

from __future__ import annotations

import numpy as np
import pytest

from hybridbio.study import StudyConfig, paired_bootstrap_ci, run_repeated_study


def test_paired_bootstrap_ci_is_reproducible() -> None:
    values = np.array([-0.2, -0.1, 0.0, -0.3], dtype=np.float64)
    result = paired_bootstrap_ci(values, n_bootstrap=200, seed=7)
    assert result.estimate == pytest.approx(-0.15)
    assert result.lower <= result.estimate <= result.upper
    assert result.n_observations == 4
    assert result == paired_bootstrap_ci(values, n_bootstrap=200, seed=7)


def test_repeated_study_returns_paired_batch_evidence() -> None:
    result = run_repeated_study(
        StudyConfig(seeds=(7, 11), n_batches=8, n_test=2, n_bootstrap=100, bootstrap_seed=3)
    )
    assert result.n_runs == 2
    assert result.nrmse_delta.n_observations == 4
    assert result.hybrid_vs_pure_ml_delta.n_observations == 4
    assert result.admissible_runs == 2
    assert len(result.pure_ml_reports) == 2
