"""Shared model-promotion rules used by workflows and registry callers."""

from __future__ import annotations


def validate_promotion(
    *,
    nrmse_mean: float,
    baseline_nrmse_mean: float,
    n_violations: int,
) -> None:
    """Reject invalid or non-improving candidates before registration."""
    if n_violations > 0:
        raise ValueError(
            f"SCIENTIFIC VALIDATION FAILED: {n_violations} constraint "
            "violation(s). Refusing to register this model regardless of its metrics."
        )
    if nrmse_mean >= baseline_nrmse_mean:
        raise ValueError(
            f"REGRESSION: hybrid nrmse {nrmse_mean:.4f} did not improve on the "
            f"mechanistic baseline {baseline_nrmse_mean:.4f}. The correction "
            "model is not earning its keep."
        )
