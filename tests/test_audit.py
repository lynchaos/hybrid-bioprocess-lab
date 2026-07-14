"""Tests for correction-model audit summaries."""

from __future__ import annotations

import numpy as np

from hybridbio import (
    FeedProfile,
    HybridModel,
    KineticParameters,
    generate_dataset,
    train_correction,
)
from hybridbio.audit import audit_correction


def test_audit_reports_finite_feature_sensitivities() -> None:
    batches = generate_dataset(n_batches=6, seed=7)
    params = KineticParameters()
    model = HybridModel(
        params=params,
        feed=FeedProfile(),
        correction=train_correction(batches, params),
    )

    audit = audit_correction(model, batches)

    assert audit.n_observations == sum(len(batch) for batch in batches)
    assert audit.correction_min > 0.0
    assert audit.correction_max >= audit.correction_min
    assert set(audit.feature_effects) == set(audit.feature_names)
    assert np.all(np.isfinite(list(audit.feature_effects.values())))
