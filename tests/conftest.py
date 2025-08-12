"""Shared fixtures.

Every fixture here is deterministic. A flaky scientific test suite is worse
than no test suite at all, because people learn to re-run it until it passes,
and at that point it has stopped being a test and become a ritual.
"""

from __future__ import annotations

import pytest

from hybridbio import (
    KineticParameters,
    TrainingConfig,
    generate_dataset,
    train_test_split_batches,
)


@pytest.fixture(scope="session")
def params() -> KineticParameters:
    return KineticParameters()


@pytest.fixture(scope="session")
def config() -> TrainingConfig:
    return TrainingConfig()


@pytest.fixture(scope="session")
def dataset():
    return generate_dataset(n_batches=20, seed=7)


@pytest.fixture(scope="session")
def split(dataset):
    return train_test_split_batches(dataset, n_test=5)
