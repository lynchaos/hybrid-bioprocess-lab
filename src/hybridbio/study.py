"""Reproducible evidence generation for synthetic hybrid-model studies.

The functions here make uncertainty visible around model comparisons. They are
for simulation studies only; they never imply that synthetic confidence
intervals transfer to an industrial process without external validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .data import generate_dataset, train_test_split_batches
from .evaluation import EvaluationReport, evaluate
from .hybrid import HybridModel
from .mechanistic import KineticParameters
from .training import TrainingConfig, train_and_evaluate

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Percentile bootstrap interval for a mean batch-level effect."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    n_observations: int


@dataclass(frozen=True, slots=True)
class StudyConfig:
    seeds: tuple[int, ...] = (7, 17, 29, 43, 59)
    n_batches: int = 24
    n_test: int = 6
    n_bootstrap: int = 2_000
    bootstrap_seed: int = 0


@dataclass(slots=True)
class StudyResult:
    """Held-out comparison from repeated synthetic batch splits."""

    config: StudyConfig
    baseline_reports: list[EvaluationReport]
    candidate_reports: list[EvaluationReport]
    nrmse_delta: ConfidenceInterval
    admissible_runs: int

    @property
    def n_runs(self) -> int:
        return len(self.candidate_reports)

    @property
    def candidate_improves(self) -> bool:
        """Require a strictly negative interval, not merely a favorable mean."""
        return self.admissible_runs == self.n_runs and self.nrmse_delta.upper < 0.0


def paired_bootstrap_ci(
    values: Array,
    n_bootstrap: int = 2_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Return a percentile CI for the mean of paired batch-level differences."""
    if len(values) == 0:
        raise ValueError("bootstrap requires at least one observation")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[sample_indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        estimate=float(values.mean()),
        lower=float(np.quantile(means, alpha)),
        upper=float(np.quantile(means, 1.0 - alpha)),
        confidence=confidence,
        n_observations=len(values),
    )


def run_repeated_study(
    config: StudyConfig | None = None,
    params: KineticParameters | None = None,
    training_config: TrainingConfig | None = None,
) -> StudyResult:
    """Compare hybrid and mechanistic models over predefined synthetic seeds.

    The comparison is paired within each held-out batch. This prevents a noisy
    batch split from being mistaken for a model improvement.
    """
    config = config or StudyConfig()
    if not config.seeds:
        raise ValueError("at least one study seed is required")
    params = params or KineticParameters()
    training_config = training_config or TrainingConfig()
    baseline_reports: list[EvaluationReport] = []
    candidate_reports: list[EvaluationReport] = []
    deltas: list[float] = []

    for seed in config.seeds:
        batches = generate_dataset(n_batches=config.n_batches, seed=seed)
        train_batches, test_batches = train_test_split_batches(batches, n_test=config.n_test)
        hybrid, candidate, baseline = train_and_evaluate(
            train_batches,
            test_batches,
            p=params,
            cfg=training_config,
        )
        baseline_reports.append(baseline)
        candidate_reports.append(candidate)
        baseline_model = HybridModel.mechanistic_only(params)
        for batch in test_batches:
            deltas.append(
                evaluate(hybrid, [batch]).metrics["nrmse_mean"]
                - evaluate(baseline_model, [batch]).metrics["nrmse_mean"]
            )

    ci = paired_bootstrap_ci(
        np.asarray(deltas, dtype=np.float64),
        n_bootstrap=config.n_bootstrap,
        seed=config.bootstrap_seed,
    )
    return StudyResult(
        config=config,
        baseline_reports=baseline_reports,
        candidate_reports=candidate_reports,
        nrmse_delta=ci,
        admissible_runs=sum(report.constraints_ok for report in candidate_reports),
    )
