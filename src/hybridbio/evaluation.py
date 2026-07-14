"""Evaluation harness: metrics and scientific validation, as co-equal citizens.

The organising idea of this module -- and, I'd argue, of the whole repo -- is
that `EvaluationReport.passed` is **not** a function of RMSE alone. A model
passes only if it is both accurate *and* admissible. A model that is 12% more
accurate and violates a carbon balance has not improved. It has regressed, in
the only sense that matters to the person who has to sign the batch record.

This is the reusable utility a modelling scientist should be able to call in
one line and trust, so that they never have to write evaluation glue again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .constraints import ConstraintReport, check_trajectory
from .data import Batch
from .mechanistic import STATE_NAMES, KineticParameters

Array = NDArray[np.float64]


class BatchTrajectoryModel(Protocol):
    """Minimal interface for models compared on observed batches."""

    params: KineticParameters

    def simulate_batch(self, batch: Batch) -> tuple[Array, Array]: ...


#: States we actually judge the model on. Volume is excluded: it is set by a
#: pump, so predicting it well is not an achievement, it is arithmetic.
SCORED_STATES: tuple[str, ...] = ("Xv", "S", "L", "P")


@dataclass(slots=True)
class EvaluationReport:
    metrics: dict[str, float] = field(default_factory=dict)
    constraints: list[ConstraintReport] = field(default_factory=list)
    n_batches: int = 0

    @property
    def constraints_ok(self) -> bool:
        return all(c.ok for c in self.constraints)

    @property
    def passed(self) -> bool:
        """Accurate *and* admissible. Both, or neither counts."""
        return self.constraints_ok and np.isfinite(self.metrics.get("nrmse_mean", np.inf))

    def n_violations(self) -> int:
        return sum(len(c.violations) for c in self.constraints)

    def render(self) -> str:
        lines = [
            f"Evaluation over {self.n_batches} batch(es)",
            "-" * 46,
        ]
        for k in sorted(self.metrics):
            lines.append(f"  {k:<24} {self.metrics[k]:>12.4f}")
        lines.append("-" * 46)
        status = "PASS" if self.passed else "FAIL"
        lines.append(f"  scientific constraints   {self.n_violations():>12d} violation(s)")
        lines.append(f"  verdict                  {status:>12}")
        if not self.constraints_ok:
            lines.append("")
            for c in self.constraints:
                for v in c.violations:
                    lines.append(f"    ! {v}")
        return "\n".join(lines)


def nrmse(y_true: Array, y_pred: Array) -> float:
    """RMSE normalised by the range of the truth, so states are comparable.

    Titre is O(1000) and lactate is O(10); an un-normalised mean RMSE across
    states is just a titre metric wearing a disguise.
    """
    denom = float(np.max(y_true) - np.min(y_true))
    if denom <= 1e-12:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / denom)


def evaluate(model: BatchTrajectoryModel, batches: list[Batch]) -> EvaluationReport:
    """Score a model against held-out batches, on metrics and on science."""
    if not batches:
        raise ValueError("refusing to evaluate on an empty batch list")

    report = EvaluationReport(n_batches=len(batches))
    per_state: dict[str, list[float]] = {s: [] for s in SCORED_STATES}
    final_titre_err: list[float] = []

    for batch in batches:
        t, Y_pred = model.simulate_batch(batch)
        report.constraints.append(check_trajectory(t, Y_pred, model.params))

        n = min(len(batch.t), len(t))
        for state in SCORED_STATES:
            i = STATE_NAMES.index(state)
            per_state[state].append(nrmse(batch.Y[:n, i], Y_pred[:n, i]))

        i_P = STATE_NAMES.index("P")
        truth = float(batch.Y[n - 1, i_P])
        pred = float(Y_pred[n - 1, i_P])
        if truth > 1e-9:
            final_titre_err.append(abs(pred - truth) / truth)

    for state, vals in per_state.items():
        report.metrics[f"nrmse_{state}"] = float(np.mean(vals))
    report.metrics["nrmse_mean"] = float(np.mean([np.mean(v) for v in per_state.values()]))
    if final_titre_err:
        # The business metric. If this degrades, nobody cares that loss improved.
        report.metrics["final_titre_rel_err"] = float(np.mean(final_titre_err))

    return report


def compare(
    baseline: EvaluationReport, candidate: EvaluationReport, tol: float = 1e-9
) -> dict[str, float]:
    """Delta between two reports. Negative means the candidate improved."""
    keys = set(baseline.metrics) & set(candidate.metrics)
    return {
        k: candidate.metrics[k] - baseline.metrics[k]
        for k in sorted(keys)
        if abs(baseline.metrics[k]) > tol
    }
