"""Biological and thermodynamic constraints on a simulated trajectory.

This module is the reason the repo exists.

A hybrid model can improve RMSE on every state variable and still be broken,
because RMSE cannot see that the model has quietly started consuming negative
glucose, or growing cells in a vessel that ran dry, or violating the carbon
balance by 30%. Loss went down. The model is wrong. Nobody notices until a
process engineer looks at a plot three weeks later and says "that can't happen".

So constraints get their own first-class module, their own report object, and
their own regression tests -- exactly the same status as metrics, not a
footnote underneath them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .mechanistic import KineticParameters

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Violation:
    name: str
    detail: str
    worst_value: float

    def __str__(self) -> str:
        return f"[{self.name}] {self.detail} (worst={self.worst_value:.4g})"


@dataclass(slots=True)
class ConstraintReport:
    """Result of checking a trajectory. Truthy iff the trajectory is admissible."""

    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def __bool__(self) -> bool:
        return self.ok

    def raise_if_violated(self) -> None:
        if not self.ok:
            lines = "\n  ".join(str(v) for v in self.violations)
            raise ScientificConstraintError(
                f"{len(self.violations)} constraint violation(s):\n  {lines}"
            )

    def summary(self) -> str:
        if self.ok:
            return "all scientific constraints satisfied"
        return f"{len(self.violations)} violation(s): " + "; ".join(v.name for v in self.violations)


class ScientificConstraintError(AssertionError):
    """Raised when a trajectory is not physically or biologically admissible."""


def check_trajectory(
    t: Array,
    Y: Array,
    p: KineticParameters,
    *,
    tol: float = 1e-6,
    carbon_tol_frac: float = 0.05,
) -> ConstraintReport:
    """Check a simulated trajectory against non-negotiable scientific facts.

    Parameters
    ----------
    tol
        Absolute slack for non-negativity, to absorb ODE solver noise. A state
        at -1e-9 is a solver artefact; a state at -3.0 is a broken model.
    carbon_tol_frac
        Fractional slack on the lactate-from-substrate carbon balance.
    """
    report = ConstraintReport()
    Xv, S, L, P, V = (Y[:, i] for i in range(5))

    for name, series in (("Xv", Xv), ("S", S), ("L", L), ("P", P), ("V", V)):
        worst = float(np.min(series))
        if worst < -tol:
            report.violations.append(
                Violation(
                    name=f"negative_{name}",
                    detail=f"{name} went negative -- physically impossible",
                    worst_value=worst,
                )
            )

    if float(np.min(V)) <= 0.0:
        report.violations.append(
            Violation("empty_vessel", "culture volume reached zero or below", float(np.min(V)))
        )

    # Volume is monotonically non-decreasing: this model only feeds, never harvests.
    dV = np.diff(V)
    if dV.size and float(np.min(dV)) < -tol:
        report.violations.append(
            Violation(
                "volume_decrease", "volume decreased in a feed-only process", float(np.min(dV))
            )
        )

    # Product is cumulative in mass: titre * volume must never fall.
    product_mass = P * V
    dPm = np.diff(product_mass)
    if dPm.size and float(np.min(dPm)) < -tol:
        report.violations.append(
            Violation("product_destroyed", "cumulative product mass decreased", float(np.min(dPm)))
        )

    # Carbon balance: lactate produced must be consistent with substrate consumed,
    # within the stoichiometric yield. Lactate cannot be conjured from nowhere.
    lactate_mass = np.trapezoid(np.gradient(L * V, t).clip(min=0.0), t)
    substrate_consumed = _substrate_consumed(t, S, V)
    max_lactate = p.Y_LS * substrate_consumed * (1.0 + carbon_tol_frac)
    if lactate_mass > max_lactate + tol:
        report.violations.append(
            Violation(
                "carbon_balance",
                f"lactate produced ({lactate_mass:.1f} mmol) exceeds what the "
                f"substrate consumed can stoichiometrically support ({max_lactate:.1f} mmol)",
                float(lactate_mass - max_lactate),
            )
        )

    # Growth rate can never exceed mu_max: no correction factor is allowed to
    # make cells divide faster than biology permits.
    if Xv.size > 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            mu_apparent = np.gradient(np.log(np.maximum(Xv, 1e-9)), t)
        finite = mu_apparent[np.isfinite(mu_apparent)]
        if finite.size:
            worst_mu = float(np.max(finite))
            if worst_mu > p.mu_max * 1.25:
                report.violations.append(
                    Violation(
                        "superluminal_growth",
                        f"apparent growth rate {worst_mu:.4f}/h exceeds mu_max={p.mu_max:.4f}/h",
                        worst_mu,
                    )
                )

    return report


def _substrate_consumed(t: Array, S: Array, V: Array, S_feed: float = 320.0) -> float:
    """Total substrate consumed [mmol] = initial + fed - remaining.

    Inferring the fed mass from the volume trajectory rather than taking a
    FeedProfile keeps this checker usable on *measured* batch data, where you
    have a volume trace and no guarantee the recorded setpoint is what the pump
    actually did.
    """
    substrate_in_vessel = S * V
    fed = float(np.trapezoid(np.gradient(V, t).clip(min=0.0), t)) * S_feed
    return max(float(substrate_in_vessel[0] + fed - substrate_in_vessel[-1]), 0.0)
