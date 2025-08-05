"""Mechanistic core: a Monod-type fed-batch mammalian cell culture model.

This is the *scientific ground truth* layer of the hybrid model. It is
deliberately simple, deliberately transparent, and deliberately conservative:
it encodes structure we are confident about (mass balances, Monod kinetics,
inhibition by lactate) and nothing we are not.

Everything the mechanistic layer cannot explain is left for the data-driven
correction layer to absorb -- see `hybridbio.hybrid`.

State vector, in order:
    0: Xv  -- viable cell density        [1e6 cells/mL]
    1: S   -- substrate (glucose)        [mM]
    2: L   -- lactate                    [mM]
    3: P   -- product (mAb titre)        [mg/L]
    4: V   -- culture volume             [L]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

Array = NDArray[np.float64]

STATE_NAMES: Final[tuple[str, ...]] = ("Xv", "S", "L", "P", "V")
N_STATES: Final[int] = len(STATE_NAMES)

#: Half-saturation for maintenance substrate uptake [mM]. Small: maintenance
#: runs at essentially full rate until glucose is nearly exhausted, then shuts
#: off smoothly rather than driving S negative. Smooth, because a hard `if S>0`
#: switch would introduce exactly the discontinuity that wrecks the integrator.
_MAINTENANCE_KM: Final[float] = 0.05


@dataclass(frozen=True, slots=True)
class KineticParameters:
    """Kinetic parameters of the mechanistic model.

    Frozen so a parameter set can be hashed, logged and compared across runs
    without any risk of a downstream caller mutating it in place.
    """

    mu_max: float = 0.045  # max specific growth rate            [1/h]
    Ks: float = 1.5  # Monod half-saturation constant      [mM]
    Ki_lac: float = 45.0  # lactate inhibition constant         [mM]
    kd: float = 0.003  # specific death rate                 [1/h]
    Y_XS: float = 1.2e8  # yield, cells per mmol substrate     [cells/mmol]
    Y_LS: float = 0.85  # lactate produced per substrate      [mol/mol]
    q_p: float = 25.0  # specific productivity        [pg/cell/day]
    m_S: float = 8.0e-11  # noqa: N815 -- literature symbol; see lint note in pyproject

    def as_dict(self) -> dict[str, float]:
        """Flat dict, ready for MLflow / Optuna / JSON."""
        return {
            "mu_max": self.mu_max,
            "Ks": self.Ks,
            "Ki_lac": self.Ki_lac,
            "kd": self.kd,
            "Y_XS": self.Y_XS,
            "Y_LS": self.Y_LS,
            "q_p": self.q_p,
            "m_S": self.m_S,
        }

    def replace(self, **kwargs: float) -> KineticParameters:
        """Return a copy with selected fields overridden."""
        return replace(self, **kwargs)


@dataclass(frozen=True, slots=True)
class FeedProfile:
    """Bolus-free continuous feed: constant rate after a start time."""

    rate: float = 0.004  # feed rate                 [L/h]
    start_h: float = 72.0  # feed start                [h]
    S_feed: float = 320.0  # substrate in feed medium  [mM]

    def flow(self, t: float) -> float:
        """Volumetric feed rate at time `t` [L/h]."""
        return self.rate if t >= self.start_h else 0.0


def specific_growth_rate(S: float, L: float, p: KineticParameters) -> float:
    """Monod growth with non-competitive lactate inhibition.

    Clipped at zero: substrate can go marginally negative under solver
    tolerance, and a negative growth rate here would be physically
    meaningless (death is handled separately by `kd`).
    """
    S_eff = max(S, 0.0)
    L_eff = max(L, 0.0)
    return p.mu_max * (S_eff / (p.Ks + S_eff)) * (p.Ki_lac / (p.Ki_lac + L_eff))


def rhs(
    t: float,
    y: Array,
    p: KineticParameters,
    feed: FeedProfile,
    mu_correction: Callable[[float, Array], float] | None = None,
) -> Array:
    """Right-hand side of the mechanistic ODE system.

    `mu_correction` is the seam where the data-driven layer plugs in. It
    returns a *multiplicative* correction on the specific growth rate. The
    mechanistic model alone is recovered exactly when it is None.

    Keeping the hybrid seam inside a single, named scalar (rather than
    letting a neural net write directly to dX/dt) is a deliberate design
    choice: it means the data-driven layer can never break a mass balance,
    only re-time the biology. Constraints are structural, not policed
    after the fact.
    """
    Xv, S, L, P, V = y

    mu = specific_growth_rate(S, L, p)
    if mu_correction is not None:
        mu *= mu_correction(t, y)
        mu = max(mu, 0.0)

    F = feed.flow(t)
    dilution = F / max(V, 1e-9)

    # cell density: growth - death - dilution   [1e6 cells/mL/h]
    dXv = (mu - p.kd) * Xv - dilution * Xv

    # substrate: consumption for growth + maintenance, plus feed, minus dilution
    #
    # The maintenance term is *gated* by substrate availability. This is not
    # cosmetic. Without the gate, cells continue paying their maintenance cost
    # out of a glucose pool that has already hit zero, and S marches merrily
    # into negative territory -- which the scientific constraint checker caught
    # on the very first run of the test suite, and which no accuracy metric
    # would ever have flagged. Uptake must cease when there is nothing to take.
    S_eff = max(S, 0.0)
    availability = S_eff / (_MAINTENANCE_KM + S_eff)
    cells_per_L = Xv * 1e6 * 1e3  # 1e6 cells/mL -> cells/L
    q_S = (mu / p.Y_XS) + p.m_S * availability  # mmol/cell/h
    dS = -q_S * cells_per_L + dilution * (feed.S_feed - S)

    # lactate: stoichiometric byproduct of substrate consumption
    dL = p.Y_LS * q_S * cells_per_L - dilution * L

    # product: growth-associated-free (qp constant), pg/cell/day -> mg/L/h
    dP = p.q_p * Xv * 1e6 * 1e-9 * (1.0 / 24.0) * 1e3 - dilution * P

    dV = F

    return np.array([dXv, dS, dL, dP, dV], dtype=np.float64)


def simulate(
    p: KineticParameters,
    feed: FeedProfile,
    y0: Array,
    t_end_h: float = 336.0,
    dt_h: float = 6.0,
    mu_correction: Callable[[float, Array], float] | None = None,
) -> tuple[Array, Array]:
    """Integrate the model. Returns `(t, Y)` with `Y` shaped (n_times, N_STATES).

    Raises
    ------
    RuntimeError
        If the integrator fails to converge -- we never silently return a
        partial trajectory, because a truncated batch that looks plausible is
        far more dangerous downstream than a loud failure here.
    """
    t_eval = np.arange(0.0, t_end_h + 1e-9, dt_h)
    sol = solve_ivp(
        rhs,
        (0.0, t_end_h),
        y0,
        t_eval=t_eval,
        args=(p, feed, mu_correction),
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    return sol.t, sol.y.T


def default_initial_state() -> Array:
    """A typical seed condition for a 2 L fed-batch."""
    return np.array([0.35, 28.0, 0.5, 0.0, 1.6], dtype=np.float64)
