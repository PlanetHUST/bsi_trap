"""Peters gravitational-wave evolution and an explicitly toy remnant model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bsi_trap.units import C_CM_S, G_CGS, MSUN_G


@dataclass(frozen=True)
class PetersEvolutionResult:
    initial_sma_cm: float
    initial_eccentricity: float
    final_sma_cm: float
    final_eccentricity: float
    elapsed_time_s: float
    integration_stop_radius_cm: float
    reached_integration_stop: bool


@dataclass(frozen=True)
class MergerRemnant:
    final_mass_msun: float
    final_spin: float
    model_name: str


def _validate_binary(m1_msun: float, m2_msun: float, sma_cm: float, eccentricity: float) -> None:
    if not all(np.isfinite(value) for value in (m1_msun, m2_msun, sma_cm, eccentricity)):
        raise ValueError("binary inputs must be finite")
    if m1_msun <= 0.0 or m2_msun <= 0.0:
        raise ValueError("binary masses [Msun] must be positive")
    if sma_cm <= 0.0:
        raise ValueError("binary semi-major axis [cm] must be positive")
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("binary eccentricity must be in [0, 1)")


def peters_da_dt_cm_s(
    m1_msun: float,
    m2_msun: float,
    sma_cm: float,
    eccentricity: float,
) -> float:
    """Return Peters da/dt in CGS [cm/s]."""
    _validate_binary(m1_msun, m2_msun, sma_cm, eccentricity)
    m1_g = m1_msun * MSUN_G
    m2_g = m2_msun * MSUN_G
    eccentricity_factor = (
        1.0
        + (73.0 / 24.0) * eccentricity**2
        + (37.0 / 96.0) * eccentricity**4
    )
    return float(
        -(64.0 / 5.0)
        * G_CGS**3
        * m1_g
        * m2_g
        * (m1_g + m2_g)
        * eccentricity_factor
        / (C_CM_S**5 * sma_cm**3 * (1.0 - eccentricity**2) ** 3.5)
    )


def peters_de_dt_s_inv(
    m1_msun: float,
    m2_msun: float,
    sma_cm: float,
    eccentricity: float,
) -> float:
    """Return Peters de/dt [s^-1]."""
    _validate_binary(m1_msun, m2_msun, sma_cm, eccentricity)
    if eccentricity == 0.0:
        return 0.0
    m1_g = m1_msun * MSUN_G
    m2_g = m2_msun * MSUN_G
    return float(
        -(304.0 / 15.0)
        * G_CGS**3
        * m1_g
        * m2_g
        * (m1_g + m2_g)
        * eccentricity
        * (1.0 + (121.0 / 304.0) * eccentricity**2)
        / (C_CM_S**5 * sma_cm**4 * (1.0 - eccentricity**2) ** 2.5)
    )


def circular_merger_time_s(m1_msun: float, m2_msun: float, sma_cm: float) -> float:
    """Return the analytic circular time from ``sma_cm`` to zero separation."""
    _validate_binary(m1_msun, m2_msun, sma_cm, 0.0)
    m1_g = m1_msun * MSUN_G
    m2_g = m2_msun * MSUN_G
    return float(
        5.0
        * C_CM_S**5
        * sma_cm**4
        / (256.0 * G_CGS**3 * m1_g * m2_g * (m1_g + m2_g))
    )


def binary_integration_stop_radius_cm(m1_msun: float, m2_msun: float) -> float:
    """Return the chosen approximate integration stop, 6 G(m1+m2)/c^2 [cm].

    This is a numerical boundary for the Peters inspiral, not a claim that the
    remnant forms according to a resolved strong-field merger calculation.
    """
    if m1_msun <= 0.0 or m2_msun <= 0.0:
        raise ValueError("binary masses [Msun] must be positive")
    return float(6.0 * G_CGS * (m1_msun + m2_msun) * MSUN_G / C_CM_S**2)


def _circular_evolution(
    m1_msun: float,
    m2_msun: float,
    initial_sma_cm: float,
    stop_radius_cm: float,
    max_time_s: float,
) -> PetersEvolutionResult:
    full_time_s = circular_merger_time_s(m1_msun, m2_msun, initial_sma_cm)
    stop_time_s = full_time_s * (1.0 - (stop_radius_cm / initial_sma_cm) ** 4)
    elapsed_time_s = min(max_time_s, stop_time_s)
    final_sma_fourth_cm4 = initial_sma_cm**4 * (
        1.0 - elapsed_time_s / full_time_s
    )
    final_sma_cm = float(max(final_sma_fourth_cm4, 0.0) ** 0.25)
    reached = elapsed_time_s >= stop_time_s
    if reached:
        final_sma_cm = stop_radius_cm
    return PetersEvolutionResult(
        initial_sma_cm=initial_sma_cm,
        initial_eccentricity=0.0,
        final_sma_cm=final_sma_cm,
        final_eccentricity=0.0,
        elapsed_time_s=float(elapsed_time_s),
        integration_stop_radius_cm=stop_radius_cm,
        reached_integration_stop=reached,
    )


def evolve_binary_peters(
    m1_msun: float,
    m2_msun: float,
    initial_sma_cm: float,
    initial_eccentricity: float,
    max_time_s: float,
    *,
    relative_tolerance: float = 1.0e-8,
    max_step_fraction: float = 0.01,
) -> PetersEvolutionResult:
    """Integrate pure Peters GW evolution until the stop radius or time limit."""
    _validate_binary(m1_msun, m2_msun, initial_sma_cm, initial_eccentricity)
    if not np.isfinite(max_time_s) or max_time_s < 0.0:
        raise ValueError("GW evolution time budget [s] must be finite and nonnegative")
    if relative_tolerance <= 0.0 or max_step_fraction <= 0.0:
        raise ValueError("GW solver tolerances must be positive")
    stop_radius_cm = binary_integration_stop_radius_cm(m1_msun, m2_msun)
    if initial_sma_cm <= stop_radius_cm:
        return PetersEvolutionResult(
            initial_sma_cm=initial_sma_cm,
            initial_eccentricity=initial_eccentricity,
            final_sma_cm=initial_sma_cm,
            final_eccentricity=initial_eccentricity,
            elapsed_time_s=0.0,
            integration_stop_radius_cm=stop_radius_cm,
            reached_integration_stop=True,
        )
    if initial_eccentricity == 0.0:
        return _circular_evolution(
            m1_msun,
            m2_msun,
            initial_sma_cm,
            stop_radius_cm,
            max_time_s,
        )
    if max_time_s == 0.0:
        return PetersEvolutionResult(
            initial_sma_cm=initial_sma_cm,
            initial_eccentricity=initial_eccentricity,
            final_sma_cm=initial_sma_cm,
            final_eccentricity=initial_eccentricity,
            elapsed_time_s=0.0,
            integration_stop_radius_cm=stop_radius_cm,
            reached_integration_stop=False,
        )

    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:
        raise ImportError("scipy is required for eccentric Peters evolution") from exc

    log_separation_limit = float(np.log(initial_sma_cm / stop_radius_cm))

    def derivatives(log_shrinkage: float, state: np.ndarray) -> tuple[float, float]:
        eccentricity = max(float(state[0]), 0.0)
        sma_cm = initial_sma_cm * np.exp(-log_shrinkage)
        da_dt_cm_s = peters_da_dt_cm_s(
            m1_msun,
            m2_msun,
            sma_cm,
            eccentricity,
        )
        dt_dlog_shrinkage_s = -sma_cm / da_dt_cm_s
        de_dlog_shrinkage = (
            peters_de_dt_s_inv(
                m1_msun,
                m2_msun,
                sma_cm,
                eccentricity,
            )
            * dt_dlog_shrinkage_s
        )
        return de_dlog_shrinkage, dt_dlog_shrinkage_s

    def time_budget_event(_log_shrinkage: float, state: np.ndarray) -> float:
        return float(max_time_s - state[1])

    time_budget_event.terminal = True  # type: ignore[attr-defined]
    time_budget_event.direction = -1.0  # type: ignore[attr-defined]
    solution = solve_ivp(
        derivatives,
        (0.0, log_separation_limit),
        (initial_eccentricity, 0.0),
        events=time_budget_event,
        rtol=relative_tolerance,
        atol=(relative_tolerance, 1.0e-6),
        max_step=max(
            log_separation_limit * max_step_fraction,
            np.spacing(log_separation_limit),
        ),
    )
    if not solution.success:
        raise RuntimeError(f"Peters integration failed: {solution.message}")
    exhausted_time_budget = bool(solution.t_events[0].size)
    reached = not exhausted_time_budget
    elapsed_time_s = float(solution.y[1, -1])
    final_sma_cm = (
        stop_radius_cm
        if reached
        else float(initial_sma_cm * np.exp(-solution.t[-1]))
    )
    final_eccentricity = max(float(solution.y[0, -1]), 0.0)
    return PetersEvolutionResult(
        initial_sma_cm=initial_sma_cm,
        initial_eccentricity=initial_eccentricity,
        final_sma_cm=final_sma_cm,
        final_eccentricity=final_eccentricity,
        elapsed_time_s=elapsed_time_s,
        integration_stop_radius_cm=stop_radius_cm,
        reached_integration_stop=reached,
    )


def gas_da_dt_cm_s(*_args: object, **_kwargs: object) -> float:
    """Return zero until a validated gas-hardening prescription is selected."""
    return 0.0


def gas_de_dt_s_inv(*_args: object, **_kwargs: object) -> float:
    """Return zero until a validated gas-eccentricity prescription is selected."""
    return 0.0


def compute_merger_remnant(m1_msun: float, m2_msun: float) -> MergerRemnant:
    """Return the retained toy 5%-mass-loss and approximate spin prescription."""
    if m1_msun <= 0.0 or m2_msun <= 0.0:
        raise ValueError("merger component masses [Msun] must be positive")
    total_mass_msun = m1_msun + m2_msun
    symmetric_mass_ratio = m1_msun * m2_msun / total_mass_msun**2
    final_spin = np.clip(
        3.464 * symmetric_mass_ratio - 2.029 * symmetric_mass_ratio**2,
        0.0,
        0.99,
    )
    return MergerRemnant(
        final_mass_msun=0.95 * total_mass_msun,
        final_spin=float(final_spin),
        model_name="toy_5pct_mass_loss_spin_polynomial",
    )


__all__ = [
    "MergerRemnant",
    "PetersEvolutionResult",
    "binary_integration_stop_radius_cm",
    "circular_merger_time_s",
    "compute_merger_remnant",
    "evolve_binary_peters",
    "gas_da_dt_cm_s",
    "gas_de_dt_s_inv",
    "peters_da_dt_cm_s",
    "peters_de_dt_s_inv",
]
