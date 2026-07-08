"""Gas-assisted BBH hardening and its wrapper around the shared orbit loop."""

import numpy as np

from bsi_trap.physics.binary.gw import ode_gw_only
from bsi_trap.simulation.state.models import (
    BBHEvolverParams,
    BBH_properties,
    DiskProfile,
    AGN_properties,
)
from bsi_trap.physics.binary.numeric import (
    safe_sma,
    safe_eccentricity,
    is_physical_state,
)

import bsi_trap.constant.constant as const
from bsi_trap.physics.binary.evolution import (
    BinaryEvolutionState,
    evolve_binary_orbit,
)
from bsi_trap.physics.binary import quantities as pq


def ode_with_gas(
    Evolparams: BBHEvolverParams,
    m1_g: float,
    m2_g: float,
    a_cm: float,
    e: float,
    agn: AGN_properties,
    disk: DiskProfile,
) -> tuple[float, float]:
    a_cm = safe_sma(Evolparams, a_cm)
    e = safe_eccentricity(Evolparams, e)
    if not is_physical_state(Evolparams, a_cm, e):
        return np.nan, np.nan

    # Reduced mass and orbital frequency
    mu = m1_g * m2_g / (m1_g + m2_g)
    Omega = np.sqrt(const.G * (m1_g + m2_g) / a_cm**3)

    # Effective eccentricity for de/dt to avoid singularity at e=1
    e_eff = max(e, Evolparams.ecc_floor)

    alpha = agn.alpha
    r_trap_cm = disk.r_trap_cm
    cs_cgs = disk.get_cs(r_trap_cm)
    sig_trap_cgs = disk.get_sigma(r_trap_cm)
    # Gas hardening terms based on Ishibashi & Grobner 2020
    da_dt_gas = (
        -24
        * np.pi
        * alpha
        * cs_cgs**2
        * sig_trap_cgs
        * (1 + e) ** 2
        * a_cm
        / (mu * Omega)
    )

    de_dt_gas = (
        12
        * np.pi
        * alpha
        * cs_cgs**2
        * sig_trap_cgs
        * np.sqrt(1 - e**2)
        * (1 + e) ** 2
        * (1 - np.sqrt(1 - e**2))
        / (e_eff * mu * Omega)
    )

    # GW terms
    da_dt_gw, de_dt_gw = ode_gw_only(Evolparams, m1_g, m2_g, a_cm, e)

    # Return combined rates
    return da_dt_gas + da_dt_gw, de_dt_gas + de_dt_gw


def _gas_hardening_time_estimate_s(
    disk: DiskProfile,
    agn: AGN_properties,
    m1_g: float,
    m2_g: float,
    a_cm: float,
    e: float,
    merger_radius_cm: float,
) -> float:
    """Estimate gas-only time to the merger boundary.

    This guard uses the local gas hardening form in ``ode_with_gas`` with disk
    and eccentricity held fixed: da/dt = -K a^(5/2).  It is only used to avoid
    spending CPU on binaries that cannot reach the merger boundary before the
    AGN lifetime; the resolved trajectory still comes from the RK integrator.
    """
    if not (
        np.isfinite(a_cm)
        and np.isfinite(e)
        and np.isfinite(merger_radius_cm)
        and a_cm > merger_radius_cm > 0.0
        and 0.0 <= e < 1.0
    ):
        return np.inf

    mu = m1_g * m2_g / (m1_g + m2_g)
    coeff = (
        24.0
        * np.pi
        * agn.alpha
        * disk.get_cs(disk.r_trap_cm) ** 2
        * disk.get_sigma(disk.r_trap_cm)
        * (1.0 + e) ** 2
        / (mu * np.sqrt(const.G * (m1_g + m2_g)))
    )
    if not np.isfinite(coeff) or coeff <= 0.0:
        return np.inf

    return float(2.0 / (3.0 * coeff) * (merger_radius_cm**-1.5 - a_cm**-1.5))


def _combined_merger_time_estimate_s(
    disk: DiskProfile,
    agn: AGN_properties,
    m1_g: float,
    m2_g: float,
    a_cm: float,
    e: float,
    merger_radius_cm: float,
) -> float:
    """Return the fastest local gas/GW merger-time estimate."""
    gas_time_s = _gas_hardening_time_estimate_s(
        disk,
        agn,
        m1_g,
        m2_g,
        a_cm,
        e,
        merger_radius_cm,
    )
    gw_time_s = pq.compute_merger_time_estimate(m1_g, m2_g, a_cm, e)
    estimates_s = np.array([gas_time_s, gw_time_s], dtype=float)
    finite_estimates_s = estimates_s[np.isfinite(estimates_s)]
    if finite_estimates_s.size == 0:
        return np.inf
    return float(np.min(finite_estimates_s))


def _can_reach_merger_before_time_limit(
    disk: DiskProfile,
    agn: AGN_properties,
    m1_g: float,
    m2_g: float,
    a_cm: float,
    e: float,
    merger_radius_cm: float,
    remaining_time_s: float,
) -> bool:
    """Check whether local gas/GW estimates permit a merger before shutdown."""
    estimated_merger_s = _combined_merger_time_estimate_s(
        disk,
        agn,
        m1_g,
        m2_g,
        a_cm,
        e,
        merger_radius_cm,
    )
    return estimated_merger_s <= remaining_time_s


def evolve_gas(
    Evolparams: BBHEvolverParams,
    disk: DiskProfile,
    bbh: BBH_properties,
    agn: AGN_properties,
    i: int,
    *,
    bsi_enabled: bool = False,
    segment_stop_s: float | None = None,
) -> float:
    """Evolve one BBH with gas plus GW hardening and return elapsed time."""
    t_ini_s = float(bbh.t_ini_s[i])

    def can_merge_before_limit(
        state: BinaryEvolutionState,
        merger_radius_cm: float,
        remaining_time_s: float,
    ) -> bool:
        return _can_reach_merger_before_time_limit(
            disk,
            agn,
            state.m1_g,
            state.m2_g,
            state.sma_cm,
            state.ecc,
            merger_radius_cm,
            remaining_time_s,
        )

    return evolve_binary_orbit(
        Evolparams,
        bbh,
        agn,
        i,
        t_ini_s=t_ini_s,
        initial_step_fraction=Evolparams.gas_initial_step_fraction,
        ode_func=ode_with_gas,
        ode_args=(agn, disk),
        bsi_enabled=bsi_enabled,
        merger_reachability_guard=can_merge_before_limit,
        segment_stop_s=segment_stop_s,
    )
