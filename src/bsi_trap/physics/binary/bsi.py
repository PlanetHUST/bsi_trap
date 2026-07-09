"""Reduced binary-single interaction rates and weak perturbation updates.

The important model boundary is:

* runtime code samples when a BSI event occurs from the current live reservoir;
* ``apply_scheduled_bsi_step`` changes the mutable BBH state at that event time;
* ``bsi_trap.simulation.timeline.construct_bsi_timeline`` later packages
  recorded flags and delays into an event ledger.

This keeps the physical update out of ``BSI_Timeline``.  Timelines answer
"when did the event happen?", while this module's step function answers "what
did that event do to the binary?".
"""

from __future__ import annotations

import numpy as np

from bsi_trap.simulation.state.models import (
    AGN_properties,
    BBHEvolverParams,
    BBH_flags,
    BBH_properties,
    DiskProfile,
)
from bsi_trap.simulation.config.model_config import ModelConfig
from bsi_trap.constant.constant import G, msun
from bsi_trap.physics.trap.context import build_trap_context
from bsi_trap.simulation.timeline.models import FloatArray
from bsi_trap.physics.pairing import (
    bbh_cross_section,
    shear_relative_velocity,
)


def _event_crossed(delay_s: float, elapsed_s: float, step_s: float) -> bool:
    """Return True when an integration step crosses a primed BSI delay."""
    if not (
        np.isfinite(delay_s)
        and delay_s >= 0.0
        and np.isfinite(elapsed_s)
        and np.isfinite(step_s)
        and step_s >= 0.0
    ):
        return False
    return elapsed_s <= delay_s <= elapsed_s + step_s


def _bsi_impact_parameter_cm(
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    bbh: BBH_properties,
    binary_idx: int,
) -> float:
    if model_config.bsi_eta <= 0.0:
        return 0.0

    sma_cm = float(bbh.sma_evolve_cm[binary_idx])
    binary_mass_msun = float(bbh.m1_msun[binary_idx] + bbh.m2_msun[binary_idx])
    if (
        not np.isfinite(sma_cm)
        or not np.isfinite(binary_mass_msun)
        or sma_cm <= 0.0
        or binary_mass_msun <= 0.0
        or agn.M_SMBH_msun <= 0.0
    ):
        return 0.0

    trap_context = build_trap_context(disk, model_config)
    hill_radius_cm = disk.r_trap_cm * (binary_mass_msun / (3.0 * agn.M_SMBH_msun)) ** (
        1.0 / 3.0
    )
    return float(
        min(
            bbh_cross_section(model_config, sma_cm),
            hill_radius_cm,
            0.5 * trap_context.width_cm,
        )
    )


def binary_single_rate_s_inv(
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    bbh: BBH_properties,
    binary_idx: int,
    active_single_masses_msun: FloatArray,
) -> float:
    if active_single_masses_msun.size == 0:
        return 0.0

    trap_context = build_trap_context(disk, model_config)
    if trap_context.volume_cm3 <= 0.0:
        return 0.0

    p_max_cm = _bsi_impact_parameter_cm(model_config, agn, disk, bbh, binary_idx)
    if not np.isfinite(p_max_cm) or p_max_cm <= 0.0:
        return 0.0

    v_inf_cms = shear_relative_velocity(
        agn.M_SMBH_msun,
        disk.r_trap_cm,
        disk.r_trap_cm + p_max_cm,
    )
    if not np.isfinite(v_inf_cms) or v_inf_cms <= 0.0:
        return 0.0

    binary_mass_g = float(bbh.m1_msun[binary_idx] + bbh.m2_msun[binary_idx]) * msun
    single_mass_g = np.asarray(active_single_masses_msun, dtype=np.float64) * msun
    focusing = 1.0 + 2.0 * G * (binary_mass_g + single_mass_g) / (
        p_max_cm * v_inf_cms**2
    )
    cross_section_cm2 = np.pi * p_max_cm**2 * focusing
    rate_s_inv = float(np.sum(cross_section_cm2 * v_inf_cms) / trap_context.volume_cm3)
    if not np.isfinite(rate_s_inv) or rate_s_inv <= 0.0:
        return 0.0
    return rate_s_inv


def _apply_weak_perturbation(
    params: BBHEvolverParams,
    bbh: BBH_properties,
    i: int,
    a_cm: float,
    e: float,
) -> tuple[float, float]:
    """Apply the current reduced weak-BSI prescription to one binary.

    The weak channel is intentionally simple: it hardens the binary, pumps
    eccentricity, and perturbs the spin-tilt angles.  It is a placeholder for a
    future scattering/outcome kernel, but it already changes the physical BBH
    state rather than merely marking a timeline event.
    """
    delta_e = min(
        params.bsi_sigma_e0 * max(1.0 - e, params.ecc_floor),
        max(params.ecc_ceiling - e, 0.0),
    )
    delta_theta = min(delta_e, 0.5)

    e_after = float(np.clip(e + delta_e, 0.0, params.ecc_ceiling))
    a_after = float(max(a_cm * (1.0 - 0.5 * delta_e), np.finfo(float).tiny))

    bbh.bsi_weak_count[i] += 1
    bbh.bsi_delta_e_total[i] += delta_e
    bbh.bsi_delta_theta_total[i] += delta_theta
    bbh.th1[i] += delta_theta
    bbh.th2[i] -= delta_theta
    return a_after, e_after


def apply_scheduled_bsi_step(
    params: BBHEvolverParams,
    bbh: BBH_properties,
    flags: BBH_flags,
    i: int,
    a_cm: float,
    e: float,
    m1_g: float,
    m2_g: float,
    elapsed_s: float,
    step_s: float,
    *,
    bsi_active: bool,
) -> tuple[float, float, float, float, int | None]:
    """Apply a primed BSI effect when the current integration step crosses it.

    ``BSI_Timeline`` is only the ledger.  This function is the reduced local
    binary update: runtime-selected BSI events perturb eccentricity/orientation
    and harden the binary.
    """
    if not bsi_active or bbh.bsi_applied_flag[i] != 0:
        return a_cm, e, m1_g, m2_g, None

    # ``bsi_timescale_s`` is a runtime-primed delay measured from BBH formation.
    # The gas/GW integrators pass their current elapsed time and accepted step
    # size, so an event is applied exactly once when that step crosses the delay.
    delay_s = float(bbh.bsi_timescale_s[i])
    if not _event_crossed(delay_s, elapsed_s, step_s):
        return a_cm, e, m1_g, m2_g, None

    bbh.bsi_applied_flag[i] = 1
    flags.bsi_event_flag[i] = 1

    a_after, e_after = _apply_weak_perturbation(params, bbh, i, a_cm, e)
    return a_after, e_after, m1_g, m2_g, None
