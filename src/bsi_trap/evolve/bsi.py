"""Reduced binary-single interaction updates and event-ledger construction.

The important model boundary is:

* ``apply_scheduled_bsi_step`` changes the mutable BBH state during evolution.
* ``bsi_trap.timeline.construct_bsi_timeline`` reads recorded flags/times and
  builds a ledger.

This keeps the physical update out of ``BSI_Timeline``.  Timelines answer
"when did the event happen?", while this module's step function answers "what
did that event do to the binary?".
"""

from __future__ import annotations

import numpy as np

from bsi_trap.classes.models import (
    AGN_properties,
    BBHEvolverParams,
    BBH_flags,
    BBH_properties,
    DiskProfile,
    Single_BH_properties,
)
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.constant.constant import G, Myr, msun
from bsi_trap.evolve.trap_context import build_trap_context
from bsi_trap.timeline.bsi import construct_bsi_timeline
from bsi_trap.timeline.models import FloatArray
from bsi_trap.functions.pairing_physics import (
    bbh_cross_section,
    shear_relative_velocity,
)


def _event_crossed(delay_s: float, elapsed_s: float, step_s: float) -> bool:
    """Return True when an integration step crosses a scheduled BSI delay."""
    if not (
        np.isfinite(delay_s)
        and delay_s >= 0.0
        and np.isfinite(elapsed_s)
        and np.isfinite(step_s)
        and step_s >= 0.0
    ):
        return False
    return elapsed_s <= delay_s <= elapsed_s + step_s


def _active_single_population(
    sbh: Single_BH_properties,
    duty_cycle_s: float,
) -> tuple[FloatArray, FloatArray]:
    arrival_s = np.asarray(sbh.t_ini_s, dtype=np.float64)
    mass_msun = np.asarray(sbh.m_msun, dtype=np.float64)
    valid = (
        np.isfinite(arrival_s)
        & np.isfinite(mass_msun)
        & (arrival_s < duty_cycle_s)
        & (arrival_s >= 0.0)
        & (mass_msun > 0.0)
    )
    if not np.any(valid):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    order = np.argsort(arrival_s[valid], kind="stable")
    return arrival_s[valid][order], mass_msun[valid][order]


def _bsi_impact_parameter_cm(
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    bbh: BBH_properties,
    binary_idx: int,
) -> float:
    if model_config.bsi_eta <= 0.0:
        return 0.0

    sma_cm = float(bbh.sma_cm[binary_idx])
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


def _binary_single_rate_s_inv(
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


def _sample_bsi_event_time_s(
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    bbh: BBH_properties,
    binary_idx: int,
    single_arrival_s: FloatArray,
    single_mass_msun: FloatArray,
    duty_cycle_s: float,
    rng: np.random.Generator,
) -> float | None:
    # sample the binary bsi event time for binary_idx
    ####################
    # only in avearge meaning

    start_s = float(bbh.t_ini_s[binary_idx])

    # check
    if not np.isfinite(start_s) or start_s < 0.0 or start_s >= duty_cycle_s:
        return None

    target_optical_depth = float(rng.exponential(1.0))

    accumulated = 0.0

    current_s = start_s

    # get next arrive single index
    #################################
    # next arrive means change for singel bh population
    active_count = int(np.searchsorted(single_arrival_s, current_s, side="right"))

    while current_s < duty_cycle_s:
        next_arrival_s = (
            float(single_arrival_s[active_count])
            if active_count < single_arrival_s.size
            else duty_cycle_s
        )

        # bsi interval terminal time
        interval_end_s = min(next_arrival_s, duty_cycle_s)

        if interval_end_s > current_s and active_count > 0:

            # calculate bsi rate for getting bsi timeline
            rate_s_inv = _binary_single_rate_s_inv(
                model_config,
                agn,
                disk,
                bbh,
                binary_idx,
                single_mass_msun[:active_count],
            )

            if rate_s_inv > 0.0:
                interval_depth = rate_s_inv * (interval_end_s - current_s)

                if accumulated + interval_depth >= target_optical_depth:
                    return float(
                        current_s + (target_optical_depth - accumulated) / rate_s_inv
                    )
                accumulated += interval_depth

        # set current time to interval terminal time
        current_s = interval_end_s

        while (
            active_count < single_arrival_s.size
            and single_arrival_s[active_count] <= current_s
        ):
            # consider next active count
            active_count += 1

        # break condition
        if active_count >= single_arrival_s.size and current_s >= duty_cycle_s:
            break

    return None


def schedule_bsi_events(
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    bbh: BBH_properties,
    sbh: Single_BH_properties,
    rng: np.random.Generator,
) -> np.ndarray:
    """Schedule reduced BSI delays from the live single-BH reservoir.

    Delays are measured from BBH formation and written to
    ``bbh.bsi_timescale_s``. Existing finite nonnegative delays are preserved so
    focused tests and deterministic fixtures can force a specific event. The
    orbital perturbation itself is applied later by ``apply_scheduled_bsi_step``.
    """
    # each value means if get next bsi event time
    scheduled_mask = np.zeros(bbh.num, dtype=bool)

    if bbh.num == 0 or sbh.num == 0:
        return scheduled_mask

    duty_cycle_s = float(agn.lifetime_myr * Myr)
    if not np.isfinite(duty_cycle_s) or duty_cycle_s <= 0.0:
        return scheduled_mask

    single_arrival_s, single_mass_msun = _active_single_population(sbh, duty_cycle_s)
    if single_arrival_s.size == 0:
        return scheduled_mask

    bbh_bsi_event_coll = np.asarray(bbh.bsi_timescale_s, dtype=np.float64)

    for binary_idx in range(bbh.num):

        # check
        if (
            np.isfinite(bbh_bsi_event_coll[binary_idx])
            and bbh_bsi_event_coll[binary_idx] >= 0.0
        ):
            continue

        t_bsi_abo_s = _sample_bsi_event_time_s(
            model_config,
            agn,
            disk,
            bbh,
            binary_idx,
            single_arrival_s,
            single_mass_msun,
            duty_cycle_s,
            rng,
        )

        if t_bsi_abo_s is None:
            continue

        t_bsi_rel_s = t_bsi_abo_s - float(bbh.t_ini_s[binary_idx])

        if (
            np.isfinite(t_bsi_rel_s)
            and t_bsi_rel_s >= 0.0
            and t_bsi_abo_s <= duty_cycle_s
        ):
            bbh.bsi_timescale_s[binary_idx] = t_bsi_rel_s
            scheduled_mask[binary_idx] = True

    return scheduled_mask


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
    """Apply the scheduled BSI effect when the current integration step crosses it.

    ``BSI_Timeline`` is only the ledger.  This function is the reduced local
    binary update: scheduled BSI events perturb eccentricity/orientation and
    harden the binary.
    """
    if not bsi_active or bbh.bsi_applied_flag[i] != 0:
        return a_cm, e, m1_g, m2_g, None

    # ``bsi_timescale_s`` is a delay measured from BBH formation.  The gas/GW
    # integrators pass their current elapsed time and accepted step size, so an
    # event is applied exactly once when that step crosses the scheduled delay.
    delay_s = float(bbh.bsi_timescale_s[i])
    if not _event_crossed(delay_s, elapsed_s, step_s):
        return a_cm, e, m1_g, m2_g, None

    bbh.bsi_applied_flag[i] = 1
    flags.bsi_event_flag[i] = 1

    a_after, e_after = _apply_weak_perturbation(params, bbh, i, a_cm, e)
    return a_after, e_after, m1_g, m2_g, None
