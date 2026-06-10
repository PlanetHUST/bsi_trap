import numpy as np
import logging

from bsi_trap.classes.classes import (
    BBHEvolver,
    BBH_properties,
    AGN_properties,
    merger_flags,
)
import bsi_trap.constant.constant as const
from bsi_trap.evolve import bsi
from bsi_trap.evolve import numeric
from bsi_trap.evolve import physical_quantity as pq
from bsi_trap.evolve import stop_conditions as stop_cond


def ode_gw_only(m1, m2, a, e):
    """Peters 1964 GW evolution formulas"""
    a = numeric._safe_sma(a)
    e = numeric._safe_eccentricity(e)
    if not numeric._is_physical_state(a, e):
        logging.warning(f"Non-physical state detected in GW evolution: a={a}, e={e}")
        return np.nan, np.nan

    factor = const.G3 * m1 * m2 * (m1 + m2) / (const.c5 * a**3 * (1 - e**2) ** 3.5)

    da_dt = -64 / 5 * factor * (1 + 73 / 24 * e**2 + 37 / 96 * e**4)
    de_dt = -304 / 15 * e * factor / (a * (1 - e**2) ** -1) * (1 + 121 / 304 * e**2)

    return da_dt, de_dt


def evolve_gw_only(
    BBH: BBH_properties,
    AGN: AGN_properties,
    i: int,
    flags: merger_flags,
    t_ini_s: float,
):
    """
    Evolve one binary with GW emission only.
    """

    # Extract parameters
    m1 = BBH.m1[i] * const.msun
    m2 = BBH.m2[i] * const.msun
    a = BBH.sma[i]
    e = BBH.ecc[i]
    R = BBH.A[i]  # outer orbital semimajor axis in cm

    # Time limits
    t_hubble_s = const.tHubble * 1e6 * const.yr
    duty_cycle_s = AGN.AGNlifetime * 1e6 * const.yr
    t_est = pq.compute_merger_time_estimate(m1, m2, a, e)

    # Initialize
    r_th = pq.threshold_radius(m1, m2)
    h = 0.01 * t_est
    t_delay_s = 0.0
    ecc10 = 0.0
    ecc0p1 = 0.0

    # Check if binary can reach migration trap.
    if not np.isfinite(t_ini_s) or t_ini_s > duty_cycle_s:
        return stop_cond._stop_for_trap_not_reached(BBH, i, flags, t_ini_s)

    flags.trap_reached_flag[i] = 1

    # Check if already very close to merger
    if (a - r_th) / r_th < 1e-1:
        flags.merger_flag[i] = 0
        flags.stop_reason_flag[i] = const.STOP_INITIAL_BINARY_TOO_TIGHT
        BBH.ecc10[i] = np.nan  # Store nan since 10 Hz not reached
        BBH.ecc0p1[i] = np.nan  # Store nan since 10 Hz not reached
        return BBH, t_delay_s, t_ini_s, flags

    # Store initial state
    freq_gw = 1.0 / np.pi * np.sqrt(const.G * (m1 + m2) / a**3)

    # Initialize tracking variables
    reached_10hz = False
    reached_0p1hz = False
    loop_num = 0

    # Evolution loop
    while (a - r_th) / r_th > 1e-1:
        if stop_cond._stop_for_walltime(flags, i):
            break

        freq_gw = 1.0 / np.pi * np.sqrt(const.G * (m1 + m2) / a**3)
        loop_num += 1

        # Check if still within AGN duty cycle
        if t_delay_s + t_ini_s <= duty_cycle_s:
            flags.continue_ng_flag[i] = 1
        else:
            flags.continue_ng_flag[i] = 0

        # Check if we've passed 10 Hz
        if not reached_10hz and freq_gw >= 10.0:
            # Compute eccentricity at exactly 10 Hz
            ecc10 = pq.compute_eccentricity_at_frequency(m1, m2, a, e, target_freq=10.0)
            reached_10hz = True

        # Check if we've passed 0.1 Hz
        if not reached_0p1hz and freq_gw >= 0.1:
            # Compute eccentricity at exactly 10 Hz
            ecc0p1 = pq.compute_eccentricity_at_frequency(m1, m2, a, e, target_freq=0.1)
            reached_0p1hz = True

        # Evolve with GW only
        a_new, e_new, h = numeric.para_evo(m1, m2, a, e, h, ode_gw_only)
        if not (
            numeric._is_physical_state(a_new, e_new) and np.isfinite(h) and h > 0.0
        ):
            stop_cond._stop_for_invalid_step(flags, i)
            break

        # Update parameters
        a, e = a_new, e_new
        bsi_active = t_delay_s + t_ini_s <= duty_cycle_s
        a, e, m1, m2, bsi_stop_reason = bsi._apply_bsi_step(
            BBH, flags, i, a, e, m1, m2, h, bsi_active=bsi_active
        )
        r_th = pq.threshold_radius(m1, m2)
        freq_gw = 1.0 / np.pi * np.sqrt(const.G * (m1 + m2) / a**3)
        t_delay_s += h

        if bsi_stop_reason is not None:
            break

        # Check time limits
        if t_delay_s + t_ini_s > t_hubble_s:
            flags.merger_flag[i] = 0
            flags.continue_ng_flag[i] = 0
            flags.stop_reason_flag[i] = const.STOP_HUBBLE_LIMIT
            break

        # Check if merged
        if (a - r_th) / r_th <= 1e-1:
            flags.merger_flag[i] = 1
            break

        if loop_num >= 1e6:
            print(f"Warning: Max loops reached for binary {i}")
            flags.merger_flag[i] = 0
            flags.stop_reason_flag[i] = const.STOP_INTEGRATION_LIMIT
            break

    # Store results
    BBH.sma2[i] = a
    BBH.ecc2[i] = e
    BBH.ecc10[i] = ecc10
    BBH.ecc0p1[i] = ecc0p1

    return BBH, t_delay_s, t_ini_s, flags
