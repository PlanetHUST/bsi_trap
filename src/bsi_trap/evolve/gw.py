import numpy as np
import logging

from bsi_trap.classes.models import (
    BBHEvolverParams,
    BBH_properties,
    AGN_properties,
    BBH_flags,
)
import bsi_trap.constant.constant as const
from bsi_trap.constant.stop_reason import StopReasons
from bsi_trap.evolve import bsi
from bsi_trap.evolve import numeric
from bsi_trap.evolve import physical_quantity as pq
from bsi_trap.evolve import stop_conditions as stop_cond
from bsi_trap.evolve.tracks import append_bbh_track


def ode_gw_only(Evolparams, m1, m2, a, e):
    """Peters 1964 GW evolution formulas"""
    a = numeric.safe_sma(Evolparams, a)
    e = numeric.safe_eccentricity(Evolparams, e)
    if not numeric.is_physical_state(Evolparams, a, e):
        logging.warning(f"Non-physical state detected in GW evolution: a={a}, e={e}")
        return np.nan, np.nan

    factor = const.G3 * m1 * m2 * (m1 + m2) / (const.c5 * a**3 * (1 - e**2) ** 3.5)

    da_dt = -64 / 5 * factor * (1 + 73 / 24 * e**2 + 37 / 96 * e**4)
    de_dt = -304 / 15 * e * factor / (a * (1 - e**2) ** -1) * (1 + 121 / 304 * e**2)

    return da_dt, de_dt


def evolve_gw_only(
    Evolparams,
    bbh: BBH_properties,
    agn: AGN_properties,
    i: int,
    t_ini_s: float,
    *,
    bsi_enabled: bool = False,
) -> float:
    """
    Evolve one binary with GW emission only.
    """

    # Extract parameters
    m1 = bbh.m1_msun[i] * const.msun
    m2 = bbh.m2_msun[i] * const.msun
    a = bbh.sma_cm[i]
    e = bbh.ecc[i]

    flags = bbh.flags

    # Time limits
    t_hubble_s = const.tHubble * const.Myr
    duty_cycle_s = agn.lifetime_myr * const.Myr
    time_limit_s = min(duty_cycle_s, t_hubble_s)
    time_limit_reason = (
        const.STOP_HUBBLE_LIMIT
        if t_hubble_s <= duty_cycle_s
        else StopReasons.STOP_AGN_LIFETIME
    )
    t_est = pq.compute_merger_time_estimate(m1, m2, a, e)

    # Initialize
    r_th = pq.threshold_radius(m1, m2)
    h = 0.01 * t_est
    t_delay_s = 0.0
    ecc10 = 0.0
    ecc0p1 = 0.0
    track_time_s = [float(t_ini_s)]
    track_sma_cm = [float(a)]
    track_ecc = [float(e)]

    # # Check if binary can reach migration trap.
    # if not np.isfinite(t_ini_s) or t_ini_s > duty_cycle_s:
    #     return stop_cond._stop_for_trap_not_reached(bbh, i, flags, t_ini_s)

    flags.trap_reached_flag[i] = 1

    # # Check if already very close to merger
    # if (a - r_th) / r_th < 1e-1:
    #     flags.merger_flag[i] = 0
    #     flags.stop_reason_flag[i] = const.STOP_INITIAL_BINARY_TOO_TIGHT
    #     bbh.ecc10[i] = np.nan  # Store nan since 10 Hz not reached
    #     bbh.ecc0p1[i] = np.nan  # Store nan since 10 Hz not reached
    #     return bbh, t_delay_s, flags

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

        remaining_time_s = time_limit_s - (t_ini_s + t_delay_s)
        if remaining_time_s <= 0.0:
            flags.merger_flag[i] = 0
            flags.continue_ng_flag[i] = 0
            flags.stop_reason_flag[i] = time_limit_reason
            break

        smooth_step_limit_s = remaining_time_s
        bsi_active = (
            bsi_enabled
            and bbh.bsi_applied_flag[i] == 0
            and t_delay_s + t_ini_s <= duty_cycle_s
        )
        if bsi_active:
            bsi_delay_s = float(bbh.bsi_timescale_s[i])
            if np.isfinite(bsi_delay_s) and bsi_delay_s >= t_delay_s:
                time_to_bsi_s = bsi_delay_s - t_delay_s
                if time_to_bsi_s <= 0.0:
                    was_applied = bbh.bsi_applied_flag[i]
                    a, e, m1, m2, bsi_stop_reason = bsi.apply_scheduled_bsi_step(
                        Evolparams,
                        bbh,
                        flags,
                        i,
                        a,
                        e,
                        m1,
                        m2,
                        t_delay_s,
                        0.0,
                        bsi_active=True,
                    )
                    if bbh.bsi_applied_flag[i] != was_applied:
                        track_time_s.append(float(t_ini_s + t_delay_s))
                        track_sma_cm.append(float(a))
                        track_ecc.append(float(e))
                    if bsi_stop_reason is not None:
                        flags.merger_flag[i] = 0
                        flags.continue_ng_flag[i] = 0
                        flags.stop_reason_flag[i] = bsi_stop_reason
                        break
                else:
                    smooth_step_limit_s = min(smooth_step_limit_s, time_to_bsi_s)

        freq_gw = 1.0 / np.pi * np.sqrt(const.G * (m1 + m2) / a**3)
        loop_num += 1

        flags.continue_ng_flag[i] = 1

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

        def record_smooth_state(elapsed_s, sma_cm, eccentricity):
            track_time_s.append(float(t_ini_s + t_delay_s + elapsed_s))
            track_sma_cm.append(float(sma_cm))
            track_ecc.append(float(eccentricity))

        # Evolve with GW only
        a_new, e_new, h = numeric.para_evo(
            Evolparams,
            m1,
            m2,
            a,
            e,
            h,
            ode_gw_only,
            max_step_s=smooth_step_limit_s,
            state_callback=record_smooth_state,
        )
        if not (
            numeric.is_physical_state(Evolparams, a_new, e_new)
            and np.isfinite(h)
            and h > 0.0
        ):
            stop_cond.stop_for_invalid_step(flags, i)
            break

        a, e = a_new, e_new

        # BSI is applied while the current orbital state is live.  The BSI
        # timeline is constructed after evolution from the recorded flags and
        # scheduled delay.
        was_applied = bbh.bsi_applied_flag[i]
        bsi_active = (
            bsi_enabled
            and was_applied == 0
            and t_delay_s + t_ini_s <= duty_cycle_s
        )
        a, e, m1, m2, bsi_stop_reason = bsi.apply_scheduled_bsi_step(
            Evolparams,
            bbh,
            flags,
            i,
            a,
            e,
            m1,
            m2,
            t_delay_s,
            h,
            bsi_active=bsi_active,
        )
        if bbh.bsi_applied_flag[i] != was_applied:
            track_time_s.append(float(t_ini_s + bbh.bsi_timescale_s[i]))
            track_sma_cm.append(float(a))
            track_ecc.append(float(e))

        if bsi_stop_reason is not None:
            flags.merger_flag[i] = 0
            flags.continue_ng_flag[i] = 0
            flags.stop_reason_flag[i] = bsi_stop_reason
            t_delay_s = float(bbh.bsi_timescale_s[i])
            break

        r_th = pq.threshold_radius(m1, m2)
        freq_gw = 1.0 / np.pi * np.sqrt(const.G * (m1 + m2) / a**3)
        t_delay_s += h

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

        if t_delay_s + t_ini_s >= duty_cycle_s:
            flags.merger_flag[i] = 0
            flags.continue_ng_flag[i] = 0
            flags.stop_reason_flag[i] = time_limit_reason
            break

        if loop_num >= 1e6:
            print(f"Warning: Max loops reached for binary {i}")
            flags.merger_flag[i] = 0
            flags.stop_reason_flag[i] = const.STOP_INTEGRATION_LIMIT
            break

    # Store results
    bbh.sma_cm[i] = a
    bbh.ecc[i] = e
    bbh.ecc10[i] = ecc10
    bbh.ecc0p1[i] = ecc0p1
    append_bbh_track(
        bbh,
        binary_idx=i,
        time_s=track_time_s,
        sma_cm=track_sma_cm,
        ecc=track_ecc,
    )

    return t_delay_s
