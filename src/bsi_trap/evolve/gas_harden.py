# gas_harden.py, contains gas-hardening evolution functions
import numpy as np

from bsi_trap.evolve.gw import ode_gw_only
from bsi_trap.classes.classes import (
    BBHEvolverParams,
    BBH_properties,
    DiskProfile,
    AGN_properties,
    BBH_flags,
)
from bsi_trap.evolve.numeric import (
    safe_sma,
    safe_eccentricity,
    is_physical_state,
    para_evo,
)

import bsi_trap.constant.constant as const
from bsi_trap.evolve import bsi
from bsi_trap.evolve import physical_quantity as pq
from bsi_trap.evolve import stop_conditions as stop_cond


def ode_with_gas(
    Evolparams: BBHEvolverParams,
    m1_g: float,
    m2_g: float,
    a_cm: float,
    e: float,
    AGN: AGN_properties,
    Disk: DiskProfile,
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

    alpha = AGN.alpha
    r_trap_cm = Disk.r_trap_cm
    cs_cgs = Disk.get_cs(r_trap_cm)
    sig_trap_cgs = Disk.get_sigma(r_trap_cm)
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


def evolve_gas(
    Evolparams: BBHEvolverParams,
    disk: DiskProfile,
    bbh: BBH_properties,
    agn: AGN_properties,
    i: int,
) -> float:
    """
    Evolve one binary through the gas hardening phase until merger or stop condition is met.
    return t_del_s
    binary dynamics parameter and flags updates without return
    """

    flags = bbh.flags
    t_ini_s = float(bbh.t_ini_s[i])

    m1_g = bbh.m1_msun[i] * const.msun
    m2_g = bbh.m2_msun[i] * const.msun
    a_cm = bbh.sma_cm[i]
    e = bbh.ecc[i]

    freq_gw = pq.frequency_from_sma(m1_g, m2_g, a_cm)

    duty_cycle_s = agn.lifetime_myr * const.Myr

    # Time limits
    t_hubble_s = const.tHubble * const.Myr
    t_est_s = pq.compute_merger_time_estimate(m1_g, m2_g, a_cm, e)

    # Initialize
    r_th = pq.threshold_radius(m1_g, m2_g)

    h = 1e-6 * t_est_s  # Initial timestep

    t_delay_s = 0.0
    ecc10 = 0.0
    ecc0p1 = 0.0

    # # Check if binary can reach migration trap.
    # if not np.isfinite(t_ini_s) or t_ini_s > duty_cycle_s:
    #     return stop_cond._stop_for_trap_not_reached(BBH, i, flags, t_ini_s)

    # # expected normal procedure, trap reached
    # flags.trap_reached_flag[i] = 1

    # Check if already very close to merger
    # if (a_cm - r_th) / r_th < 1e-1:
    #     # this maybe the result of sma distribution sampling, too small sma is non-physical, so omitted here
    #     flags.merger_flag[i] = 0
    #     flags.stop_reason_flag[i] = const.STOP_INITIAL_BINARY_TOO_TIGHT
    #     bbh.ecc10[i] = np.nan  # Store nan since 10 Hz not reached
    #     bbh.ecc0p1[i] = np.nan  # Store nan since 10 Hz not reached
    #     return t_delay_s, t_ini_s

    # Evolution loop
    reached_10hz = False
    reached_0p1hz = False
    loop_num = 0
    max_loops = 1e6

    while (a_cm - r_th) / r_th > 1e-1 and loop_num < max_loops:
        if stop_cond._stop_for_walltime(flags, i):
            break

        freq_gw = pq.frequency_from_sma(m1_g, m2_g, a_cm)
        loop_num += 1

        # Check if we've passed 10 Hz
        if not reached_10hz and freq_gw >= 10.0:
            # Compute eccentricity at exactly 10 Hz
            ecc10 = pq.compute_eccentricity_at_frequency(
                m1_g, m2_g, a_cm, e, target_freq=10.0
            )
            reached_10hz = True

        # Check if we've passed 0.1 Hz
        if not reached_0p1hz and freq_gw >= 0.1:
            # Compute eccentricity at exactly 0.1 Hz
            ecc0p1 = pq.compute_eccentricity_at_frequency(
                m1_g, m2_g, a_cm, e, target_freq=0.1
            )
            reached_0p1hz = True

        # Check if still within AGN duty cycle
        if t_delay_s + t_ini_s <= duty_cycle_s:
            # normal evolution in the period of dutycycle
            flags.continue_ng_flag[i] = 1

            # Evolve with gas hardening
            a_new, e_new, h = para_evo(
                Evolparams, m1_g, m2_g, a_cm, e, h, ode_with_gas, agn, disk
            )
        else:
            # AGN disk closed, evolve with GW only
            flags.continue_ng_flag[i] = 0

            a_new, e_new, h = para_evo(Evolparams, m1_g, m2_g, a_cm, e, h, ode_gw_only)

        if not (
            is_physical_state(Evolparams, a_new, e_new) and np.isfinite(h) and h > 0.0
        ):
            # stop_cond._stop_for_invalid_step(flags, i)
            break

        a_cm, e = a_new, e_new
        # bsi_active = t_delay_s + t_ini_s <= duty_cycle_s

        # a, e, m1_g, m2_g, bsi_stop_reason = bsi._apply_bsi_step(
        #     BBH, flags, i, a, e, m1_g, m2_g, h, bsi_active=bsi_active
        # )

        r_th = pq.threshold_radius(m1_g, m2_g)
        freq_gw = pq.frequency_from_sma(m1_g, m2_g, a_cm)

        # Update time
        t_delay_s += h

        # if bsi_stop_reason is not None:
        #     break

        # Check time limits
        if t_delay_s + t_ini_s > t_hubble_s:
            # higher than the Hubble time, no merger & no continue ng
            flags.continue_ng_flag[i] = 0
            flags.merger_flag[i] = 0
            flags.stop_reason_flag[i] = const.STOP_HUBBLE_LIMIT
            break

        # Check if merged
        if (a_cm - r_th) / r_th <= 1e-1:
            flags.merger_flag[i] = 1
            # Store final state before break
            break

        if loop_num >= max_loops:
            print(f"Warning: Max loops reached for binary {i}")
            flags.merger_flag[i] = 0
            flags.stop_reason_flag[i] = const.STOP_INTEGRATION_LIMIT
            break

    # Store results
    bbh.ecc10[i] = ecc10
    bbh.ecc0p1[i] = ecc0p1

    return t_delay_s
