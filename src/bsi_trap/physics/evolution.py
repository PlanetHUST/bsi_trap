"""Compact BBH evolution physics for the demo.

The module keeps the main AGN-BBH merger ingredients visible:

* gravitational-wave hardening,
* gas hardening at the trap,
* weak binary-single interaction perturbations,
* final BBH merger/remnant estimates.
"""

from __future__ import annotations

import numpy as np

from bsi_trap.simulation.models import (
    ArrivalTimeline,
    BinaryPopulation,
    DemoConfig,
    DiskProfile,
    EmbeddedPopulation,
    EvolutionTimeline,
    PairingTimeline,
)
from bsi_trap.units import AU, C, G, MSUN, MYR


def build_binary_population(
    config: DemoConfig,
    population: EmbeddedPopulation,
    pairing: PairingTimeline,
) -> BinaryPopulation:
    first_idx = pairing.first_idx.astype(np.intp, copy=True)
    second_idx = pairing.second_idx.astype(np.intp, copy=True)
    size = pairing.size
    return BinaryPopulation(
        first_idx=first_idx,
        second_idx=second_idx,
        m1_msun=population.mass_msun[first_idx].astype(float, copy=True),
        m2_msun=population.mass_msun[second_idx].astype(float, copy=True),
        semi_major_axis_cm=np.full(size, config.initial_bbh_sma_au * AU),
        eccentricity=np.full(size, config.initial_bbh_eccentricity),
        formation_s=pairing.formation_s.astype(float, copy=True),
    )


def merger_radius_cm(m1_msun: float, m2_msun: float) -> float:
    """Use 6 GM/c^2 as the compact merger boundary."""
    return 6.0 * G * (m1_msun + m2_msun) * MSUN / C**2


def gw_merger_time_s(
    m1_msun: float,
    m2_msun: float,
    semi_major_axis_cm: float,
    eccentricity: float,
) -> float:
    """Peters-style GW merger time estimate."""
    e = float(np.clip(eccentricity, 0.0, 0.999))
    m1_g = m1_msun * MSUN
    m2_g = m2_msun * MSUN
    return float(
        5.0
        / 256.0
        * C**5
        / G**3
        * semi_major_axis_cm**4
        * (1.0 - e**2) ** 3.5
        / (m1_g * m2_g * (m1_g + m2_g))
    )


def gas_hardening_time_s(
    config: DemoConfig,
    disk: DiskProfile,
    m1_msun: float,
    m2_msun: float,
    semi_major_axis_cm: float,
    eccentricity: float,
) -> float:
    """Local gas-hardening estimate from the trap gas properties."""
    r_merge = merger_radius_cm(m1_msun, m2_msun)
    if semi_major_axis_cm <= r_merge:
        return 0.0

    e = float(np.clip(eccentricity, 0.0, 0.999))
    m1_g = m1_msun * MSUN
    m2_g = m2_msun * MSUN
    mu_g = m1_g * m2_g / (m1_g + m2_g)
    coeff = (
        24.0
        * np.pi
        * config.alpha
        * disk.trap_cs_cm_s**2
        * disk.trap_sigma_g_cm2
        * (1.0 + e) ** 2
        / (mu_g * np.sqrt(G * (m1_g + m2_g)))
    )
    if not np.isfinite(coeff) or coeff <= 0.0:
        return np.inf
    return float(
        2.0 / (3.0 * coeff) * (r_merge**-1.5 - semi_major_axis_cm**-1.5)
    )


def _trap_volume_cm3(disk: DiskProfile) -> float:
    height_cm = 2.0 * disk.trap_scale_height_cm
    width_cm = max(height_cm, 0.03 * disk.r_trap_cm)
    return float(2.0 * np.pi * disk.r_trap_cm * width_cm * height_cm)


def binary_single_delay_s(
    config: DemoConfig,
    disk: DiskProfile,
    m1_msun: float,
    m2_msun: float,
    semi_major_axis_cm: float,
    active_single_masses_msun: np.ndarray,
) -> float:
    """Estimate the first weak binary-single encounter delay."""
    if active_single_masses_msun.size == 0:
        return np.inf

    binary_mass_msun = m1_msun + m2_msun
    hill_radius_cm = disk.r_trap_cm * (
        binary_mass_msun / (3.0 * config.smbh_mass_msun)
    ) ** (1.0 / 3.0)
    p_max_cm = min(
        config.bsi_cross_section_sma * semi_major_axis_cm,
        hill_radius_cm,
        disk.trap_scale_height_cm,
    )
    volume_cm3 = _trap_volume_cm3(disk)
    if p_max_cm <= 0.0 or volume_cm3 <= 0.0:
        return np.inf

    omega_smbh = np.sqrt(G * config.smbh_mass_msun * MSUN / disk.r_trap_cm**3)
    v_inf_cm_s = max(1.5 * omega_smbh * p_max_cm, 0.01 * disk.trap_cs_cm_s)
    binary_mass_g = binary_mass_msun * MSUN
    single_mass_g = np.asarray(active_single_masses_msun, dtype=float) * MSUN
    focusing = 1.0 + 2.0 * G * (binary_mass_g + single_mass_g) / (
        p_max_cm * v_inf_cm_s**2
    )
    rate_s_inv = float(
        np.sum(np.pi * p_max_cm**2 * focusing * v_inf_cm_s) / volume_cm3
    )
    if not np.isfinite(rate_s_inv) or rate_s_inv <= 0.0:
        return np.inf
    return 1.0 / rate_s_inv


def apply_bsi_perturbation(
    config: DemoConfig,
    semi_major_axis_cm: float,
    eccentricity: float,
) -> tuple[float, float]:
    """Weak BSI hardens the BBH and pumps eccentricity once."""
    hardened_a_cm = semi_major_axis_cm * (1.0 - config.bsi_hardening_fraction)
    pumped_e = eccentricity + config.bsi_eccentricity_pump * (1.0 - eccentricity)
    return float(hardened_a_cm), float(np.clip(pumped_e, 0.0, 0.95))


def merge_remnant(m1_msun: float, m2_msun: float) -> tuple[float, float]:
    """Return simple remnant mass and spin estimates."""
    total_mass = m1_msun + m2_msun
    eta = m1_msun * m2_msun / total_mass**2
    final_mass_msun = 0.95 * total_mass
    final_spin = np.clip(3.464 * eta - 2.029 * eta**2, 0.0, 0.99)
    return float(final_mass_msun), float(final_spin)


def _single_background_for_binary(
    population: EmbeddedPopulation,
    arrivals: ArrivalTimeline,
    first_idx: int,
    second_idx: int,
) -> np.ndarray:
    active = arrivals.reached_mask.copy()
    active[first_idx] = False
    active[second_idx] = False
    return population.mass_msun[active]


def evolve_bbh_population(
    config: DemoConfig,
    disk: DiskProfile,
    population: EmbeddedPopulation,
    arrivals: ArrivalTimeline,
    pairing: PairingTimeline,
) -> tuple[BinaryPopulation, EvolutionTimeline]:
    binaries = build_binary_population(config, population, pairing)
    size = binaries.size
    binary_idx = np.arange(size, dtype=np.intp)
    gas_delay_s = np.empty(size, dtype=float)
    gw_delay_s = np.empty(size, dtype=float)
    bsi_event_s = np.full(size, np.nan, dtype=float)
    delay_s = np.empty(size, dtype=float)
    end_s = np.empty(size, dtype=float)
    merger_s = np.full(size, np.nan, dtype=float)
    merged = np.zeros(size, dtype=bool)
    final_mass_msun = np.full(size, np.nan, dtype=float)
    final_spin = np.full(size, np.nan, dtype=float)
    channel = np.full(size, "agn-stop", dtype="<U12")
    lifetime_s = config.agn_lifetime_myr * MYR

    for row in range(size):
        m1 = float(binaries.m1_msun[row])
        m2 = float(binaries.m2_msun[row])
        a0 = float(binaries.semi_major_axis_cm[row])
        e0 = float(binaries.eccentricity[row])
        start = float(binaries.formation_s[row])
        remaining_s = max(lifetime_s - start, 0.0)

        gas0 = gas_hardening_time_s(config, disk, m1, m2, a0, e0)
        gw0 = gw_merger_time_s(m1, m2, a0, e0)
        gas_delay_s[row] = gas0
        gw_delay_s[row] = gw0

        smooth_delay = min(gas0, gw0)
        smooth_channel = "gas" if gas0 <= gw0 else "gw"
        selected_delay = smooth_delay
        selected_channel = smooth_channel

        singles = _single_background_for_binary(
            population,
            arrivals,
            int(binaries.first_idx[row]),
            int(binaries.second_idx[row]),
        )
        bsi_delay = binary_single_delay_s(config, disk, m1, m2, a0, singles)
        if bsi_delay < selected_delay and bsi_delay < remaining_s:
            a_after, e_after = apply_bsi_perturbation(config, a0, e0)
            gas_after = gas_hardening_time_s(config, disk, m1, m2, a_after, e_after)
            gw_after = gw_merger_time_s(m1, m2, a_after, e_after)
            post_delay = min(gas_after, gw_after)
            post_channel = "gas" if gas_after <= gw_after else "gw"
            selected_delay = bsi_delay + post_delay
            selected_channel = f"bsi+{post_channel}"
            bsi_event_s[row] = start + bsi_delay

        if not np.isfinite(selected_delay):
            selected_delay = remaining_s

        did_merge = selected_delay <= remaining_s
        merged[row] = did_merge
        delay_s[row] = selected_delay if did_merge else remaining_s
        end_s[row] = start + delay_s[row]
        channel[row] = selected_channel if did_merge else "agn-stop"
        if did_merge:
            merger_s[row] = end_s[row]
            final_mass_msun[row], final_spin[row] = merge_remnant(m1, m2)

    return binaries, EvolutionTimeline(
        binary_idx=binary_idx,
        start_s=binaries.formation_s,
        gas_delay_s=gas_delay_s,
        gw_delay_s=gw_delay_s,
        bsi_event_s=bsi_event_s,
        delay_s=delay_s,
        end_s=end_s,
        merger_s=merger_s,
        merged=merged,
        final_mass_msun=final_mass_msun,
        final_spin=final_spin,
        channel=channel,
    )


__all__ = [
    "apply_bsi_perturbation",
    "binary_single_delay_s",
    "build_binary_population",
    "evolve_bbh_population",
    "gas_hardening_time_s",
    "gw_merger_time_s",
    "merge_remnant",
    "merger_radius_cm",
]
