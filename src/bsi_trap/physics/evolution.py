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
    BinaryCollect,
    DemoConfig,
    DiskProfile,
    EmbeddedPopulation,
    EvolutionTimeline,
    PairingTimeline,
)
from bsi_trap.units import AU, C, G, MSUN, MYR


def merger_radius_cm(m1_msun: float, m2_msun: float) -> float:
    """Use 6 GM/c^2 as the compact merger boundary."""
    return 6.0 * G * (m1_msun + m2_msun) * MSUN / C**2


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
    rate_s_inv = float(np.sum(np.pi * p_max_cm**2 * focusing * v_inf_cm_s) / volume_cm3)
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
    """demo version remnant"""
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


def gw_acc(
    config: DemoConfig, current_time: float, bbh: BinaryPopulation
) -> tuple[float, float]:
    dadt = 0
    dedt = 0
    return dadt, dedt


def gas_acc(
    config: DemoConfig, current_time: float, disk: DiskProfile, bbh: BinaryPopulation
) -> tuple[float, float]:
    dadt = 0
    dedt = 0
    return dadt, dedt


def apply_state_change(
    config, current_time_s, step_time_s, bbh, *, dadt, dedt, delta_a, delta_e
):
    pass


def evolve_kernal(
    current_time_s: float,
    config: DemoConfig,
    disk: DiskProfile,
    population: EmbeddedPopulation,
    arrivals: ArrivalTimeline,
    bbh_list: BinaryCollect,
):
    """smooth evolution as well as bsi"""
