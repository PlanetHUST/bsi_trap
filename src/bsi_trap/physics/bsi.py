"""Discrete binary-single encounter rates and a labelled toy perturbation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bsi_trap.physics.pairing import hill_radius_cm
from bsi_trap.simulation.models import BinaryPopulation, DemoConfig, DiskProfile
from bsi_trap.units import G_CGS, MSUN_G


@dataclass(frozen=True)
class EncounterEnvironment:
    active_single_ids: np.ndarray
    per_single_rate_s_inv: np.ndarray
    total_rate_s_inv: float
    maximum_impact_parameter_cm: float
    relative_velocity_cm_s: float
    trap_volume_cm3: float


@dataclass(frozen=True)
class BSIEventSample:
    delay_s: float
    single_bh_id: int | None
    encounter_rate_s_inv: float


def calculate_encounter_environment(
    config: DemoConfig,
    disk: DiskProfile,
    binary: BinaryPopulation,
    active_single_ids: np.ndarray,
    active_single_masses_msun: np.ndarray,
) -> EncounterEnvironment:
    """Return gravitationally focused encounter rates for active singles.

    The cross-section is pi b_max^2 [1 + 2 G M_tot/(b_max v_inf^2)].
    All inputs are CGS except explicitly suffixed solar masses.
    """
    active_ids = np.asarray(active_single_ids, dtype=np.int64)
    single_masses_msun = np.asarray(active_single_masses_msun, dtype=float)
    if active_ids.shape != single_masses_msun.shape:
        raise ValueError("active single IDs and masses must have the same shape")
    if active_ids.ndim != 1:
        raise ValueError("active single arrays must be one-dimensional")
    if active_ids.size == 0:
        return EncounterEnvironment(
            active_single_ids=active_ids,
            per_single_rate_s_inv=np.array([], dtype=float),
            total_rate_s_inv=0.0,
            maximum_impact_parameter_cm=0.0,
            relative_velocity_cm_s=0.0,
            trap_volume_cm3=0.0,
        )
    if not np.all(np.isfinite(single_masses_msun) & (single_masses_msun > 0.0)):
        raise ValueError("active single masses [Msun] must be finite and positive")

    radius_cm = binary.formation_radius_cm
    disk_index = disk.closest_index(radius_cm)
    scale_height_cm = float(disk.scale_height_cm[disk_index])
    sound_speed_cm_s = float(disk.sound_speed_cm_s[disk_index])
    binary_hill_radius_cm = hill_radius_cm(
        radius_cm,
        binary.m1_msun,
        binary.m2_msun,
        config.smbh_mass_msun,
    )
    maximum_impact_parameter_cm = min(
        config.bsi_cross_section_sma * binary.sma_cm,
        binary_hill_radius_cm,
        scale_height_cm,
    )
    annulus_height_cm = 2.0 * scale_height_cm
    annulus_width_cm = max(annulus_height_cm, 0.03 * radius_cm)
    trap_volume_cm3 = (
        2.0 * np.pi * radius_cm * annulus_width_cm * annulus_height_cm
    )
    if maximum_impact_parameter_cm <= 0.0 or trap_volume_cm3 <= 0.0:
        raise ValueError("encounter geometry must be positive")
    omega_s_inv = float(disk.omega_s_inv[disk_index])
    relative_velocity_cm_s = max(
        1.5 * omega_s_inv * maximum_impact_parameter_cm,
        0.01 * sound_speed_cm_s,
    )
    binary_mass_g = (binary.m1_msun + binary.m2_msun) * MSUN_G
    single_mass_g = single_masses_msun * MSUN_G
    gravitational_focusing = 1.0 + (
        2.0
        * G_CGS
        * (binary_mass_g + single_mass_g)
        / (maximum_impact_parameter_cm * relative_velocity_cm_s**2)
    )
    cross_section_cm2 = (
        np.pi * maximum_impact_parameter_cm**2 * gravitational_focusing
    )
    per_single_rate_s_inv = (
        cross_section_cm2 * relative_velocity_cm_s / trap_volume_cm3
    )
    total_rate_s_inv = float(np.sum(per_single_rate_s_inv))
    if not np.isfinite(total_rate_s_inv) or total_rate_s_inv <= 0.0:
        raise ValueError("encounter rate [s^-1] must be finite and positive")
    return EncounterEnvironment(
        active_single_ids=active_ids,
        per_single_rate_s_inv=per_single_rate_s_inv,
        total_rate_s_inv=total_rate_s_inv,
        maximum_impact_parameter_cm=float(maximum_impact_parameter_cm),
        relative_velocity_cm_s=float(relative_velocity_cm_s),
        trap_volume_cm3=float(trap_volume_cm3),
    )


def sample_next_bsi_event(
    environment: EncounterEnvironment,
    rng: np.random.Generator,
) -> BSIEventSample:
    """Sample the next Poisson encounter delay and participating physical ID."""
    if environment.total_rate_s_inv == 0.0:
        return BSIEventSample(np.inf, None, 0.0)
    delay_s = float(rng.exponential(1.0 / environment.total_rate_s_inv))
    probabilities = (
        environment.per_single_rate_s_inv / environment.total_rate_s_inv
    )
    single_bh_id = int(rng.choice(environment.active_single_ids, p=probabilities))
    return BSIEventSample(
        delay_s=delay_s,
        single_bh_id=single_bh_id,
        encounter_rate_s_inv=environment.total_rate_s_inv,
    )


def apply_phenomenological_bsi(
    config: DemoConfig,
    sma_cm: float,
    eccentricity: float,
) -> tuple[float, float]:
    """Apply the project's labelled toy hardening/eccentricity prescription."""
    if sma_cm <= 0.0 or not 0.0 <= eccentricity < 1.0:
        raise ValueError("binary state is outside the BSI prescription domain")
    hardened_sma_cm = sma_cm * (1.0 - config.bsi_hardening_fraction)
    pumped_eccentricity = eccentricity + config.bsi_eccentricity_pump * (
        1.0 - eccentricity
    )
    if hardened_sma_cm <= 0.0 or not 0.0 <= pumped_eccentricity < 1.0:
        raise ValueError("BSI prescription produced a nonphysical binary state")
    return float(hardened_sma_cm), float(pumped_eccentricity)


__all__ = [
    "BSIEventSample",
    "EncounterEnvironment",
    "apply_phenomenological_bsi",
    "calculate_encounter_environment",
    "sample_next_bsi_event",
]
