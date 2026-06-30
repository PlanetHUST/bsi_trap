"""Pure physical quantities used by initial single-BH pairing dynamics."""

from __future__ import annotations

import numpy as np

from bsi_trap.classes.models import DiskProfile, TrapContext
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.constant.constant import G, msun


def pair_up_timescale(
    disk: DiskProfile,
    m1_msun: float,
    m2_msun: float,
    trap_radius_cm: float | None = None,
) -> float:
    """Return the gas-driven BBH pair-up delay in seconds.

    This is the gas dynamical-friction estimate evaluated at the migration
    trap,

    ``t_pair = 3 c_s^3 / (4 pi G^2 rho_g M_tot)``,

    using the vertically averaged midplane density
    ``rho_g = Sigma / (2 H)``. Both component masses are in solar masses; disk
    quantities are in CGS units. The result is a delay after capture selection,
    not an independently sampled encounter waiting time.
    """
    if m1_msun <= 0.0 or m2_msun <= 0.0:
        raise ValueError("BBH component masses must be positive")

    if trap_radius_cm is None:
        trap_radius_cm = disk.r_trap_cm
    if trap_radius_cm <= 0.0:
        raise ValueError("Pairing radius must be positive")
    sound_speed_cgs = disk.get_cs(trap_radius_cm)
    scale_height_cm = disk.get_scale_height_cm(trap_radius_cm)
    surface_density_cgs = disk.get_sigma(trap_radius_cm)
    if sound_speed_cgs <= 0.0 or scale_height_cm <= 0.0 or surface_density_cgs <= 0.0:
        raise ValueError(
            "Trap disk sound speed, scale height, and density must be positive"
        )

    total_mass_g = (m1_msun + m2_msun) * msun
    midplane_density_cgs = surface_density_cgs / (2.0 * scale_height_cm)
    return float(
        3.0
        * sound_speed_cgs**3
        / (4.0 * np.pi * G**2 * midplane_density_cgs * total_mass_g)
    )


def shear_relative_velocity(
    smbh_mass_msun: float,
    first_radius_cm: float,
    second_radius_cm: float,
) -> float:
    """Return the circular-orbit shear speed between two local trap radii."""
    if smbh_mass_msun <= 0.0 or first_radius_cm <= 0.0 or second_radius_cm <= 0.0:
        raise ValueError("SMBH mass and pairing radii must be positive")

    smbh_mass_g = smbh_mass_msun * msun
    first_omega_s_inv = np.sqrt(G * smbh_mass_g / first_radius_cm**3)
    second_omega_s_inv = np.sqrt(G * smbh_mass_g / second_radius_cm**3)
    mean_radius_cm = 0.5 * (first_radius_cm + second_radius_cm)

    return float(mean_radius_cm * abs(first_omega_s_inv - second_omega_s_inv))


def conjunction_rate(
    smbh_mass_msun: float,
    first_radius_cm: float,
    second_radius_cm: float,
) -> float:
    """Return the phase-averaged conjunction rate for two circular orbits."""

    if smbh_mass_msun <= 0.0 or first_radius_cm <= 0.0 or second_radius_cm <= 0.0:
        raise ValueError("SMBH mass and pairing radii must be positive")

    smbh_mass_g = smbh_mass_msun * msun
    first_omega_s_inv = np.sqrt(G * smbh_mass_g / first_radius_cm**3)
    second_omega_s_inv = np.sqrt(G * smbh_mass_g / second_radius_cm**3)

    return float(abs(first_omega_s_inv - second_omega_s_inv) / (2.0 * np.pi))


def is_strong_gravity_encounter(
    m1_msun: float,
    m2_msun: float,
    relative_velocity_cms: float,
    impact_parameter_cm: float,
) -> bool:
    """
    Check whether shear kinetic energy is below the potential at ``b``.
    """
    if m1_msun <= 0.0 or m2_msun <= 0.0:
        raise ValueError("BBH component masses must be positive")
    if relative_velocity_cms < 0.0 or impact_parameter_cm <= 0.0:
        raise ValueError(
            "Relative velocity must be nonnegative and impact parameter positive"
        )

    total_mass_g = (m1_msun + m2_msun) * msun

    return bool(relative_velocity_cms**2 < 2.0 * G * total_mass_g / impact_parameter_cm)


def effective_capture_impact_parameter(
    model_config: ModelConfig,
    total_mass_msun: float,
    trap_radius_cm: float,
    smbh_mass_msun: float,
    trap_width_cm: float,
) -> float:
    """Return the radially admissible single--single capture impact parameter.

    ``pair_up_eta`` scales the mutual Hill radius at the migration trap.  The
    impact parameter is restricted to half the local trap width so a pair is
    never sampled across the entire thermal bath.
    """
    if total_mass_msun <= 0.0 or smbh_mass_msun <= 0.0:
        raise ValueError("The interacting and SMBH masses must be positive")
    if trap_radius_cm <= 0.0 or trap_width_cm <= 0.0:
        raise ValueError("Trap radius and width must be positive")
    if model_config.pair_up_eta <= 0.0:
        raise ValueError("pair_up_eta must be positive")

    mutual_hill_radius_cm = trap_radius_cm * (
        total_mass_msun / (3.0 * smbh_mass_msun)
    ) ** (1.0 / 3.0)
    return float(
        min(model_config.pair_up_eta * mutual_hill_radius_cm, 0.5 * trap_width_cm)
    )


def sbh_cross_section(
    model_config: ModelConfig,
    total_mass_msun: float,
    trap_radius_cm: float,
    smbh_mass_msun: float,
    trap_context: TrapContext,
) -> float:
    """Return a finite-layer sBH--sBH capture cross section in cm^2.

    The raw capture scale is ``pair_up_eta`` times the mutual Hill radius.
    Its radial reach is bounded by half the trap width.  For a uniform thermal
    layer with half-thickness ``H_BH``, the cross section is the area of a
    radius-``b`` circle clipped to ``|z| <= H_BH``.  This approaches
    ``4 b H_BH`` when ``b`` is much larger than the layer thickness instead of
    incorrectly retaining the unbounded ``pi b**2`` area.
    """
    impact_parameter_cm = effective_capture_impact_parameter(
        model_config=model_config,
        total_mass_msun=total_mass_msun,
        trap_radius_cm=trap_radius_cm,
        smbh_mass_msun=smbh_mass_msun,
        trap_width_cm=trap_context.width_cm,
    )
    layer_half_thickness_cm = trap_context.scale_height_cm
    if impact_parameter_cm <= layer_half_thickness_cm:
        return float(np.pi * impact_parameter_cm**2)

    height_cm = layer_half_thickness_cm
    return float(
        2.0
        * (
            height_cm * np.sqrt(impact_parameter_cm**2 - height_cm**2)
            + impact_parameter_cm**2 * np.arcsin(height_cm / impact_parameter_cm)
        )
    )


def bbh_cross_section(model_config: ModelConfig, sma_cm: float) -> float:
    """Return the maximum BSI impact parameter in cm for a BBH of size ``a``."""
    if sma_cm <= 0.0:
        raise ValueError("BBH semimajor axis must be positive")
    return float(model_config.bsi_eta * sma_cm)
