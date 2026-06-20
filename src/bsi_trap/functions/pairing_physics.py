"""Pure physical quantities used by initial single-BH pairing dynamics."""

from __future__ import annotations

import numpy as np

from bsi_trap.classes.models import DiskProfile
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.constant.constant import G, c, msun


def pair_up_timescale(disk: DiskProfile, m1_msun: float, m2_msun: float) -> float:
    """Return the gas-driven BBH pair-up delay in seconds.

    This is the Qian--Lai dynamical-friction estimate evaluated at the
    migration trap. Both component masses are in solar masses; disk quantities
    are in CGS units. The result is a delay after a stochastic pair interaction,
    not the interaction waiting time itself.
    """
    if m1_msun <= 0.0 or m2_msun <= 0.0:
        raise ValueError("BBH component masses must be positive")

    r_trap_cm = disk.r_trap_cm
    sound_speed_cgs = disk.get_cs(r_trap_cm)
    scale_height_cm = disk.get_scale_height_cm(r_trap_cm)
    surface_density_cgs = disk.get_sigma(r_trap_cm)
    if sound_speed_cgs <= 0.0 or scale_height_cm <= 0.0 or surface_density_cgs <= 0.0:
        raise ValueError("Trap disk sound speed, scale height, and density must be positive")

    total_mass_g = (m1_msun + m2_msun) * msun
    return float(
        3.0
        * sound_speed_cgs**3
        * scale_height_cm
        * r_trap_cm
        / (4.0 * np.pi * G**2 * surface_density_cgs * total_mass_g)
    )


def sbh_cross_section(model_config: ModelConfig, total_mass_msun: float) -> float:
    """Return the effective sBH--sBH pairing cross section in cm^2.

    ``pair_up_eta`` multiplies the gravitational radius of the two-body system
    to define the encounter radius. The geometric cross section is then
    ``pi * (pair_up_eta * r_g)^2``.
    """
    if total_mass_msun <= 0.0:
        raise ValueError("The interacting sBH total mass must be positive")

    gravitational_radius_cm = G * total_mass_msun * msun / c**2
    encounter_radius_cm = model_config.pair_up_eta * gravitational_radius_cm
    return float(np.pi * encounter_radius_cm**2)


def bbh_cross_section(model_config: ModelConfig, sma_cm: float) -> float:
    """Return the maximum BSI impact parameter in cm for a BBH of size ``a``."""
    if sma_cm <= 0.0:
        raise ValueError("BBH semimajor axis must be positive")
    return float(model_config.bsi_eta * sma_cm)
