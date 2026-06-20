import numpy as np
from numpy.random import Generator
from typing import Tuple

from bsi_trap.config.model_config import ModelConfig
from bsi_trap.classes.models import AGN_properties, DiskProfile, BBH_properties
from bsi_trap.constant.constant import Rsun


def sample_smbh_mass(size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample SMBH masses in Msun."""
    log10_mass = rng.normal(loc=6.576, scale=0.591, size=size)
    return np.clip(10**log10_mass, None, 5e7)


def sample_duty_cycle_myr(rng: np.random.Generator) -> float:
    """Sample AGN duty cycle in Myr."""
    value = rng.normal(loc=0.22, scale=0.80)
    return float(np.clip(10**value, 0.0, None))


def sample_bbh_ecc(num: int, rng: Generator) -> np.ndarray:
    """Sample the thermal eccentricity distribution P(e) = 2e."""
    return np.sqrt(rng.random(num))


def sample_bbh_semimajor(
    AGN: AGN_properties,
    disk: DiskProfile,
    bbh: BBH_properties,
    rng: Generator,
) -> np.ndarray:
    """Sample BBH inner semimajor axes in cm with p(a) proportional to a^(9/2)."""
    if bbh.num == 0:
        return np.empty(0, dtype=float)

    total_mass_msun = bbh.m1_msun + bbh.m2_msun
    if np.any(total_mass_msun <= 0.0):
        raise ValueError("BBH component masses must be positive")

    a_min_cm = Rsun
    max_total_mass_msun = float(np.max(total_mass_msun))
    a_max_cm = disk.r_trap_cm * (max_total_mass_msun / AGN.M_SMBH_msun) ** (1.0 / 3.0)
    if not np.isfinite(a_max_cm) or a_max_cm <= a_min_cm:
        raise ValueError(
            "The trap Hill radius must exceed one solar radius for BBH sampling"
        )

    power = 11.0 / 2.0
    u = rng.random(bbh.num)
    return (a_min_cm**power + u * (a_max_cm**power - a_min_cm**power)) ** (1.0 / power)


def sample_spin_values(num: int, rng: Generator, rms=0.1) -> np.ndarray:
    """Sample a dimensionless spin magnitude truncated at one."""
    spin_values = np.linalg.norm(rng.normal(0.0, rms, size=(num, 3)), axis=1)
    invalid = spin_values > 1.0
    while np.any(invalid):
        spin_values[invalid] = np.linalg.norm(
            rng.normal(0.0, rms, size=(int(np.sum(invalid)), 3)),
            axis=1,
        )
        invalid = spin_values > 1.0
    return spin_values


def sample_spin_tilts(
    num: int,
    model_config: ModelConfig,
    rng: Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign spin-tilt angles using the gas/no-gas convention."""

    if model_config.gas_mode == "with_gas":
        cos_theta_1 = np.clip(
            rng.normal(loc=1.0, scale=0.1, size=num),
            -1.0,
            1.0,
        )
        cos_theta_2 = np.clip(
            rng.normal(loc=1.0, scale=0.1, size=num),
            -1.0,
            1.0,
        )

    elif model_config.gas_mode == "without_gas":
        cos_theta_1 = rng.uniform(-1.0, 1.0, size=num)
        cos_theta_2 = rng.uniform(-1.0, 1.0, size=num)

    else:
        raise ValueError(f"Unknown gas_mode: {model_config.gas_mode}")

    return np.arccos(cos_theta_1), np.arccos(cos_theta_2)


################### NSC stars' SMA Sampleing ###############
def sample_embedded_radius(
    N: int, AGN: AGN_properties, disk: DiskProfile, rng: Generator, gamma=7 / 4
):
    gamma = -gamma + 2  # change from number density to power index in volume intergral

    rmin_cm = disk.r_in_cm
    rmax_cm = disk.r_out_cm
    gamma1 = gamma + 1.0  # for reverse method

    rmin_pow = rmin_cm**gamma1
    rmax_pow = rmax_cm**gamma1

    u = rng.random(N)
    sma = (rmin_pow + u * (rmax_pow - rmin_pow)) ** (1 / gamma1)
    return sma
