import numpy as np
from numpy.random import Generator

from bsi_trap.config.model_config import ModelConfig
from bsi_trap.classes.classes import AGN_properties, DiskProfile, BBH_properties


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
    AGN: AGN_properties, disk: DiskProfile, bbh: BBH_properties
) -> np.ndarray:
    """Sample inner binary separations using the legacy a^(9/2) law."""
    pass


def sample_spin_values(rng: Generator, rms=0.1) -> float:
    """Sample a dimensionless spin magnitude truncated at one."""
    while True:
        value = float(np.linalg.norm(rng.normal(0.0, rms, size=3)))
        if value <= 1.0:
            return value


def sample_spin_tilts(
    BBH: BBH_properties,
    model_config: ModelConfig,
    rng: Generator,
) -> None:
    """Assign spin-tilt angles using the gas/no-gas convention."""

    num = BBH.num

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

    BBH.th1 = np.arccos(cos_theta_1)
    BBH.th2 = np.arccos(cos_theta_2)


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


# def semimajor_out_random_linear(N, AGN, min_index, max_index):
#     rg = AGN.rg
#     rmin = 10**min_index * rg
#     rmax = 10**max_index * rg
#     u = np.random.random(N)
#     sma = u * (rmax - rmin) + rmin  # Linear interpolation
#     return sma


# # Alternative (uniform logarithmic):
# def semimajor_out_random_log(N, AGN, min_index, max_index):
#     rg = AGN.rg
#     u = np.random.random(N)
#     log_sma = u * (max_index - min_index) + min_index  # Linear in log space
#     sma = 10**log_sma * rg  # Convert back to linear
#     return sma


# def fBH_sample():
#     """sample the BH fraction in the nsc with Gaussion distribution"""
#     val = np.random.normal(loc=0.04, scale=0.01)
#     return float(np.clip(val, 0.0, 1.0))
