import numpy as np


def sample_smbh_mass(size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample SMBH masses in Msun."""
    log10_mass = rng.normal(loc=6.576, scale=0.591, size=size)
    return np.clip(10**log10_mass, None, 5e7)


def sample_duty_cycle_myr(rng: np.random.Generator) -> float:
    """Sample AGN duty cycle in Myr."""
    value = rng.normal(loc=0.22, scale=0.80)
    return float(np.clip(10**value, 0.0, None))


def ecc(values) -> np.ndarray:
    """Sample the thermal eccentricity distribution P(e) = 2e."""
    return np.sqrt(np.random.random(len(values)))


def semimajor(m1, m2, AGN) -> np.ndarray:
    """Sample inner binary separations using the legacy a^(9/2) law."""
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    a_min = 6.957e10
    a_max = AGN.rtrap * (
        (np.max(m1) + np.max(m2)) / float(AGN.M_SMBH)
    ) ** (1.0 / 3.0)
    x = np.random.random(len(m1))
    return (
        (a_max ** (11.0 / 2.0) - a_min ** (11.0 / 2.0)) * x
        + a_min ** (11.0 / 2.0)
    ) ** (2.0 / 11.0)


def secondary(m1, m2min) -> float:
    """Sample a secondary mass using the legacy O'Leary distribution."""
    normalization = (2.0 * m1) ** 5.0 - (m1 + m2min) ** 5.0
    if normalization <= 0.0:
        return float(m2min)
    return float(
        (
            np.random.random() * normalization
            + (m1 + m2min) ** 5.0
        ) ** (1.0 / 5.0)
        - m1
    )


def spin(rms=0.1) -> float:
    """Sample a dimensionless spin magnitude truncated at one."""
    while True:
        value = float(np.linalg.norm(np.random.normal(0.0, rms, size=3)))
        if value <= 1.0:
            return value


################### NSC stars' SMA Sampleing ###############
def semimajor_out(N, AGN, min_index, max_index, gamma=7 / 4):
    gamma = -gamma + 2  # change from number density to power index in volume intergral

    rg = AGN.rg  # in cm
    rmin = 10**min_index * rg
    rmax = 10**max_index * rg
    gamma1 = gamma + 1.0  # for reverse method

    rmin_pow = rmin**gamma1
    rmax_pow = rmax**gamma1

    u = np.random.random(N)
    sma = (rmin_pow + u * (rmax_pow - rmin_pow)) ** (1 / gamma1)
    return sma


def semimajor_out_random_linear(N, AGN, min_index, max_index):
    rg = AGN.rg
    rmin = 10**min_index * rg
    rmax = 10**max_index * rg
    u = np.random.random(N)
    sma = u * (rmax - rmin) + rmin  # Linear interpolation
    return sma


# Alternative (uniform logarithmic):
def semimajor_out_random_log(N, AGN, min_index, max_index):
    rg = AGN.rg
    u = np.random.random(N)
    log_sma = u * (max_index - min_index) + min_index  # Linear in log space
    sma = 10**log_sma * rg  # Convert back to linear
    return sma


def fBH_sample():
    """sample the BH fraction in the nsc with Gaussion distribution"""
    val = np.random.normal(loc=0.04, scale=0.01)
    return float(np.clip(val, 0.0, 1.0))
