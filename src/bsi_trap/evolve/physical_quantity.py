import numpy as np

import bsi_trap.constant.constant as const
from bsi_trap.classes.classes import AGN_properties, DiskProfile


def frequency_from_sma(m1, m2, a):
    """Compute the gravitational wave frequency (Hz) from the semi-major axis."""
    return 1.0 / np.pi * np.sqrt(const.G * (m1 + m2) / a**3)


def threshold_radius(m1, m2):
    """Compute the threshold radius for merger (6 GM/c^2)."""
    return 6 * const.G * (m1 + m2) / const.c**2


def compute_merger_time_estimate(m1, m2, a, e):
    """Approximate merger time (Peters 1964)"""
    return (
        5
        / 256
        * const.c**5
        / const.G**3
        * a**4
        * (1 - e**2) ** 3.5
        / (m1 * m2 * (m1 + m2))
    )


def compute_eccentricity_at_frequency(m1, m2, a, e, target_freq):
    """Compute eccentricity when GW frequency reaches target_freq Hz"""
    freq = 1.0 / np.pi * np.sqrt(const.G * (m1 + m2) / a**3)
    if abs(freq - target_freq) < abs(freq) * 0.1:
        # print(f'ecc10 set as {e}')
        return e
    return float(e) if np.isfinite(e) else np.nan


def migration_threshold(
    Disk: DiskProfile,
    AGN: AGN_properties,
    r=None,
    alpha=None,
    return_q=False,
):
    if r is None:
        r = AGN.rtrap
    if alpha is None:
        alpha = AGN.alpha

    h = Disk.get_h(r)

    # Gap-opening threshold:
    # q_gap ~ sqrt(27*pi/8) * alpha^(1/2) * (H/r)^(5/2)
    q_visc = np.sqrt(27.0 * np.pi / 8.0) * np.sqrt(alpha) * h**2.5

    # Pressure-closing condition:
    # H/r < (1/40) * sqrt(q/alpha)  ->  q > 1600 * alpha * (H/r)^2
    q_press = 1600.0 * alpha * h**2

    # Conservative threshold: require both conditions to be satisfied.
    qcrit = max(q_visc, q_press)
    mcrit_msun = qcrit * AGN.M_SMBH

    if return_q:
        return mcrit_msun, qcrit, h
    return mcrit_msun


def trap_width_elimate():
    pass
