from typing import Literal, overload

import numpy as np

import bsi_trap.constant.constant as const
from bsi_trap.classes.models import AGN_properties, DiskProfile


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


@overload
def migration_threshold(
    Disk: DiskProfile,
    AGN: AGN_properties,
    r: float | None = None,
    alpha: float | None = None,
    return_q: Literal[False] = False,
) -> float: ...


@overload
def migration_threshold(
    Disk: DiskProfile,
    AGN: AGN_properties,
    r: float | None = None,
    alpha: float | None = None,
    return_q: Literal[True] = True,
) -> tuple[float, float, float]: ...


def migration_threshold(
    Disk: DiskProfile,
    AGN: AGN_properties,
    r: float | None = None,
    alpha: float | None = None,
    return_q: bool = False,
) -> float | tuple[float, float, float]:
    if r is None:
        r = Disk.r_trap_cm
    if alpha is None:
        alpha = AGN.alpha

    h = float(Disk.get_h(r))

    # Gap-opening threshold:
    # q_gap ~ sqrt(27*pi/8) * alpha^(1/2) * (H/r)^(5/2)
    q_visc = float(np.sqrt(27.0 * np.pi / 8.0) * np.sqrt(alpha) * h**2.5)

    # Pressure-closing condition:
    # H/r < (1/40) * sqrt(q/alpha)  ->  q > 1600 * alpha * (H/r)^2
    q_press = float(1600.0 * alpha * h**2)

    # Conservative threshold: require both conditions to be satisfied.
    qcrit = float(max(q_visc, q_press))
    mcrit_msun = float(qcrit * AGN.M_SMBH_msun)

    if return_q:
        return mcrit_msun, qcrit, h
    return mcrit_msun


def trap_width_elimate():
    pass
