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


def compute_eccentricity_at_frequency(
    m1,
    m2,
    a,
    e,
    target_freq,
    *,
    previous_a=None,
    previous_e=None,
):
    """Interpolate eccentricity when a step brackets ``target_freq``."""
    freq = frequency_from_sma(m1, m2, a)
    if not (
        np.isfinite(freq)
        and np.isfinite(target_freq)
        and target_freq > 0.0
        and np.isfinite(e)
    ):
        return np.nan

    if previous_a is not None and previous_e is not None:
        previous_freq = frequency_from_sma(m1, m2, previous_a)
        if (
            np.isfinite(previous_freq)
            and np.isfinite(previous_e)
            and previous_freq > 0.0
            and min(previous_freq, freq) <= target_freq <= max(previous_freq, freq)
        ):
            if np.isclose(previous_freq, freq):
                return float(e)
            weight = (
                np.log(target_freq) - np.log(previous_freq)
            ) / (np.log(freq) - np.log(previous_freq))
            weight = float(np.clip(weight, 0.0, 1.0))
            return float(previous_e + weight * (e - previous_e))

    if np.isclose(freq, target_freq):
        return float(e)
    return np.nan


@overload
def migration_threshold(
    Disk: DiskProfile,
    AGN: AGN_properties,
    r: float | None = None,
    alpha: float | None = None,
) -> float: ...


@overload
def migration_threshold(
    Disk: DiskProfile,
    AGN: AGN_properties,
    r: float | None = None,
    alpha: float | None = None,
    *,
    return_q: Literal[True],
) -> tuple[float, float, float]: ...


@overload
def migration_threshold(
    Disk: DiskProfile,
    AGN: AGN_properties,
    r: float | None = None,
    alpha: float | None = None,
    *,
    return_q: Literal[False],
) -> float: ...


@overload
def migration_threshold(
    Disk: DiskProfile,
    AGN: AGN_properties,
    r: float | None,
    alpha: float | None,
    return_q: Literal[False],
) -> float: ...


@overload
def migration_threshold(
    Disk: DiskProfile,
    AGN: AGN_properties,
    r: float | None,
    alpha: float | None,
    return_q: Literal[True],
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
