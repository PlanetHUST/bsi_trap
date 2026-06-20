import numpy as np
from bsi_trap.constant.constant import G, msun
from bsi_trap.classes.models import AGN_properties, DiskProfile


def migration_timescale_torque(
    m_msun: np.ndarray,  # shape (N,)
    R_ini_cm: np.ndarray,  # shape (N,)
    AGN: AGN_properties,
    Disk: DiskProfile,
) -> np.ndarray:
    """Calculate migration times without allocating a population-by-radius grid."""
    torque_profile = AGN.torque_profile
    if torque_profile is None:
        raise ValueError("AGN.torque_profile must be initialized before migration")

    # pAGN torque is in SI; convert N m to erg for the CGS migration equation.
    gamma_total = np.asarray(torque_profile.gamma_tot, dtype=float) * 1.0e7
    radius_cm = np.asarray(torque_profile.radius_cm, dtype=float)
    omega = np.sqrt(G * AGN.M_SMBH_msun * msun / radius_cm**3)
    m_msun = np.atleast_1d(np.asarray(m_msun, dtype=float))
    R_ini_cm = np.atleast_1d(np.asarray(R_ini_cm, dtype=float))
    if m_msun.shape != R_ini_cm.shape:
        raise ValueError("m_msun and R_ini_cm must have the same shape")

    r_trap = Disk.r_trap_cm
    r_antitrap = Disk.r_antitrap_cm
    target_radius = np.where(
        R_ini_cm > r_trap, r_trap, np.where(R_ini_cm > r_antitrap, r_antitrap, np.nan)
    )
    start = np.minimum(R_ini_cm, target_radius)
    stop = np.maximum(R_ini_cm, target_radius)

    # Migration time is linear in BH mass. Integrate the one-Msun inverse
    # migration speed once, then query the cumulative integral for every BH.
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse_speed = np.abs(
            msun * radius_cm * omega / (2.0 * gamma_total)
        )
    interval_valid = np.isfinite(gamma_total[:-1]) & np.isfinite(
        gamma_total[1:]
    )
    interval_singular = interval_valid & (
        ~np.isfinite(inverse_speed[:-1]) | ~np.isfinite(inverse_speed[1:])
    )
    interval_integral = (
        0.5
        * (inverse_speed[:-1] + inverse_speed[1:])
        * np.diff(radius_cm)
    )
    interval_integral = np.where(
        interval_valid & ~interval_singular,
        interval_integral,
        0.0,
    )

    cumulative_time = np.concatenate(([0.0], np.cumsum(interval_integral)))
    cumulative_valid = np.concatenate(
        ([0], np.cumsum(interval_valid, dtype=np.int64))
    )
    cumulative_singular = np.concatenate(
        ([0], np.cumsum(interval_singular, dtype=np.int64))
    )

    left = np.searchsorted(radius_cm, start, side="left")
    right = np.searchsorted(radius_cm, stop, side="right") - 1
    valid_bounds = (
        np.isfinite(target_radius)
        & (left >= 0)
        & (right < radius_cm.size)
        & (right > left)
    )

    result = np.full(m_msun.shape, np.inf, dtype=float)
    valid_idx = np.flatnonzero(valid_bounds)
    if valid_idx.size:
        valid_intervals = (
            cumulative_valid[right[valid_idx]] - cumulative_valid[left[valid_idx]]
        )
        usable = valid_idx[valid_intervals > 0]
        singular_intervals = (
            cumulative_singular[right[usable]]
            - cumulative_singular[left[usable]]
        )
        finite = usable[singular_intervals == 0]
        result[finite] = np.abs(m_msun[finite]) * (
            cumulative_time[right[finite]] - cumulative_time[left[finite]]
        )

    return result
