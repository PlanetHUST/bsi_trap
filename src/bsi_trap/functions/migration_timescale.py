import numpy as np
from bsi_trap.constant.constant import G, msun
from bsi_trap.classes.classes import AGN_properties, DiskProfile
from bsi_trap.disk_profile.migration_torque import compute_torque_profile


def migration_timescale_torque(
    m_msun: np.ndarray,  # shape (N,)
    R_ini_cm: np.ndarray,  # shape (N,)
    AGN: AGN_properties,
    Disk: DiskProfile,
) -> np.ndarray:
    """
    Fully vectorized migration timescale computation.
    Eliminates Python loops using masked 2D trapezoidal integration.
    """
    torque_profile = compute_torque_profile(Disk)
    gamma_total = torque_profile.gamma_tot  # shape (M,)
    radius_cm = torque_profile.radius_cm  # shape (M,)
    Omega = np.sqrt(G * AGN.M_SMBH * msun / radius_cm**3)  # shape (M,)

    m_msun = np.atleast_1d(m_msun)[:, None]  # shape (N,1)
    R_ini_cm = np.atleast_1d(R_ini_cm)[:, None]  # shape (N,1)

    # Compute dr/dt for all masses x radii
    dr_dt = (
        2.0
        * gamma_total[None, :]
        / (m_msun * msun * radius_cm[None, :] * Omega[None, :])
    )  # shape (N,M)

    # Determine target radius for each object
    r_trap = Disk.r_trap_cm
    r_antitrap = Disk.r_antitrap_cm
    target_radius = np.where(
        R_ini_cm > r_trap, r_trap, np.where(R_ini_cm > r_antitrap, r_antitrap, np.nan)
    )  # shape (N,1)

    # Mask invalid cases
    invalid_mask = np.isnan(target_radius).flatten()

    # Compute start/stop bounds for each object
    start = np.minimum(R_ini_cm, target_radius)  # shape (N,1)
    stop = np.maximum(R_ini_cm, target_radius)  # shape (N,1)

    # Create mask of valid radii for each object
    radius_mask = (
        (radius_cm[None, :] >= start)
        & (radius_cm[None, :] <= stop)
        & np.isfinite(dr_dt)
    )  # shape (N,M)

    # Compute dx for trapezoid: differences along radius
    dx = np.diff(radius_cm)  # shape (M-1,)

    # Compute 1/dr_dt midpoints for trapezoid
    dr_dt_mid = 0.5 * (1.0 / dr_dt[:, :-1] + 1.0 / dr_dt[:, 1:])  # shape (N, M-1)

    # Mask the midpoints: valid if both endpoints are valid
    mask_mid = radius_mask[:, :-1] & radius_mask[:, 1:]  # shape (N, M-1)

    # Apply trapezoidal integration
    integral = np.sum(dr_dt_mid * dx[None, :] * mask_mid, axis=1)  # shape (N,)

    # Assign np.inf to invalid cases
    integral[invalid_mask] = np.inf
    integral[np.sum(mask_mid, axis=1) < 1] = np.inf  # ensure minimal valid points

    return integral
