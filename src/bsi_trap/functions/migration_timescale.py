import numpy as np

from bsi_trap.constant.constant import G, msun
from bsi_trap.classes.classes import AGN_properties, DiskProfile
from bsi_trap.disk_profile.migration_torque import compute_torque_profile


def migration_timescale_torque(
    m_msun: list,
    AGN: AGN_properties,
    Disk: DiskProfile,
    R_ini_cm: list,
) -> float:
    """Migration timescale estimate based on the torque profile, returned in seconds."""
    torque_profile = compute_torque_profile(Disk)
    gamma_total = torque_profile.gamma_tot
    radius_cm = torque_profile.radius_cm
    Omega = np.sqrt(G * AGN.M_SMBH * msun / radius_cm**3.0)
    # dr/dt = 2 * Gamma / (m * r * Omega)
    dr_dt = 2.0 * gamma_total / (m_msun * msun * radius_cm * Omega)
    r_trap_cm = Disk.r_trap_cm
    r_anti_trap_cm = Disk.r_antitrap_cm
    R_ini_cm = float(R_ini_cm)
    target_radius = r_trap_cm
    if R_ini_cm > r_trap_cm:
        target_radius = r_trap_cm
    elif R_ini_cm > r_anti_trap_cm:
        target_radius = r_anti_trap_cm
    else:
        return np.inf

    start, stop = sorted((R_ini_cm, target_radius))
    mask = (radius_cm >= start) & (radius_cm <= stop) & np.isfinite(dr_dt)
    if np.count_nonzero(mask) < 2:
        return np.inf
    return float(abs(np.trapezoid(1.0 / dr_dt[mask], x=radius_cm[mask])))
