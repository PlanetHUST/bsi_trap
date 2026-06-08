import numpy as np

from bsi_trap.constant.constant import G, msun
from bsi_trap.classes import AGN_properties
from bsi_trap.disk_profile.migration_torque import compute_torque_profile


def migration_timscale_torque(
    disk,
    AGN: AGN_properties,
    R_ini_cm: list,
) -> float:
    """Migration timescale estimate based on the torque profile, returned in seconds."""
    raw_disk = getattr(disk, "disk", disk)
    torque_profile = compute_torque_profile(raw_disk)
    gamma_total = torque_profile.gamma_tot
    radius_cm = torque_profile.radius_cm
    Omega = np.sqrt(G * AGN.M_SMBH * msun / radius_cm**3.0)
    # dr/dt = 2 * Gamma / (m * r * Omega)
    dr_dt = 2.0 * gamma_total / (AGN.migrator_mass * radius_cm * Omega)
    r_trap_cm = AGN.rtrap
    r_anti_trap_cm = getattr(AGN, "rantitrap", r_trap_cm)
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
    return float(abs(np.trapz(1.0 / dr_dt[mask], x=radius_cm[mask])))
