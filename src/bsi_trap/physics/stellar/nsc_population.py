# agn/nsc.py

import numpy as np

import bsi_trap.constant.constant as const
from bsi_trap.simulation.state.models import AGN_properties, DiskProfile

gamma = 1.3


def t_relax(M_SMBH_msun: float, R_cm: float) -> float:
    mave_msun = 1.0
    mave_g = mave_msun * const.msun
    n_cm3 = number_density_NSC(M_SMBH_msun, mave_msun, R_cm)
    logGamma = 10.0

    if n_cm3 <= 0.0:
        raise ValueError(f"NSC number density must be positive, got {n_cm3}")

    return (
        0.34
        * velocity_dispersion(M_SMBH_msun) ** 3
        / (const.G**2 * mave_g**2 * n_cm3 * logGamma)
    )


def velocity_dispersion(M_smbh_msun: float) -> float:
    """Return NSC velocity dispersion in cm/s."""
    if M_smbh_msun <= 0:
        raise ValueError(f"M_smbh_msun must be positive, got {M_smbh_msun}")

    return 2.3e5 * M_smbh_msun ** (1 / 4.38)  # in cgs


def nsc_mass(M_smbh_msun: float) -> float:
    """Return NSC mass in Msun."""
    if M_smbh_msun <= 0:
        raise ValueError(f"M_smbh_msun must be positive, got {M_smbh_msun}")

    if M_smbh_msun > 5e7:
        return 1e-1

    value = M_smbh_msun * ((5e7 / M_smbh_msun) ** (2 / 3) - 1)  # Mnsc in msun

    return float(np.clip(value, 1e-1, None))


def influential_radius(M_SMBH_msun: float) -> float:
    # influential raidus of SMBH in cm
    return const.G * M_SMBH_msun * const.msun * velocity_disp(M_SMBH_msun) ** (-2)


def mass_enclosed_NSC(M_SMBH_msun: float, R_cm: float, gamma: float = 1.3) -> float:

    rm_cm = influential_radius(M_SMBH_msun)

    m_enclosed_msun = nsc_mass(M_SMBH_msun) * (R_cm / rm_cm) ** (
        3 - gamma
    )  # effectibe enclosed NSC masses in msun

    return m_enclosed_msun


def BH_fraction_NSC():
    return 1e-2


def number_enclosed_NSC(
    M_SMBH_msun: float,
    mave_msun: float,
    R_cm: float,
    gamma: float = 1.3,
) -> float:
    """Expected enclosed number of NSC objects inside R_cm."""
    return mass_enclosed_NSC(M_SMBH_msun, R_cm, gamma) / mave_msun


def number_density_NSC(
    M_SMBH_msun: float,
    mave_msun: float,
    R_cm: float,
    gamma: float = 1.3,
) -> float:
    """Return local NSC number density n(R) in cm^-3."""
    if M_SMBH_msun <= 0:
        raise ValueError(f"M_SMBH_msun must be positive, got {M_SMBH_msun}")
    if mave_msun <= 0:
        raise ValueError(f"mave_msun must be positive, got {mave_msun}")
    if R_cm <= 0:
        raise ValueError(f"R_cm must be positive, got {R_cm}")

    rm_cm = influential_radius(M_SMBH_msun)

    rho_msun_cm3 = (
        (3.0 - gamma)
        * nsc_mass(M_SMBH_msun)
        / (4.0 * np.pi * rm_cm**3)
        * (R_cm / rm_cm) ** (-gamma)
    )

    return rho_msun_cm3 / mave_msun


def velocity_disp(M_SMBH_msun: float) -> float:
    vel_cgs = 2.3e5 * (M_SMBH_msun) ** (1 / 4.38)  # in cm/s
    return vel_cgs


def vesc(AGN: AGN_properties, disk: DiskProfile):
    Msmbh_msun = AGN.M_SMBH_msun

    M_tot_msun = Msmbh_msun + mass_enclosed_NSC(Msmbh_msun, disk.r_trap_cm)
    vesc = np.sqrt(2 * const.G * (M_tot_msun * const.msun) / disk.r_trap_cm)
    return vesc, M_tot_msun
