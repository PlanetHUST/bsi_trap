# agn/nsc.py

import numpy as np

import bsi_trap.constant.constant as const
from bsi_trap.classes.classes import AGN_properties, stellar_properties

gamma =1.3

def t_relax(M_SMBH, R):
    mave_msun = 1
    mave_g = mave_msun * const.msun
    number_R = number_R_NSC(M_SMBH, mave_msun, R)
    logGamma = 10
    trelax_s = (
        0.34 * velocity_disp(M_SMBH) ** 3 / (const.G**2 * mave_g**2 * number_R * logGamma)
    )
    return trelax_s


def velocity_dispersion(M_smbh_msun: float) -> float:
    """Return NSC velocity dispersion in cm/s."""
    if M_smbh_msun <= 0:
        raise ValueError(f"M_smbh_msun must be positive, got {M_smbh_msun}")

    return 2.3e5 * M_smbh_msun ** (1 / 4.38) # in cgs


def nsc_mass(M_smbh_msun: float) -> float:
    """Return NSC mass in Msun."""
    if M_smbh_msun <= 0:
        raise ValueError(f"M_smbh_msun must be positive, got {M_smbh_msun}")

    if M_smbh_msun > 5e7:
        return 1e-1

    value = M_smbh_msun * ((5e7 / M_smbh_msun) ** (2 / 3) - 1) # Mnsc in msun

    return float(np.clip(value, 1e-1, None))


def influential_radius(M_SMBH_msun: float)-> float:
    # influential raidus of SMBH in cm
    return const.G * M_SMBH_msun * const.msun * velocity_disp(M_SMBH_msun) ** (-2) 

def mass_R_NSC(M_SMBH_msun: float, R_cm:float, gamma: float=1.3)->float:

    rm_cm = influential_radius(M_SMBH_msun) 

    m_enclosed_msun = nsc_mass(M_SMBH_msun) * (R_cm / rm_cm) ** (3 - gamma)  # effectibe enclosed NSC masses in msun

    return m_enclosed_msun


def number_R_NSC(M_SMBH_msun: float, m_msun: float, R_cm: float, gamma: float=1.3):
    # number enclosed in R_cm
    m_g = m_msun*const.msun
    M_SMBH_g = M_SMBH_msun * const.msun
    rm_cm = influential_radius(M_SMBH_msun)
    number = ((3 - gamma) * M_SMBH_g) / (2 * np.pi * m_g) * rm_cm ** (-3) * (R_cm / rm_cm) ** (-gamma)
    return number


def velocity_disp(M_SMBH_msun: float)-> float:
    vel_cgs = 2.3e5 * (M_SMBH_msun) ** (1 / 4.38)  # in cm/s
    return vel_cgs


def vesc(AGN:AGN_properties):
    Msmbh_msun = AGN.M_SMBH_msun

    M_tot_msun = Msmbh_msun + mass_R_NSC(Msmbh_msun, AGN.rtrap)
    vesc = np.sqrt(2 * const.G * (M_tot_msun * const.msun) / (AGN.rtrap))
    return vesc, M_tot_msun
