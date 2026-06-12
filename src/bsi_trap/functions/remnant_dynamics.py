import numpy as np

import bsi_trap.constant.constant as const
from bsi_trap.functions import merger_properties as merg
from bsi_trap.functions.nsc_population import mass_R_NSC

# def apply_recoil_and_orbit_update(BBH, AGN, flags, i, rmax):
#     """Apply GW recoil and update the remnant's post-merger outer orbit."""
#     BBH.vk[i] = merg.relativistic_kick(
#         BBH.m1[i],
#         BBH.m2[i],
#         BBH.chi1[i],
#         BBH.chi2[i],
#         BBH.th1[i],
#         BBH.th2[i],
#     )

#     enclosed_mass = AGN.M_SMBH + mass_R_NSC(AGN.M_SMBH, AGN.rtrap)
#     v_kep = np.sqrt(const.G * enclosed_mass * const.msun / AGN.rtrap)

#     if BBH.vk[i] + v_kep <= AGN.vesc:
#         flags.GWescape_flag[i] = 0
#         BBH.A[i] = merg.final_semimajor(AGN, BBH.vk[i] + v_kep)
#         if BBH.A[i] > rmax:
#             flags.GWescape_flag[i] = 1
#     else:
#         flags.GWescape_flag[i] = 1


def apply_recoil_and_orbit_update(
    BBH, agn, disk, flags, i: int, rmax_cm: float
) -> None:
    r_trap_cm = disk.r_trap_cm

    BBH.vk_cgs[i] = merg.relativistic_kick(
        BBH.m1_msun[i],
        BBH.m2_msun[i],
        BBH.chi1[i],
        BBH.chi2[i],
        BBH.th1[i],
        BBH.th2[i],
    )

    enclosed_mass_msun = agn.M_SMBH_msun + mass_R_NSC(
        agn.M_SMBH_msun,
        r_trap_cm,
    )

    v_kep = np.sqrt(const.G * enclosed_mass_msun * const.msun / r_trap_cm)

    vesc = np.sqrt(2.0 * const.G * enclosed_mass_msun * const.msun / r_trap_cm)

    if BBH.vk_cgs[i] + v_kep <= vesc:
        flags.GWescape_flag[i] = 0
        BBH.A_cm[i] = merg.final_semimajor(agn, disk, BBH.vk_cgs[i] + v_kep)
        if BBH.A_cm[i] > rmax_cm:
            flags.GWescape_flag[i] = 1
    else:
        flags.GWescape_flag[i] = 1
