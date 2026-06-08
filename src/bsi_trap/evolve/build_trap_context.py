'''placeholder for build_trap_context.py'''
import numpy as np

import bsi_trap.constant.constant as const
from bsi_trap.classes.classes import (
    AGN_properties,
    BBH_properties,
    DiskProfile,
    single_BH_properties,
)

def build_trap_context(BBH: BBH_properties, sinBHs: single_BH_properties, Disk: DiskProfile, AGN: AGN_properties) -> dict:
    """Construct a trap-local context dictionary for BSI calculations."""
    trap_context = {
        "number_density": Disk.sigma_trap / (Disk.h_trap * const.msun),  # in cm^-3
        "v_rel": np.sqrt(const.G * AGN.M_SMBH_msun * const.msun / AGN.r_trap_cm),  # in cm/s
        "perturber_mass": sinBHs.m_msun*const.msun,  # in g
    }
    return trap_context
