import numpy as np

import bsi_trap.constant.constant as const
from bsi_trap.constant.stop_reason import StopReasons
from bsi_trap.classes.classes import BBH_properties, DiskProfile, AGN_properties, merger_flags
from bsi_trap.evolve.physical_quantity import migration_threshold

def stop_for_remanent_mass(BBH: BBH_properties, flags: merger_flags, Disk: DiskProfile, AGN: AGN_properties):
    N=len(BBH.m1_msun)
    
    if BBH.mrem_msun is not None and N > 0:
        typeii_threshold = migration_threshold(Disk, AGN, AGN.r_trap_cm)
        flags.stop_reason_flag[BBH.mrem_msun*const.msun>typeii_threshold]=StopReasons.STOP_REMNANT_TRAP_OVERLOAD
        

def apply_hierarchical_boundaries(BBH: BBH_properties, flags: merger_flags, Disk: DiskProfile, AGN: AGN_properties):
    pass


def process_merged_binary():
    pass
