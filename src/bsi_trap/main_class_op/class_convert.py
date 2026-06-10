from typing import Tuple
import numpy as np

from bsi_trap.classes.classes import (
    Embedded_BH,
    BBH_properties,
    single_BH_properties,
    DiskProfile,
    AGN_properties,
    bbh_flags,
)
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.functions.migration_timescale import migration_timescale_torque
from bsi_trap.functions.pair_up import construct_trap_population
from bsi_trap.constant.constant import Myr


def embedded_to_BBH(
    model_config: ModelConfig, agn: AGN_properties, disk: DiskProfile, ebh: Embedded_BH
) -> Tuple[BBH_properties, single_BH_properties]:

    # 1. calculate migration timescales
    t_mig_s = migration_timescale_torque(ebh.m_msun, agn, disk, ebh.R_cm)

    # 2. mask for trap reaching
    t_reach_s = t_mig_s + ebh.t_form_s

    mask_reach = t_reach_s < agn.lifetime_myr * Myr

    ebh.trap_flag[~mask_reach] = -1

    construct_trap_population(ebh, mask_reach, seed=model_config.pair_up_seed)

    bbh_1g = init_bbh(ebh)
    sbh_1g = init_sbh(ebh)

    return bbh_1g, sbh_1g


def init_bbh(ebh: Embedded_BH) -> BBH_properties:
    """
    Convert paired Embedded_BH objects into BBH_properties.

    ebh.trap_flag:
        -1 : did not reach trap
        -2 : single BH in trap
        j >= 0 : paired with BH index j
    """

    pair_i = []
    pair_j = []

    for i, j in enumerate(ebh.trap_flag):
        if j >= 0 and i < j:
            pair_i.append(i)
            pair_j.append(j)

    pair_i = np.asarray(pair_i, dtype=int)
    pair_j = np.asarray(pair_j, dtype=int)

    num_bbh = len(pair_i)

    bbh = BBH_properties(num_bbh)
    bbh.bbh_flag = bbh_flags(num_bbh)

    m_i = ebh.m_msun[pair_i]
    m_j = ebh.m_msun[pair_j]

    primary_is_i = m_i >= m_j

    idx_primary = np.where(primary_is_i, pair_i, pair_j)
    idx_secondary = np.where(primary_is_i, pair_j, pair_i)

    bbh.m1_msun = ebh.m_msun[idx_primary]
    bbh.m2_msun = ebh.m_msun[idx_secondary]

    bbh.t_ini_s = np.maximum(
        ebh.t_ini_s[pair_i],
        ebh.t_ini_s[pair_j],
    )

    return bbh


def init_sbh(ebh: Embedded_BH) -> single_BH_properties:
    """
    Convert unpaired trap BHs into single_BH_properties.
    """

    sbh_idx = np.where(ebh.trap_flag == -2)[0]
    num_sbh = len(sbh_idx)

    sbh = single_BH_properties(num_sbh)

    sbh.m_msun = ebh.m_msun[sbh_idx]
    sbh.t_ini_s = ebh.t_ini_s[sbh_idx]

    return sbh


def BBH_to_ngen():
    pass
