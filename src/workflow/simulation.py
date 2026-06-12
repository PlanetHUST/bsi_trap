from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from bsi_trap.classes import (
    AGN_properties,
    BBH_properties,
    DiskProfile,
    BBHEvolverParams,
    single_BH_properties,
)

from bsi_trap.config.model_config import ModelConfig
from bsi_trap.main_class_op.embedded_BH_initialize import embedded_BH_initialize
from bsi_trap.main_class_op.class_convert import embedded_to_BBH, BBH_to_ngen
from bsi_trap.disk_profile.migration_torque import find_traps
from bsi_trap.disk_profile.pagn_disk_adapter import build_pagn_disk_profile
from bsi_trap.main_class_op.BH_params_initialize import BBH_params_initialize


@dataclass(frozen=True)
class SampleWorkflowResult:
    output_file: str


def _build_environment(
    model_config: ModelConfig,
    smbh_mass: float,
) -> tuple[AGN_properties, DiskProfile]:

    # 0. Initialize AGN global parameters
    agn = AGN_properties(
        M_SMBH_msun=smbh_mass,
        lifetime_myr=model_config.agn_lifetime_myr,
    )
    agn.alpha = model_config.alpha

    # 1. from pAGN get disk profile
    # QUESTION: model_config details for rin, rout et al. setting
    pagn_profile = build_pagn_disk_profile(
        smbh_mass, alpha=model_config.alpha, n_resolution=model_config.disk_resolution
    )

    # initialize trap radius
    # QUESTION: trap radius is mass dependent
    r_trap_cm, r_antitrap_cm, torque_profile = find_traps(
        pagn_profile.disk, model_config
    )

    # build disk profile
    disk = DiskProfile(
        radius_cm=pagn_profile.r_list,
        h_over_r=pagn_profile.h_list,
        sigma_cgs=pagn_profile.sigma_list,
        cs_cgs=pagn_profile.cs_list,
        r_in_cm=pagn_profile.r_in,
        r_out_cm=pagn_profile.r_out,
        r_trap_cm=r_trap_cm,
        r_antitrap_cm=r_antitrap_cm,
    )

    agn.disk_profile = disk
    agn.torque_profile = torque_profile

    return agn, disk


def initialize_to_trap(
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    rng: np.random.Generator,
) -> Tuple[BBH_properties, single_BH_properties, BBH_properties, single_BH_properties]:

    # 1. initialize embedded BH properties
    ebh = embedded_BH_initialize(agn, disk)

    # 2. embedded BH to BBHs in trap
    bbh_1g, sbh_1g = embedded_to_BBH(model_config, agn, disk, ebh)

    # random parameter filling (a,e,chi)
    BBH_params_initialize(agn, disk, bbh_1g, rng)

    # 3. hierarchical mergers
    bbh_ng, sbh_ng = BBH_to_ngen()

    return bbh_1g, sbh_1g, bbh_ng, sbh_ng


def run_physics_sample(
    model_config: ModelConfig,
    sample_index: int,
    smbh_mass: float,
    seed: int,
    output_file: Path,
    rng: np.random.Generator,
) -> SampleWorkflowResult:
    """single physical task for one AGN-NSC sample"""
    agn, disk = _build_environment(model_config, smbh_mass)

    # 1. initialize bbh properties
    bbh_1g, sbh_1g, bbh_ng, sbh_ng = initialize_to_trap(model_config, agn, disk, rng)

    # 2. storage for the sample

    # 3. DETAIL for generation data get and storage

    return SampleWorkflowResult(output_file)
