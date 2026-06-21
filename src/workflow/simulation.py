from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from bsi_trap.classes.models import (
    AGN_properties,
    BBH_properties,
    DiskProfile,
    Embedded_BH,
    Single_BH_properties,
)

from bsi_trap.config.model_config import ModelConfig
from bsi_trap.evolve.trap_migration import (
    find_catalog_median_reference_trap,
)
from bsi_trap.initialization.embedded_bh import (
    embedded_BH_initialize,
    load_filtered_stellar_population,
)
from bsi_trap.initialization.trap_population import construct_initial_trap_condition
from bsi_trap.disk_profile.pagn_disk_adapter import build_pagn_disk_profile
from bsi_trap.initialization.bbh_parameters import BBH_params_initialize
from bsi_trap.config.rng import resolve_seed, make_rng_streams


@dataclass(frozen=True)
class SampleWorkflowResult:
    output_file: str


@dataclass(frozen=True)
class TrapPopulation:
    embedded_bh: Embedded_BH
    binaries: BBH_properties
    singles: Single_BH_properties


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
        smbh_mass,
        model=model_config.disk_model,
        alpha=model_config.alpha,
        n_resolution=model_config.disk_resolution,
    )

    _, catalog_mass_msun, _ = load_filtered_stellar_population()
    (
        r_trap_cm,
        r_antitrap_cm,
        torque_profile,
        trap_reference_mass_msun,
    ) = find_catalog_median_reference_trap(
        pagn_profile.disk,
        catalog_mass_msun,
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
    agn.pagn_disk = pagn_profile.disk
    agn.trap_reference_mass_msun = trap_reference_mass_msun

    return agn, disk


def initialize_trap_population(
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    rng_pair_up: np.random.Generator,
    rng_nsc: np.random.Generator,
    rng_orbit: np.random.Generator,
    rng_spin: np.random.Generator,
) -> TrapPopulation:
    """Build the initial embedded, binary, and single BH trap populations."""

    # 1. initialize embedded BH properties
    ebh = embedded_BH_initialize(agn, disk, rng_nsc)

    # 2. embedded BH to BBHs in trap
    bbh_1g, sbh_1g = construct_initial_trap_condition(
        model_config=model_config,
        agn=agn,
        disk=disk,
        ebh=ebh,
        rng_pair_up=rng_pair_up,
    )

    # random parameter filling (a,e,chi)
    BBH_params_initialize(
        agn=agn,
        disk=disk,
        bbh=bbh_1g,
        rng_orbit=rng_orbit,
        rng_spin=rng_spin,
        model_config=model_config,
    )

    return TrapPopulation(
        embedded_bh=ebh,
        binaries=bbh_1g,
        singles=sbh_1g,
    )


def run_physics_sample(
    model_config: ModelConfig,
    sample_index: int,
    smbh_mass: float,
    output_file: Path,
    seed: int | None = None,
) -> SampleWorkflowResult:

    # 1. random seed generator
    master_seed = resolve_seed(seed if seed is not None else model_config.master_seed)
    sample_seed = master_seed if seed is not None else master_seed + sample_index
    rng = make_rng_streams(sample_seed)

    # 2. build environment
    agn, disk = _build_environment(
        model_config=model_config,
        smbh_mass=smbh_mass,
    )

    # 3. initialize trap population
    initialize_trap_population(
        model_config=model_config,
        agn=agn,
        disk=disk,
        rng_pair_up=rng.pair_up,
        rng_nsc=rng.nsc,
        rng_orbit=rng.dynamics,
        rng_spin=rng.spin,
    )

    # 4. data storage

    return SampleWorkflowResult(str(output_file))
