from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bsi_trap.classes import AGN_properties, BBH_properties, DiskProfile, merger_flags
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.constant import constant as const
from bsi_trap.disk_profile.migration_torque import find_outermost_stable_trap
from bsi_trap.disk_profile.pagn_disk_adapter import build_pagn_disk_profile
from bsi_trap.evolve.gas_harden import evolve_gas
from bsi_trap.evolve.gw import evolve_gw_only
from bsi_trap.evolve.physical_quantity import threshold_radius
from bsi_trap.storage.contract import HDF5Contract
from bsi_trap.storage.save_gen import save_generation_to_file


@dataclass(frozen=True)
class SampleWorkflowResult:
    output_file: str
    merger_count: int
    generation_count: int = 1


def _build_environment(
    model_config: ModelConfig,
    smbh_mass: float,
) -> tuple[AGN_properties, DiskProfile, int]:
    pagn_profile = build_pagn_disk_profile(
        smbh_mass,
        model=model_config.disk_model,
        n_resolution=model_config.disk_resolution,
    )
    rtrap, trap_radii, torque_profile = find_outermost_stable_trap(
        pagn_profile.disk,
        migrator_mass_msun=model_config.migrator_mass_msun,
    )

    disk = DiskProfile(
        radius_cm=pagn_profile.r_list,
        h_over_r=pagn_profile.h_list,
        sigma_cgs=pagn_profile.sigma_list,
        cs_cgs=pagn_profile.cs_list,
        r_in_cm=pagn_profile.r_in,
        r_out_cm=pagn_profile.r_out,
        r_trap_cm=rtrap,
        r_antitrap_cm=rtrap,
    )

    agn = AGN_properties(
        M_SMBH_msun=smbh_mass,
        lifetime_myr=model_config.agn_lifetime_myr,
    )
    agn.alpha = 0.01
    agn.rg = pagn_profile.rg
    agn.rtrap = rtrap
    agn.disk_profile = disk
    agn.torque_profile = torque_profile

    return agn, disk, int(len(trap_radii))


def _initialize_synthetic_bbh(
    rng: np.random.Generator,
    rtrap_cm: float,
) -> BBH_properties:
    bbh = BBH_properties(1)
    bbh.idi[0] = 0
    bbh.m1_msun[0] = rng.uniform(20.0, 40.0)
    bbh.m2_msun[0] = rng.uniform(10.0, bbh.m1_msun[0])
    bbh.ecc[0] = rng.uniform(0.0, 0.3)
    bbh.chi1[0] = rng.uniform(0.0, 0.3)
    bbh.chi2[0] = rng.uniform(0.0, 0.3)
    bbh.th1[0] = 0.0
    bbh.th2[0] = 0.0
    bbh.A_cm[0] = rtrap_cm

    m1 = bbh.m1_msun[0] * const.msun
    m2 = bbh.m2_msun[0] * const.msun
    bbh.sma_cm[0] = 1.2 * threshold_radius(m1, m2)
    return bbh


def _evolve_sample(
    model_config: ModelConfig,
    bbh: BBH_properties,
    disk: DiskProfile,
    agn: AGN_properties,
) -> tuple[BBH_properties, merger_flags, float]:
    flags = merger_flags(bbh.num)

    if model_config.gas_mode == "with_gas":
        bbh, delay_s, _, flags = evolve_gas(
            disk,
            bbh,
            agn,
            0,
            flags,
            t_ini_s=0.0,
        )
    else:
        bbh, delay_s, _, flags = evolve_gw_only(
            bbh,
            agn,
            0,
            flags,
            t_ini_s=0.0,
        )

    return bbh, flags, float(delay_s)


def _save_sample(
    *,
    model_config: ModelConfig,
    sample_index: int,
    agn: AGN_properties,
    disk: DiskProfile,
    trap_count: int,
    bbh: BBH_properties,
    flags: merger_flags,
    delay_s: float,
    output_root: Path,
) -> Path:
    contract = HDF5Contract(root_dir=output_root)
    trap_radius = agn.rtrap

    generation_data = {
        contract.key_agn: {
            "smbh_mass_msun": agn.M_SMBH_msun,
            "agn_lifetime_myr": agn.AGNlifetime,
            "alpha": agn.alpha,
            "rg": agn.rg,
            "rtrap": trap_radius,
            "rtrap_rg": trap_radius / agn.rg,
            "trap_count": trap_count,
            "disk_model": model_config.disk_model,
            "h_at_trap": disk.get_h(trap_radius),
            "sigma_at_trap_cgs": disk.get_sigma(trap_radius),
            "cs_at_trap_cgs": disk.get_cs(trap_radius),
        },
        contract.key_bbh: bbh,
        contract.key_flags: flags,
        contract.key_timescales: {
            "tini_myr": np.array([0.0]),
            "tdel_myr": np.array([delay_s / const.Myr]),
            "tmerg_myr": np.array([delay_s / const.Myr]),
        },
        contract.key_model_config: model_config,
        contract.key_generation_type: "first_generation",
        contract.key_process_id: os.getpid(),
    }

    return save_generation_to_file(
        simulation_type=model_config.simulation_type,
        sample_index=sample_index,
        generation=1,
        generation_data=generation_data,
        contract=contract,
    )


def run_physics_sample(
    *,
    model_config: ModelConfig,
    sample_index: int,
    smbh_mass: float,
    seed: int,
    output_file: Path,
) -> SampleWorkflowResult:
    """Run one concrete disk, trap, and single-BBH workflow."""
    rng = np.random.default_rng(seed)
    agn, disk, trap_count = _build_environment(model_config, smbh_mass)
    bbh = _initialize_synthetic_bbh(rng, agn.rtrap)
    bbh, flags, delay_s = _evolve_sample(model_config, bbh, disk, agn)
    saved_file = _save_sample(
        model_config=model_config,
        sample_index=sample_index,
        agn=agn,
        disk=disk,
        trap_count=trap_count,
        bbh=bbh,
        flags=flags,
        delay_s=delay_s,
        output_root=output_file.parent,
    )

    return SampleWorkflowResult(
        output_file=str(saved_file),
        merger_count=int(np.sum(flags.merger_flag)),
    )
