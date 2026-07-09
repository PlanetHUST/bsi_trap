from __future__ import annotations

from bsi_trap.simulation.state.models import AGN_properties, DiskProfile
from bsi_trap.simulation.config.model_config import ModelConfig
from bsi_trap.physics.disk.pagn_adapter import build_pagn_disk_profile
from bsi_trap.physics.trap.migration import find_catalog_median_reference_trap
from bsi_trap.simulation.trap_init.embedded_bh import load_filtered_stellar_population


def make_environment(
    model_config: ModelConfig,
    smbh_mass: float,
) -> tuple[AGN_properties, DiskProfile]:
    agn = AGN_properties(
        M_SMBH_msun=smbh_mass,
        lifetime_myr=model_config.agn_lifetime_myr,
    )
    agn.alpha = model_config.alpha

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
        _,
    ) = find_catalog_median_reference_trap(
        pagn_profile.disk,
        catalog_mass_msun,
    )

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

    return agn, disk
