"""Visible one-pass simulation path for the compact demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bsi_trap.dataoi.catalog import DEFAULT_CATALOG
from bsi_trap.simulation.models import DemoConfig, DemoResult, BinaryCollect
from bsi_trap.units import MYR


def validate_config(config: DemoConfig) -> None:
    if config.smbh_mass_msun <= 0.0:
        raise ValueError("SMBH mass must be positive")
    if config.agn_lifetime_myr <= 0.0:
        raise ValueError("AGN lifetime must be positive")
    if config.population_size <= 0:
        raise ValueError("Population size must be positive")
    if config.disk_resolution < 32:
        raise ValueError("Disk resolution must be at least 32")
    if config.alpha <= 0.0:
        raise ValueError("Disk alpha must be positive")
    if config.migration_time_myr <= 0.0:
        raise ValueError("Migration timescale must be positive")
    if config.pair_delay_scale <= 0.0:
        raise ValueError("Pair delay scale must be positive")
    if config.initial_bbh_sma_au <= 0.0:
        raise ValueError("Initial BBH semi-major axis must be positive")
    if not 0.0 <= config.initial_bbh_eccentricity < 1.0:
        raise ValueError("Initial BBH eccentricity must be in [0, 1)")
    if config.bsi_cross_section_sma <= 0.0:
        raise ValueError("BSI cross-section scale must be positive")
    if not 0.0 <= config.bsi_hardening_fraction < 1.0:
        raise ValueError("BSI hardening fraction must be in [0, 1)")
    if config.bsi_eccentricity_pump < 0.0:
        raise ValueError("BSI eccentricity pump must be nonnegative")


def run_simulation(
    config: DemoConfig,
    *,
    catalog_path: Path = DEFAULT_CATALOG,
):
    """Run the compact path: disk -> population -> pair-up -> BBH evolution."""
    validate_config(config)

    from bsi_trap.physics import (
        build_disk_profile,
        construct_arrivals,
        pairing_kernal,
        evolve_kernal,
        sample_embedded_population,
    )

    rng = np.random.default_rng(config.seed)
    disk = build_disk_profile(config)
    population = sample_embedded_population(config, disk, rng, catalog_path)
    arrivals = construct_arrivals(config, disk, population)
    current_time_s = 0
    end_time_s = config.agn_lifetime_myr * MYR
    bbh_list = BinaryCollect(current_time_s)

    while current_time_s < end_time_s:
        pairing_kernal(current_time_s, config, disk, arrivals, bbh_list, rng)

    return DemoResult(
        config, disk, population, arrivals, pairing, binaries, evolution, paired_mask
    )


__all__ = ["run_simulation", "validate_config"]
