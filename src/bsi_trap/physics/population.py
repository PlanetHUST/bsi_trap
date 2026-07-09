"""Simple embedded-remnant population and trap-arrival logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bsi_trap.dataoi import DEFAULT_CATALOG, load_remnant_masses
from bsi_trap.simulation.models import (
    ArrivalTimeline,
    DemoConfig,
    DiskProfile,
    EmbeddedPopulation,
)
from bsi_trap.units import MYR


def sample_embedded_population(
    config: DemoConfig,
    disk: DiskProfile,
    rng: np.random.Generator,
    catalog_path: Path = DEFAULT_CATALOG,
) -> EmbeddedPopulation:
    masses = load_remnant_masses(catalog_path, max_rows=config.catalog_rows)
    mass_msun = rng.choice(masses, size=config.population_size, replace=True)
    power = 3.0 - 7.0 / 4.0
    u = rng.random(config.population_size)
    radius_cm = (
        disk.r_in_cm**power + u * (disk.r_out_cm**power - disk.r_in_cm**power)
    ) ** (1.0 / power)
    return EmbeddedPopulation(mass_msun=mass_msun, initial_radius_cm=radius_cm)


def construct_arrivals(
    config: DemoConfig,
    disk: DiskProfile,
    population: EmbeddedPopulation,
) -> ArrivalTimeline:
    """Assign simple mass-dependent migration times to the trap."""
    distance = np.abs(np.log(population.initial_radius_cm / disk.r_trap_cm))
    mass_factor = 10.0 / np.maximum(population.mass_msun, 1.0e-12)
    # simple migration timescale version here
    migration_s = config.migration_time_myr * MYR * (0.15 + distance) * mass_factor
    # ignore formation timescale here
    reached_mask = migration_s <= config.agn_lifetime_myr * MYR
    reached_idx = np.flatnonzero(reached_mask)
    order = reached_idx[np.argsort(migration_s[reached_idx], kind="stable")]
    return ArrivalTimeline(
        migration_s, population.mass_msun, reached_mask, order.astype(np.intp)
    )


__all__ = ["construct_arrivals", "sample_embedded_population"]
