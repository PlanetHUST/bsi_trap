"""Embedded-BH sampling and torque-derived migration arrivals."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bsi_trap.dataoi.catalog import DEFAULT_CATALOG, load_remnant_masses
from bsi_trap.physics.migration import MigrationResult, migrate_object
from bsi_trap.physics.migration_torque import (
    MigrationParameters,
    TorqueProfile,
    compute_torque_profile,
)
from bsi_trap.simulation.models import (
    ArrivalTimeline,
    DemoConfig,
    DiskProfile,
    EmbeddedPopulation,
)
from bsi_trap.units import MYR_S


def sample_embedded_population(
    config: DemoConfig,
    disk: DiskProfile,
    rng: np.random.Generator,
    catalog_path: Path = DEFAULT_CATALOG,
) -> EmbeddedPopulation:
    """Sample immutable masses and initial radii with permanent IDs."""
    masses = load_remnant_masses(catalog_path, max_rows=config.catalog_rows)
    mass_msun = rng.choice(masses, size=config.population_size, replace=True)
    radial_power = 3.0 - 7.0 / 4.0
    uniform_draw = rng.random(config.population_size)
    radius_cm = (
        disk.inner_radius_cm**radial_power
        + uniform_draw
        * (disk.outer_radius_cm**radial_power - disk.inner_radius_cm**radial_power)
    ) ** (1.0 / radial_power)
    return EmbeddedPopulation(
        bh_id=np.arange(config.population_size, dtype=np.int64),
        mass_msun=mass_msun,
        initial_radius_cm=radius_cm,
    )


def arrival_timeline_from_results(
    population: EmbeddedPopulation,
    migration_results: list[MigrationResult],
) -> ArrivalTimeline:
    """Translate ID-aligned migration outcomes without reordering population rows."""
    if len(migration_results) != population.size:
        raise ValueError("migration results must contain one row per physical BH")
    for expected_bh_id, result in enumerate(migration_results):
        if result.bh_id != expected_bh_id:
            raise ValueError("migration results must be ordered by permanent BH ID")

    arrival_time_s = np.asarray(
        [result.arrival_time_s for result in migration_results],
        dtype=float,
    )
    reached_mask = np.asarray(
        [result.reached_stable_trap for result in migration_results],
        dtype=bool,
    )
    arrival_time_s = np.where(reached_mask, arrival_time_s, np.inf)
    reached_bh_ids = population.bh_id[reached_mask]
    if reached_bh_ids.size:
        order_indices = np.lexsort(
            (reached_bh_ids, arrival_time_s[reached_mask])
        )
        arrival_order = reached_bh_ids[order_indices]
    else:
        arrival_order = np.array([], dtype=np.int64)
    return ArrivalTimeline(
        bh_id=population.bh_id,
        arrival_time_s=arrival_time_s,
        migration_elapsed_time_s=np.asarray(
            [result.elapsed_time_s for result in migration_results],
            dtype=float,
        ),
        mass_msun=population.mass_msun,
        reached_mask=reached_mask,
        order=arrival_order,
        final_radius_cm=np.asarray(
            [result.final_radius_cm for result in migration_results],
            dtype=float,
        ),
        trap_radius_cm=np.asarray(
            [
                np.nan if result.trap_radius_cm is None else result.trap_radius_cm
                for result in migration_results
            ],
            dtype=float,
        ),
        stop_reason=np.asarray(
            [result.stop_reason.value for result in migration_results],
            dtype=str,
        ),
    )


def construct_arrivals(
    config: DemoConfig,
    disk: DiskProfile,
    population: EmbeddedPopulation,
) -> ArrivalTimeline:
    """Migrate every BH through its own mass-dependent torque profile."""
    max_time_s = config.agn_lifetime_myr * MYR_S
    torque_cache: dict[float, TorqueProfile] = {}
    migration_results: list[MigrationResult] = []
    parameters = MigrationParameters(alpha=config.alpha)
    for bh_id in population.bh_id:
        index = int(bh_id)
        mass_msun = float(population.mass_msun[index])
        if mass_msun not in torque_cache:
            torque_cache[mass_msun] = compute_torque_profile(
                disk,
                smbh_mass_msun=config.smbh_mass_msun,
                migrator_mass_msun=mass_msun,
                smoothing=config.torque_smoothing,
                params=parameters,
                gap_opening_condition=config.gap_opening_condition,
                consider_type_ii_migration=config.consider_type_ii_migration,
            )
        migration_results.append(
            migrate_object(
                initial_radius_cm=float(population.initial_radius_cm[index]),
                mass_msun=mass_msun,
                disk=disk,
                torque_profile=torque_cache[mass_msun],
                max_time_s=max_time_s,
                bh_id=index,
                arrival_tolerance_cells=config.migration_arrival_tolerance_cells,
                min_trap_separation_cells=config.trap_min_separation_cells,
            )
        )
    return arrival_timeline_from_results(population, migration_results)


__all__ = [
    "arrival_timeline_from_results",
    "construct_arrivals",
    "sample_embedded_population",
]
