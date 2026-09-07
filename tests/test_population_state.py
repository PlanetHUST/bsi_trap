"""Permanent population identity and state-transition checks."""

from __future__ import annotations

import numpy as np
import pytest

from bsi_trap.physics.migration import MigrationResult, MigrationStopReason
from bsi_trap.physics.population import arrival_timeline_from_results
from bsi_trap.simulation.models import EmbeddedPopulation, SimulationState


def _population() -> EmbeddedPopulation:
    return EmbeddedPopulation(
        bh_id=np.arange(5, dtype=np.int64),
        mass_msun=np.array([8.0, 12.0, 20.0, 30.0, 45.0]),
        initial_radius_cm=np.array([1.0e15, 2.0e15, 3.0e15, 4.0e15, 5.0e15]),
    )


def test_pairing_marks_permanent_ids_without_compacting_population() -> None:
    population = _population()
    original_ids = population.bh_id.copy()
    original_masses = population.mass_msun.copy()
    state = SimulationState.from_population(population, np.random.default_rng(1))

    for bh_id in (0, 2, 4):
        state.mark_arrived(bh_id)
    state.mark_paired(4, 0)

    np.testing.assert_array_equal(population.bh_id, original_ids)
    np.testing.assert_array_equal(population.mass_msun, original_masses)
    assert population.size == 5
    np.testing.assert_array_equal(
        state.active_single_mask,
        np.array([False, False, True, False, False]),
    )
    np.testing.assert_array_equal(
        state.paired_mask,
        np.array([True, False, False, False, True]),
    )
    assert not np.any(state.merged_mask)


def test_state_time_is_monotonic_and_owned_by_state() -> None:
    state = SimulationState.from_population(_population(), np.random.default_rng(1))
    state.advance_time(10.0)
    state.advance_time(10.0)
    assert state.current_time_s == 10.0

    with pytest.raises(ValueError, match="cannot decrease"):
        state.advance_time(9.0)


def test_population_rejects_noncanonical_ids() -> None:
    with pytest.raises(ValueError, match="permanent array index"):
        EmbeddedPopulation(
            bh_id=np.array([1, 2]),
            mass_msun=np.array([10.0, 20.0]),
            initial_radius_cm=np.array([1.0e15, 2.0e15]),
        )


def test_population_arrays_are_immutable() -> None:
    population = _population()
    with pytest.raises(ValueError):
        population.mass_msun[0] = 99.0


def test_arrival_order_uses_time_then_permanent_id() -> None:
    population = _population()
    results = [
        MigrationResult(
            bh_id=bh_id,
            initial_radius_cm=float(population.initial_radius_cm[bh_id]),
            final_radius_cm=float(population.initial_radius_cm[bh_id]),
            elapsed_time_s=(2.0 if bh_id in (0, 4) else float(bh_id + 1)),
            arrival_time_s=(2.0 if bh_id in (0, 4) else np.inf),
            reached_stable_trap=bh_id in (0, 4),
            trap_radius_cm=(3.0e15 if bh_id in (0, 4) else None),
            stop_reason=(
                MigrationStopReason.STABLE_TRAP
                if bh_id in (0, 4)
                else MigrationStopReason.AGN_LIFETIME
            ),
        )
        for bh_id in range(population.size)
    ]
    arrivals = arrival_timeline_from_results(population, results)

    np.testing.assert_array_equal(arrivals.bh_id, population.bh_id)
    np.testing.assert_array_equal(arrivals.order, np.array([0, 4]))
    assert np.all(np.isinf(arrivals.arrival_time_s[[1, 2, 3]]))
    assert not np.any(arrivals.reached_mask[[1, 2, 3]])
