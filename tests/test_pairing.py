"""Physical-ID pairing and binary-initialization checks."""

from __future__ import annotations

import numpy as np
import pytest

from bsi_trap.physics.pairing import hill_radius_cm, select_pair_candidate_ids
from bsi_trap.simulation.models import (
    ArrivalTimeline,
    BinaryPopulation,
    DiskProfile,
    EmbeddedPopulation,
    SimulationState,
)


def _disk() -> DiskProfile:
    radius_cm = np.linspace(1.0e15, 2.0e15, 8)
    ones = np.ones(radius_cm.shape)
    return DiskProfile(
        model_name="synthetic",
        radius_cm=radius_cm,
        temperature_k=1.0e4 * ones,
        surface_density_g_cm2=2.0e7 * ones,
        density_g_cm3=1.0e-8 * ones,
        sound_speed_cm_s=1.0e6 * ones,
        aspect_ratio=0.1 * ones,
        optical_depth=10.0 * ones,
        opacity_cm2_g=1.0 * ones,
        toomre_q=2.0 * ones,
        omega_s_inv=1.0e-6 * ones,
        effective_temperature_fourth_k4=1.0e12 * ones,
        inner_radius_cm=float(radius_cm[0]),
        outer_radius_cm=float(radius_cm[-1]),
        gravitational_radius_cm=1.0e11,
        schwarzschild_radius_cm=2.0e11,
    )


def _population_and_arrivals() -> tuple[EmbeddedPopulation, ArrivalTimeline]:
    population = EmbeddedPopulation(
        bh_id=np.arange(4),
        mass_msun=np.array([10.0, 20.0, 30.0, 40.0]),
        initial_radius_cm=np.array([1.1e15, 1.2e15, 1.3e15, 1.4e15]),
    )
    arrivals = ArrivalTimeline(
        bh_id=population.bh_id,
        arrival_time_s=np.array([1.0, 2.0, 3.0, 4.0]),
        migration_elapsed_time_s=np.array([1.0, 2.0, 3.0, 4.0]),
        mass_msun=population.mass_msun,
        reached_mask=np.ones(4, dtype=bool),
        order=np.arange(4),
        final_radius_cm=np.full(4, 1.5e15),
        trap_radius_cm=np.full(4, 1.5e15),
        stop_reason=np.full(4, "STABLE_TRAP"),
    )
    return population, arrivals


def test_hill_radius_uses_cube_root_convention() -> None:
    radius_cm = hill_radius_cm(1.0e16, 20.0, 10.0, 1.0e6)
    assert radius_cm == pytest.approx(1.0e16 * (30.0 / 3.0e6) ** (1.0 / 3.0))


def test_pair_selection_translates_local_positions_to_physical_ids() -> None:
    population, arrivals = _population_and_arrivals()
    state = SimulationState.from_population(population, np.random.default_rng(8))
    state.mark_arrived(2)
    state.mark_arrived(3)
    selected = select_pair_candidate_ids(
        state,
        arrivals,
        _disk(),
        3,
        region_scale_heights=1.0,
    )
    assert selected is not None
    assert set(selected) == {2, 3}


def test_binary_requires_complete_physical_initial_conditions() -> None:
    with pytest.raises(ValueError, match="same BH"):
        BinaryPopulation(
            binary_id=0,
            bh1_id=2,
            bh2_id=2,
            m1_msun=10.0,
            m2_msun=10.0,
            formation_time_s=2.0,
            formation_radius_cm=1.5e15,
            sma_cm=1.0e12,
            eccentricity=0.2,
        )
    with pytest.raises(ValueError, match="eccentricity"):
        BinaryPopulation(
            binary_id=0,
            bh1_id=1,
            bh2_id=2,
            m1_msun=10.0,
            m2_msun=10.0,
            formation_time_s=2.0,
            formation_radius_cm=1.5e15,
            sma_cm=1.0e12,
            eccentricity=1.0,
        )
