"""Discrete BSI encounter-rate and phenomenological-transition checks."""

from __future__ import annotations

import numpy as np

from bsi_trap.physics.bsi import (
    BSIEventSample,
    apply_phenomenological_bsi,
    calculate_encounter_environment,
    sample_next_bsi_event,
)
from bsi_trap.simulation.models import BinaryPopulation, DemoConfig, DiskProfile


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


def _binary() -> BinaryPopulation:
    return BinaryPopulation(
        binary_id=0,
        bh1_id=0,
        bh2_id=1,
        m1_msun=30.0,
        m2_msun=20.0,
        formation_time_s=100.0,
        formation_radius_cm=1.5e15,
        sma_cm=1.0e11,
        eccentricity=0.2,
    )


def test_encounter_environment_has_dimensionally_positive_rate() -> None:
    environment = calculate_encounter_environment(
        DemoConfig(),
        _disk(),
        _binary(),
        np.array([3, 7]),
        np.array([10.0, 40.0]),
    )
    assert environment.maximum_impact_parameter_cm > 0.0
    assert environment.relative_velocity_cm_s > 0.0
    assert environment.trap_volume_cm3 > 0.0
    assert np.all(environment.per_single_rate_s_inv > 0.0)
    assert environment.total_rate_s_inv == np.sum(environment.per_single_rate_s_inv)


def test_no_active_singles_means_no_bsi_event() -> None:
    environment = calculate_encounter_environment(
        DemoConfig(),
        _disk(),
        _binary(),
        np.array([], dtype=int),
        np.array([], dtype=float),
    )
    sample = sample_next_bsi_event(environment, np.random.default_rng(4))
    assert np.isinf(sample.delay_s)
    assert sample.single_bh_id is None
    assert sample.encounter_rate_s_inv == 0.0


def test_bsi_sampling_is_seed_reproducible_and_uses_physical_ids() -> None:
    environment = calculate_encounter_environment(
        DemoConfig(),
        _disk(),
        _binary(),
        np.array([3, 7]),
        np.array([10.0, 40.0]),
    )
    first = sample_next_bsi_event(environment, np.random.default_rng(9))
    second = sample_next_bsi_event(environment, np.random.default_rng(9))
    assert first == second
    assert first.single_bh_id in {3, 7}


def test_phenomenological_bsi_hardens_and_pumps_eccentricity() -> None:
    config = DemoConfig(bsi_hardening_fraction=0.2, bsi_eccentricity_pump=0.25)
    sma_cm, eccentricity = apply_phenomenological_bsi(config, 1.0e11, 0.2)
    assert sma_cm == 8.0e10
    assert eccentricity == 0.4


def fixed_immediate_bsi_sample(active_bh_id: int) -> BSIEventSample:
    """Small helper for event-loop tests without stochastic waiting-time scale."""
    return BSIEventSample(10.0, active_bh_id, 0.1)
