"""Chronological event-loop and state-conservation checks."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from bsi_trap.physics.bsi import BSIEventSample
from bsi_trap.dataoi.summary import format_summary, result_to_dict, write_result_json
from bsi_trap.simulation.models import (
    ArrivalTimeline,
    DemoConfig,
    DemoResult,
    DiskProfile,
    EmbeddedPopulation,
    SimulationState,
)
from bsi_trap.simulation.run import EventType, run_event_loop
from bsi_trap.simulation.run import run_simulation, validate_config


def test_default_configuration_is_valid() -> None:
    validate_config(DemoConfig())


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("smbh_mass_msun", 0.0),
        ("agn_lifetime_myr", 0.0),
        ("population_size", -1),
        ("alpha", 0.0),
        ("disk_resolution", 31),
        ("migration_arrival_tolerance_cells", 0.0),
        ("initial_bbh_sma_au", 0.0),
        ("initial_bbh_eccentricity", 1.0),
    ],
)
def test_invalid_configuration_is_rejected(
    field_name: str,
    invalid_value: float | int,
) -> None:
    with pytest.raises(ValueError):
        validate_config(replace(DemoConfig(), **{field_name: invalid_value}))


def _inputs() -> tuple[DemoConfig, DiskProfile, EmbeddedPopulation, ArrivalTimeline]:
    config = DemoConfig(
        population_size=5,
        seed=17,
        agn_lifetime_myr=1.0e-5,
        initial_bbh_sma_au=0.01,
    )
    radius_cm = np.linspace(1.0e15, 2.0e15, 8)
    ones = np.ones(radius_cm.shape)
    disk = DiskProfile(
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
    population = EmbeddedPopulation(
        bh_id=np.arange(5),
        mass_msun=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        initial_radius_cm=np.linspace(1.1e15, 1.5e15, 5),
    )
    arrivals = ArrivalTimeline(
        bh_id=population.bh_id,
        arrival_time_s=np.array([1.0, 2.0, 3.0, 4.0, np.inf]),
        migration_elapsed_time_s=np.array([1.0, 2.0, 3.0, 4.0, 100.0]),
        mass_msun=population.mass_msun,
        reached_mask=np.array([True, True, True, True, False]),
        order=np.array([0, 1, 2, 3]),
        final_radius_cm=np.full(5, 1.5e15),
        trap_radius_cm=np.array([1.5e15, 1.5e15, 1.5e15, 1.5e15, np.nan]),
        stop_reason=np.array(
            ["STABLE_TRAP", "STABLE_TRAP", "STABLE_TRAP", "STABLE_TRAP", "AGN_LIFETIME"]
        ),
    )
    return config, disk, population, arrivals


def _run_once() -> SimulationState:
    config, disk, population, arrivals = _inputs()
    state = SimulationState.from_population(
        population,
        np.random.default_rng(config.seed),
    )
    return run_event_loop(config, disk, population, arrivals, state)


def test_event_loop_is_monotonic_and_preserves_state_identity() -> None:
    config, _, population, arrivals = _inputs()
    state = _run_once()
    times_s = np.asarray([event.time_s for event in state.event_history])
    assert np.all(np.diff(times_s) >= 0.0)
    assert np.all(times_s <= config.agn_lifetime_myr * 3.15576e13)
    assert state.event_history[-1].event_type == EventType.AGN_END.value
    assert len(state.binaries) == 2
    component_ids = [
        bh_id
        for binary in state.binaries
        for bh_id in (binary.bh1_id, binary.bh2_id)
    ]
    assert len(component_ids) == len(set(component_ids))
    assert set(component_ids) == {0, 1, 2, 3}
    assert not state.active_single_mask[4]
    assert not arrivals.reached_mask[4]
    np.testing.assert_array_equal(population.bh_id, np.arange(5))
    state.validate_masks()
    for binary in state.binaries:
        assert binary.formation_time_s >= arrivals.arrival_time_s[binary.bh1_id]
        assert binary.formation_time_s >= arrivals.arrival_time_s[binary.bh2_id]


def test_fixed_seed_reproduces_event_history_and_pairings() -> None:
    first = _run_once()
    second = _run_once()
    assert first.event_history == second.event_history
    assert first.pairing_records == second.pairing_records


def test_merger_events_follow_formation_and_update_component_state() -> None:
    config, disk, population, arrivals = _inputs()
    config = replace(
        config,
        initial_bbh_sma_au=5.0e-6,
        initial_bbh_eccentricity=0.0,
    )
    state = SimulationState.from_population(
        population,
        np.random.default_rng(config.seed),
    )
    run_event_loop(config, disk, population, arrivals, state)

    assert len(state.merger_records) == 2
    formation_time_by_binary = {
        record.binary_id: record.formation_time_s for record in state.pairing_records
    }
    for merger in state.merger_records:
        assert merger.merger_time_s >= formation_time_by_binary[merger.binary_id]
        assert merger.remnant_model_name.startswith("toy_")
    assert not np.any(state.paired_mask)
    np.testing.assert_array_equal(
        state.merged_mask,
        np.array([True, True, True, True, False]),
    )


def test_bsi_is_a_discrete_event_using_an_active_single(monkeypatch) -> None:
    config, disk, population, arrivals = _inputs()
    config = replace(
        config,
        initial_bbh_sma_au=0.01,
        initial_bbh_eccentricity=0.0,
        bsi_hardening_fraction=0.9999,
        bsi_eccentricity_pump=0.0,
    )
    arrivals = ArrivalTimeline(
        bh_id=population.bh_id,
        arrival_time_s=np.arange(1.0, 6.0),
        migration_elapsed_time_s=np.arange(1.0, 6.0),
        mass_msun=population.mass_msun,
        reached_mask=np.ones(5, dtype=bool),
        order=np.arange(5),
        final_radius_cm=np.full(5, 1.5e15),
        trap_radius_cm=np.full(5, 1.5e15),
        stop_reason=np.full(5, "STABLE_TRAP"),
    )

    def immediate_sample(environment, _rng):
        if environment.active_single_ids.size == 0:
            return BSIEventSample(np.inf, None, 0.0)
        return BSIEventSample(
            10.0,
            int(environment.active_single_ids[0]),
            environment.total_rate_s_inv,
        )

    monkeypatch.setattr("bsi_trap.simulation.run.sample_next_bsi_event", immediate_sample)
    state = SimulationState.from_population(
        population,
        np.random.default_rng(config.seed),
    )
    run_event_loop(config, disk, population, arrivals, state)

    assert state.bsi_records
    assert all(record.single_bh_id == 4 for record in state.bsi_records)
    assert all(record.sma_after_cm < record.sma_before_cm for record in state.bsi_records)
    assert any(event.event_type == EventType.BSI.value for event in state.event_history)


def test_result_translation_is_auditable_strict_json(tmp_path) -> None:
    config, disk, population, arrivals = _inputs()
    state = SimulationState.from_population(
        population,
        np.random.default_rng(config.seed),
    )
    run_event_loop(config, disk, population, arrivals, state)
    result = DemoResult(config, disk, population, arrivals, state)
    payload = result_to_dict(result)
    encoded = json.dumps(payload, allow_nan=False)

    assert payload["seed"] == config.seed
    assert payload["migration_outcomes"][4]["arrival_time_s"] is None
    assert payload["migration_outcomes"][4]["trap_radius_cm"] is None
    assert "Infinity" not in encoded and "NaN" not in encoded
    assert "BH_ARRIVAL" in encoded
    assert "unreached" in format_summary(result)

    output_path = tmp_path / "simulation_result.json"
    write_result_json(result, output_path)
    assert json.loads(output_path.read_text())["seed"] == config.seed


def test_reduced_physical_end_to_end_path(monkeypatch) -> None:
    monkeypatch.setattr(
        "bsi_trap.physics.population.load_remnant_masses",
        lambda *_args, **_kwargs: np.array([80.0]),
    )
    config = DemoConfig(
        population_size=6,
        seed=42,
        disk_resolution=96,
        gap_opening_condition="crida",
        initial_bbh_sma_au=5.0e-6,
        initial_bbh_eccentricity=0.0,
        enable_bsi=False,
    )
    first = run_simulation(config)
    second = run_simulation(config)

    assert first.arrivals.reached_count == 6
    assert len(first.state.binaries) == 3
    assert len(first.state.merger_records) == 3
    assert first.state.event_history == second.state.event_history
    np.testing.assert_array_equal(first.population.mass_msun, second.population.mass_msun)
    np.testing.assert_array_equal(first.population.initial_radius_cm, second.population.initial_radius_cm)
    assert all(
        record.merger_time_s >= first.state.binaries[record.binary_id].formation_time_s
        for record in first.state.merger_records
    )
