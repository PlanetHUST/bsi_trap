"""Chronological event orchestration for one deterministic realization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import heapq
from pathlib import Path
from typing import Any

import numpy as np

from bsi_trap.dataoi.catalog import DEFAULT_CATALOG
from bsi_trap.physics.bsi import (
    apply_phenomenological_bsi,
    calculate_encounter_environment,
    sample_next_bsi_event,
)
from bsi_trap.physics.disk import build_disk_profile
from bsi_trap.physics.evolution import (
    PetersEvolutionResult,
    compute_merger_remnant,
    evolve_binary_peters,
)
from bsi_trap.physics.pairing import (
    formation_radius_cm,
    pair_delay_s,
    select_pair_candidate_ids,
)
from bsi_trap.physics.population import construct_arrivals, sample_embedded_population
from bsi_trap.simulation.models import (
    ArrivalTimeline,
    BSIRecord,
    BinaryPopulation,
    DemoConfig,
    DemoResult,
    DiskProfile,
    EmbeddedPopulation,
    MergerRecord,
    PairingRecord,
    SimulationState,
    StateEventRecord,
)
from bsi_trap.units import AU_CM, MYR_S


class EventType(str, Enum):
    BH_ARRIVAL = "BH_ARRIVAL"
    BINARY_FORMATION = "BINARY_FORMATION"
    BINARY_MERGER = "BINARY_MERGER"
    BSI = "BSI"
    AGN_END = "AGN_END"


EVENT_PRIORITY = {
    EventType.BH_ARRIVAL: 0,
    EventType.BINARY_FORMATION: 1,
    EventType.BINARY_MERGER: 2,
    EventType.BSI: 3,
    EventType.AGN_END: 99,
}


@dataclass(order=True, frozen=True)
class _ScheduledEvent:
    time_s: float
    priority: int
    sequence: int
    event_type: EventType = field(compare=False)
    payload: tuple[Any, ...] = field(default=(), compare=False)


def validate_config(config: DemoConfig) -> None:
    """Reject invalid scientific or numerical configuration values."""
    if config.smbh_mass_msun <= 0.0:
        raise ValueError("SMBH mass [Msun] must be positive")
    if config.agn_lifetime_myr <= 0.0:
        raise ValueError("AGN lifetime [Myr] must be positive")
    if config.population_size < 0:
        raise ValueError("population size must be nonnegative")
    if config.disk_model not in ("sirko", "thompson"):
        raise ValueError("disk model must be 'sirko' or 'thompson'")
    if config.disk_resolution < 32:
        raise ValueError("disk resolution must be at least 32")
    if config.alpha <= 0.0:
        raise ValueError("disk alpha must be positive")
    if config.disk_eddington_ratio <= 0.0:
        raise ValueError("disk Eddington ratio must be positive")
    if config.disk_accretion_rate_g_s is not None and config.disk_accretion_rate_g_s <= 0.0:
        raise ValueError("disk accretion rate [g/s] must be positive")
    if config.thompson_outer_radius_rg <= 0.0:
        raise ValueError("Thompson outer radius [rg] must be positive")
    if (
        config.thompson_outer_accretion_rate_msun_per_yr is not None
        and config.thompson_outer_accretion_rate_msun_per_yr <= 0.0
    ):
        raise ValueError("Thompson outer accretion rate [Msun/yr] must be positive")
    if config.torque_smoothing < 0.0:
        raise ValueError("torque smoothing must be nonnegative")
    if config.gap_opening_condition not in ("kii", "crida"):
        raise ValueError("gap-opening condition must be 'kii' or 'crida'")
    if config.trap_min_separation_cells < 0.0:
        raise ValueError("trap minimum separation [cells] must be nonnegative")
    if config.migration_arrival_tolerance_cells <= 0.0:
        raise ValueError("migration arrival tolerance [cells] must be positive")
    if config.pair_delay_scale <= 0.0:
        raise ValueError("pair delay scale must be positive")
    if config.pairing_region_scale_heights <= 0.0:
        raise ValueError("pairing region scale [H] must be positive")
    if config.initial_bbh_sma_au <= 0.0:
        raise ValueError("initial binary semi-major axis [AU] must be positive")
    if not 0.0 <= config.initial_bbh_eccentricity < 1.0:
        raise ValueError("initial binary eccentricity must be in [0, 1)")
    if config.gw_relative_tolerance <= 0.0:
        raise ValueError("GW relative tolerance must be positive")
    if config.gw_max_step_fraction <= 0.0:
        raise ValueError("GW maximum step fraction must be positive")
    if config.bsi_cross_section_sma <= 0.0:
        raise ValueError("BSI cross-section scale must be positive")
    if not 0.0 <= config.bsi_hardening_fraction < 1.0:
        raise ValueError("BSI hardening fraction must be in [0, 1)")
    if not 0.0 <= config.bsi_eccentricity_pump < 1.0:
        raise ValueError("BSI eccentricity pump must be in [0, 1)")


def run_event_loop(
    config: DemoConfig,
    disk: DiskProfile,
    population: EmbeddedPopulation,
    arrivals: ArrivalTimeline,
    state: SimulationState,
) -> SimulationState:
    """Process arrivals and binary formation in strict chronological order."""
    end_time_s = config.agn_lifetime_myr * MYR_S
    queue: list[_ScheduledEvent] = []
    next_sequence = 0

    def schedule(
        time_s: float,
        event_type: EventType,
        payload: tuple[Any, ...] = (),
    ) -> None:
        nonlocal next_sequence
        heapq.heappush(
            queue,
            _ScheduledEvent(
                time_s=float(time_s),
                priority=EVENT_PRIORITY[event_type],
                sequence=next_sequence,
                event_type=event_type,
                payload=payload,
            ),
        )
        next_sequence += 1

    def advance_binary_to_current_time(
        binary: BinaryPopulation,
    ) -> PetersEvolutionResult:
        if binary.last_evolution_time_s is None:
            raise RuntimeError("binary has no authoritative evolution time")
        elapsed_budget_s = state.current_time_s - binary.last_evolution_time_s
        if elapsed_budget_s < 0.0:
            raise RuntimeError("binary evolution time would move backward")
        evolution = evolve_binary_peters(
            binary.m1_msun,
            binary.m2_msun,
            binary.sma_cm,
            binary.eccentricity,
            elapsed_budget_s,
            relative_tolerance=config.gw_relative_tolerance,
            max_step_fraction=config.gw_max_step_fraction,
        )
        binary.sma_cm = evolution.final_sma_cm
        binary.eccentricity = evolution.final_eccentricity
        binary.last_evolution_time_s = state.current_time_s
        return evolution

    def local_active_single_ids(binary: BinaryPopulation) -> np.ndarray:
        active_ids = np.flatnonzero(state.active_single_mask)
        if active_ids.size == 0:
            return active_ids
        disk_index = disk.closest_index(binary.formation_radius_cm)
        region_width_cm = (
            config.pairing_region_scale_heights
            * float(disk.scale_height_cm[disk_index])
        )
        trap_radius_cm = arrivals.trap_radius_cm[active_ids]
        return active_ids[
            np.isfinite(trap_radius_cm)
            & (np.abs(trap_radius_cm - binary.formation_radius_cm) <= region_width_cm)
        ]

    def plan_binary_event(binary: BinaryPopulation) -> None:
        if binary.status != "active":
            return
        binary.evolution_revision += 1
        revision = binary.evolution_revision
        synchronized = advance_binary_to_current_time(binary)
        if synchronized.reached_integration_stop:
            schedule(
                state.current_time_s,
                EventType.BINARY_MERGER,
                (binary.binary_id, revision, synchronized),
            )
            return
        remaining_time_s = end_time_s - state.current_time_s
        if remaining_time_s <= 0.0:
            return
        gw_outcome = evolve_binary_peters(
            binary.m1_msun,
            binary.m2_msun,
            binary.sma_cm,
            binary.eccentricity,
            remaining_time_s,
            relative_tolerance=config.gw_relative_tolerance,
            max_step_fraction=config.gw_max_step_fraction,
        )
        merger_delay_s = (
            gw_outcome.elapsed_time_s
            if gw_outcome.reached_integration_stop
            else np.inf
        )
        bsi_sample_delay_s = np.inf
        bsi_sample_single_id: int | None = None
        bsi_rate_s_inv = 0.0
        if config.enable_bsi:
            active_ids = local_active_single_ids(binary)
            environment = calculate_encounter_environment(
                config,
                disk,
                binary,
                active_ids,
                population.mass_msun[active_ids],
            )
            sample = sample_next_bsi_event(environment, state.rng)
            bsi_sample_delay_s = sample.delay_s
            bsi_sample_single_id = sample.single_bh_id
            bsi_rate_s_inv = sample.encounter_rate_s_inv
        if (
            bsi_sample_single_id is not None
            and bsi_sample_delay_s < merger_delay_s
            and bsi_sample_delay_s <= remaining_time_s
        ):
            schedule(
                state.current_time_s + bsi_sample_delay_s,
                EventType.BSI,
                (
                    binary.binary_id,
                    revision,
                    bsi_sample_single_id,
                    bsi_rate_s_inv,
                ),
            )
        elif np.isfinite(merger_delay_s) and merger_delay_s <= remaining_time_s:
            schedule(
                state.current_time_s + merger_delay_s,
                EventType.BINARY_MERGER,
                (binary.binary_id, revision, gw_outcome),
            )

    def replan_all_active_binaries() -> None:
        for binary in state.binaries:
            plan_binary_event(binary)

    for bh_id in arrivals.order:
        arrival_time_s = float(arrivals.arrival_time_s[int(bh_id)])
        if arrival_time_s <= end_time_s:
            schedule(arrival_time_s, EventType.BH_ARRIVAL, (int(bh_id),))
    schedule(end_time_s, EventType.AGN_END)

    def try_schedule_pair(trigger_bh_id: int) -> None:
        candidate_ids = select_pair_candidate_ids(
            state,
            arrivals,
            disk,
            trigger_bh_id,
            region_scale_heights=config.pairing_region_scale_heights,
        )
        if candidate_ids is None:
            return
        bh1_id, bh2_id = candidate_ids
        pair_radius_cm = formation_radius_cm(arrivals, bh1_id, bh2_id)
        delay_s = pair_delay_s(
            disk,
            pair_radius_cm,
            float(population.mass_msun[bh1_id]),
            float(population.mass_msun[bh2_id]),
            delay_scale=config.pair_delay_scale,
        )
        formation_time_s = max(
            state.current_time_s,
            float(arrivals.arrival_time_s[bh1_id]),
            float(arrivals.arrival_time_s[bh2_id]),
        ) + delay_s
        if not np.isfinite(formation_time_s) or formation_time_s > end_time_s:
            return
        state.reserve_pair(bh1_id, bh2_id)
        later_arrival_time_s = max(
            float(arrivals.arrival_time_s[bh1_id]),
            float(arrivals.arrival_time_s[bh2_id]),
        )
        schedule(
            formation_time_s,
            EventType.BINARY_FORMATION,
            (bh1_id, bh2_id, pair_radius_cm, delay_s, later_arrival_time_s),
        )

    while queue:
        event = heapq.heappop(queue)
        if event.time_s < state.current_time_s:
            raise RuntimeError("event queue produced a decreasing simulation time")
        if event.time_s > end_time_s:
            break
        state.advance_time(event.time_s)

        if event.event_type is EventType.BH_ARRIVAL:
            bh_id = int(event.payload[0])
            state.mark_arrived(bh_id)
            state.event_history.append(
                StateEventRecord(
                    time_s=state.current_time_s,
                    event_type=event.event_type.value,
                    bh_ids=(bh_id,),
                )
            )
            try_schedule_pair(bh_id)
            replan_all_active_binaries()
            continue

        if event.event_type is EventType.BINARY_FORMATION:
            bh1_id, bh2_id = int(event.payload[0]), int(event.payload[1])
            pair_radius_cm = float(event.payload[2])
            delay_s = float(event.payload[3])
            later_arrival_time_s = float(event.payload[4])
            state.form_reserved_pair(bh1_id, bh2_id)
            binary_id = len(state.binaries)
            binary = BinaryPopulation(
                binary_id=binary_id,
                bh1_id=bh1_id,
                bh2_id=bh2_id,
                m1_msun=float(population.mass_msun[bh1_id]),
                m2_msun=float(population.mass_msun[bh2_id]),
                formation_time_s=state.current_time_s,
                formation_radius_cm=pair_radius_cm,
                sma_cm=config.initial_bbh_sma_au * AU_CM,
                eccentricity=config.initial_bbh_eccentricity,
            )
            state.binaries.append(binary)
            state.pairing_records.append(
                PairingRecord(
                    binary_id=binary_id,
                    bh1_id=bh1_id,
                    bh2_id=bh2_id,
                    later_arrival_time_s=later_arrival_time_s,
                    pair_delay_s=delay_s,
                    formation_time_s=state.current_time_s,
                    formation_radius_cm=pair_radius_cm,
                )
            )
            state.event_history.append(
                StateEventRecord(
                    time_s=state.current_time_s,
                    event_type=event.event_type.value,
                    bh_ids=(bh1_id, bh2_id),
                    binary_id=binary_id,
                )
            )
            replan_all_active_binaries()
            continue

        if event.event_type is EventType.BINARY_MERGER:
            binary_id = int(event.payload[0])
            revision = int(event.payload[1])
            planned_evolution = event.payload[2]
            if not isinstance(planned_evolution, PetersEvolutionResult):
                raise RuntimeError("binary merger event has an invalid evolution payload")
            binary = state.binaries[binary_id]
            if binary.status != "active" or revision != binary.evolution_revision:
                continue
            if not planned_evolution.reached_integration_stop:
                raise RuntimeError("scheduled binary merger did not reach its stop radius")
            evolution = planned_evolution
            binary.sma_cm = evolution.final_sma_cm
            binary.eccentricity = evolution.final_eccentricity
            binary.last_evolution_time_s = state.current_time_s
            binary.status = "merged"
            state.mark_merged(binary.bh1_id, binary.bh2_id)
            remnant = compute_merger_remnant(binary.m1_msun, binary.m2_msun)
            state.merger_records.append(
                MergerRecord(
                    binary_id=binary_id,
                    bh1_id=binary.bh1_id,
                    bh2_id=binary.bh2_id,
                    merger_time_s=state.current_time_s,
                    integration_stop_radius_cm=evolution.integration_stop_radius_cm,
                    final_mass_msun=remnant.final_mass_msun,
                    final_spin=remnant.final_spin,
                    remnant_model_name=remnant.model_name,
                )
            )
            state.event_history.append(
                StateEventRecord(
                    time_s=state.current_time_s,
                    event_type=event.event_type.value,
                    bh_ids=(binary.bh1_id, binary.bh2_id),
                    binary_id=binary_id,
                )
            )
            continue

        if event.event_type is EventType.BSI:
            binary_id = int(event.payload[0])
            revision = int(event.payload[1])
            single_bh_id = int(event.payload[2])
            encounter_rate_s_inv = float(event.payload[3])
            binary = state.binaries[binary_id]
            if binary.status != "active" or revision != binary.evolution_revision:
                continue
            if not state.active_single_mask[single_bh_id]:
                raise RuntimeError("scheduled BSI participant is no longer an active single")
            evolution = advance_binary_to_current_time(binary)
            if evolution.reached_integration_stop:
                schedule(
                    state.current_time_s,
                    EventType.BINARY_MERGER,
                    (binary_id, revision, evolution),
                )
                continue
            sma_before_cm = binary.sma_cm
            eccentricity_before = binary.eccentricity
            binary.sma_cm, binary.eccentricity = apply_phenomenological_bsi(
                config,
                binary.sma_cm,
                binary.eccentricity,
            )
            binary.bsi_encounter_count += 1
            if binary.bsi_encounter_count > 10_000:
                raise RuntimeError("binary exceeded the BSI event safety limit")
            state.bsi_records.append(
                BSIRecord(
                    binary_id=binary_id,
                    single_bh_id=single_bh_id,
                    event_time_s=state.current_time_s,
                    encounter_rate_s_inv=encounter_rate_s_inv,
                    sma_before_cm=sma_before_cm,
                    sma_after_cm=binary.sma_cm,
                    eccentricity_before=eccentricity_before,
                    eccentricity_after=binary.eccentricity,
                    prescription_name="phenomenological_fractional_hardening",
                )
            )
            state.event_history.append(
                StateEventRecord(
                    time_s=state.current_time_s,
                    event_type=event.event_type.value,
                    bh_ids=(single_bh_id,),
                    binary_id=binary_id,
                )
            )
            plan_binary_event(binary)
            continue

        if event.event_type is EventType.AGN_END:
            for binary in state.binaries:
                if binary.status == "active":
                    evolution = advance_binary_to_current_time(binary)
                    if evolution.reached_integration_stop:
                        raise RuntimeError("active binary reached merger before AGN_END")
                    binary.status = "agn_end"
            state.release_pending_pairs()
            state.event_history.append(
                StateEventRecord(
                    time_s=state.current_time_s,
                    event_type=event.event_type.value,
                )
            )
            break

        raise RuntimeError(f"unsupported event type at this gate: {event.event_type}")

    state.validate_masks()
    return state


def run_simulation(
    config: DemoConfig,
    *,
    catalog_path: Path = DEFAULT_CATALOG,
) -> DemoResult:
    """Run disk, population, migration arrivals, and chronological pairing."""
    validate_config(config)
    rng = np.random.default_rng(config.seed)
    disk = build_disk_profile(config)
    population = sample_embedded_population(config, disk, rng, catalog_path)
    arrivals = construct_arrivals(config, disk, population)
    state = SimulationState.from_population(population, rng)
    run_event_loop(config, disk, population, arrivals, state)
    return DemoResult(
        config=config,
        disk=disk,
        population=population,
        arrivals=arrivals,
        state=state,
    )


__all__ = ["EventType", "run_event_loop", "run_simulation", "validate_config"]
