from __future__ import annotations

from copy import deepcopy

import numpy as np

from bsi_trap.simulation.state.models import (
    AGN_properties,
    BBHEvolverParams,
    DiskProfile,
)
from bsi_trap.simulation.config.model_config import ModelConfig
from bsi_trap.simulation.evolution.generation import (
    EvolutionResult,
    evolve_single_binary,
)
from bsi_trap.physics.binary.tracks import finalize_bbh_tracks
from bsi_trap.simulation.trap_init.population import TrapPopulation
from bsi_trap.simulation.runtime.remnants import build_hierarchical_mergers
from bsi_trap.simulation.runtime.singles import combine_single_populations
from bsi_trap.simulation.runtime.state import (
    ActiveBinary,
    EvolutionRecord,
    StateLog,
    record_event,
)
from bsi_trap.simulation.timeline.bsi import construct_bsi_timeline
from bsi_trap.simulation.timeline.evolution import construct_bbh_evolution_timeline


def binary_is_final(population: TrapPopulation) -> bool:
    bbh = population.binaries

    if bbh.num == 0:
        return True

    return bool(
        np.any(bbh.flags.merger_flag != 0) or np.any(bbh.flags.stop_reason_flag != 0)
    )


def advance_bbh_segment(
    *,
    Evolparams: BBHEvolverParams,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    bbh_candi: ActiveBinary,
    boundary_s: float,
) -> float:
    """Advance an active BBH to a global boundary or to its physical stop."""

    bbh = bbh_candi.population.binaries

    if bbh.num == 0:
        return float(boundary_s)

    t_del_s = evolve_single_binary(
        Evolparams,
        model_config,
        agn,
        disk,
        bbh,
        0,
        segment_stop_s=boundary_s,
    )

    bbh.t_del_s[0] = t_del_s

    return float(bbh.t_ini_s[0] + bbh.t_del_s[0])


def predict_active_event_s(
    *,
    Evolparams: BBHEvolverParams,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    active_binaries: list[ActiveBinary],
    boundary_s: float,
) -> float | None:
    """next active event"""

    event_times: list[float] = []

    for bbh_candi in active_binaries:
        prediction = ActiveBinary(
            record_index=bbh_candi.record_index,
            generation=bbh_candi.generation,
            population=deepcopy(bbh_candi.population),
        )

        event_s = advance_bbh_segment(
            Evolparams=Evolparams,
            model_config=model_config,
            agn=agn,
            disk=disk,
            bbh_candi=prediction,
            boundary_s=boundary_s,
        )

        if binary_is_final(prediction.population) and event_s <= boundary_s:
            event_times.append(float(event_s))

    if not event_times:
        return None

    return min(event_times)


def finalize_active_binary(
    *,
    agn: AGN_properties,
    disk: DiskProfile,
    active: ActiveBinary,
    rng_orbit: np.random.Generator,
) -> EvolutionRecord:
    bbh = active.population.binaries
    finalize_bbh_tracks(bbh)
    bsi_timeline = construct_bsi_timeline(bbh)
    evolution_timeline = construct_bbh_evolution_timeline(
        bbh,
        bsi_timeline=bsi_timeline,
    )
    hierarchical_mergers = build_hierarchical_mergers(
        agn=agn,
        disk=disk,
        bbh=bbh,
        rng=rng_orbit,
    )
    return EvolutionRecord(
        record_index=active.record_index,
        generation=active.generation,
        population=active.population,
        outcome=EvolutionResult(
            evolution_timeline=evolution_timeline,
            hierarchical_mergers=hierarchical_mergers,
        ),
    )


def advance_active_binaries(
    *,
    state: StateLog,
    records: list[EvolutionRecord],
    Evolparams: BBHEvolverParams,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    rng_orbit: np.random.Generator,
    boundary_s: float,
) -> None:

    remaining: list[ActiveBinary] = []
    for active in state.active_binaries:
        event_s = advance_bbh_segment(
            Evolparams=Evolparams,
            model_config=model_config,
            agn=agn,
            disk=disk,
            bbh_candi=active,
            boundary_s=boundary_s,
        )
        if binary_is_final(active.population) and event_s <= boundary_s:
            record = finalize_active_binary(
                agn=agn,
                disk=disk,
                active=active,
                rng_orbit=rng_orbit,
            )
            records.append(record)
            state.live_singles = combine_single_populations(
                state.live_singles,
                record.outcome.hierarchical_mergers.next_generation_singles,
            )
            event_kind = (
                "merge"
                if np.any(active.population.binaries.flags.merger_flag != 0)
                else "stop"
            )
            record_event(
                state,
                kind=event_kind,
                time_s=event_s,
                generation=active.generation,
                record_index=active.record_index,
            )
            continue
        remaining.append(active)
    state.active_binaries = remaining


def activate_pair_event(
    *,
    state: StateLog,
    population: TrapPopulation,
    record_index: int,
) -> None:
    bbh = population.binaries
    if bbh.num == 0:
        return

    generation = int(bbh.generation[0])
    state.live_singles = population.singles
    active = ActiveBinary(
        record_index=record_index,
        generation=generation,
        population=population,
    )
    state.active_binaries.append(active)
    record_event(
        state,
        kind="pair",
        time_s=float(bbh.formation_s[0]),
        generation=generation,
        record_index=record_index,
    )
