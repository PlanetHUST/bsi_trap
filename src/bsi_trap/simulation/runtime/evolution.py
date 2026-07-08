from __future__ import annotations

import numpy as np

from bsi_trap.simulation.state.models import (
    AGN_properties,
    BBHEvolverParams,
    DiskProfile,
)
from bsi_trap.simulation.config.model_config import ModelConfig
from bsi_trap.constant.constant import Myr
from bsi_trap.simulation.evolution.generation import EvolutionResult
from bsi_trap.simulation.initial_conditions.population import (
    TrapArrivalState,
    empty_trap_population,
)
from bsi_trap.simulation.runtime.binary_events import (
    activate_pair_event as _activate_pair_event,
    advance_active_binaries as _advance_active_binaries,
    advance_bbh_segment,
    binary_is_final as _binary_is_final,
    finalize_active_binary as _finalize_active_binary,
    predict_active_event_s as _predict_active_event_s,
)
from bsi_trap.simulation.runtime.bsi_events import (
    active_single_masses_at as _active_single_masses_at,
    prime_bsi_event as _prime_bsi_event,
    sample_next_bsi_event as _sample_next_bsi_event,
)
from bsi_trap.simulation.runtime.pairing import (
    build_bbh_from_pairing as _build_bbh_from_pairing,
    build_pair_event_population as _build_pair_event_population,
    remaining_singles_after_pairing as _remaining_singles_after_pairing,
)
from bsi_trap.simulation.runtime.remnants import HierarchicalMergerLedger
from bsi_trap.simulation.runtime.singles import (
    combine_single_populations as _combine_single_populations,
    copy_singles_by_index as _copy_singles_by_index,
    next_single_arrival_s as _next_single_arrival_s,
)
from bsi_trap.simulation.runtime.state import (
    ActiveBinary,
    EvolutionRecord,
    RuntimeEvent,
    StateLog,
    TrapEvolveRun,
    record_event as _record_event,
)
from bsi_trap.simulation.timeline.models import EvolutionTimeline

__all__ = [
    "ActiveBinary",
    "EvolutionRecord",
    "RuntimeEvent",
    "StateLog",
    "TrapEvolveRun",
    "advance_bbh_segment",
    "evolve_trap_runtime",
]


def evolve_trap_runtime(
    *,
    Evolparams: BBHEvolverParams,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    initial_state: TrapArrivalState,
    rng_pair_up: np.random.Generator,
    rng_orbit: np.random.Generator,
    rng_spin: np.random.Generator,
) -> TrapEvolveRun:
    """Evolve the trap population with one top-level physical event clock."""

    end_time_s = float(agn.lifetime_myr * Myr)
    state = StateLog(
        current_time_s=0.0,
        live_singles=_combine_single_populations(initial_state.singles),
        embedded_bh=initial_state.embedded_bh,
    )
    records: list[EvolutionRecord] = []
    next_record_index = 1

    while state.current_time_s < end_time_s:
        next_arrival_s = _next_single_arrival_s(
            state.live_singles,
            state.current_time_s,
            end_time_s,
        )
        step_end_s = end_time_s if next_arrival_s is None else next_arrival_s

        active_event_s = _predict_active_event_s(
            Evolparams=Evolparams,
            model_config=model_config,
            agn=agn,
            disk=disk,
            active_binaries=state.active_binaries,
            boundary_s=step_end_s,
        )
        if active_event_s is not None and active_event_s < step_end_s:
            step_end_s = active_event_s

        bsi_event = _sample_next_bsi_event(
            model_config=model_config,
            agn=agn,
            disk=disk,
            active_binaries=state.active_binaries,
            live_singles=state.live_singles,
            current_time_s=state.current_time_s,
            step_end_s=step_end_s,
            rng_orbit=rng_orbit,
        )

        if bsi_event is not None and bsi_event.time_s < step_end_s:
            step_end_s = bsi_event.time_s

        pair_population = _build_pair_event_population(
            embedded_bh=state.embedded_bh,
            source_singles=state.live_singles,
            current_time_s=state.current_time_s,
            step_end_s=step_end_s,
            model_config=model_config,
            agn=agn,
            disk=disk,
            rng_pair_up=rng_pair_up,
            rng_orbit=rng_orbit,
            rng_spin=rng_spin,
        )
        pair_time_s = (
            float(pair_population.pairing_timeline.formation_s[0])
            if pair_population.binaries.num > 0
            else None
        )
        boundary_s = step_end_s if pair_time_s is None else pair_time_s

        if boundary_s <= state.current_time_s:
            break

        boundary_bsi_event = None
        if (
            bsi_event is not None
            and (pair_time_s is None or bsi_event.time_s < pair_time_s)
            and np.isclose(boundary_s, bsi_event.time_s)
        ):
            boundary_bsi_event = bsi_event

        if boundary_bsi_event is not None:
            _prime_bsi_event(
                state.active_binaries[boundary_bsi_event.active_index],
                boundary_bsi_event,
            )

        _advance_active_binaries(
            state=state,
            records=records,
            Evolparams=Evolparams,
            model_config=model_config,
            agn=agn,
            disk=disk,
            rng_orbit=rng_orbit,
            boundary_s=boundary_s,
        )
        state.current_time_s = float(boundary_s)

        if boundary_bsi_event is not None:
            _record_event(
                state,
                kind="bsi",
                time_s=boundary_bsi_event.time_s,
                generation=boundary_bsi_event.generation,
                record_index=boundary_bsi_event.record_index,
                binary_idx=boundary_bsi_event.binary_idx,
                active_index=boundary_bsi_event.active_index,
            )
            continue

        if active_event_s is not None and np.isclose(boundary_s, active_event_s):
            continue

        if pair_time_s is not None and np.isclose(boundary_s, pair_time_s):
            _activate_pair_event(
                state=state,
                population=pair_population,
                record_index=next_record_index,
            )
            next_record_index += 1
            continue

        if next_arrival_s is not None and np.isclose(boundary_s, next_arrival_s):
            continue

        if pair_time_s is None and not state.active_binaries:
            break

    if not records and not state.active_binaries:
        records.append(
            EvolutionRecord(
                record_index=next_record_index,
                generation=0,
                population=empty_trap_population(initial_state),
                outcome=EvolutionResult(
                    evolution_timeline=EvolutionTimeline.empty(),
                    hierarchical_mergers=HierarchicalMergerLedger.empty(),
                ),
            )
        )

    return TrapEvolveRun(
        records=tuple(records),
        events=tuple(state.events),
        final_state=state,
    )
