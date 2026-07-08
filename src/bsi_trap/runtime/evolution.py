from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from bsi_trap.classes.models import (
    AGN_properties,
    BBHEvolverParams,
    DiskProfile,
    Single_BH_properties,
)
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.constant.constant import Myr
from bsi_trap.evolve.bsi import schedule_bsi_events
from bsi_trap.evolve.gen_evolve import EvolutionResult, evolve_single_binary
from bsi_trap.evolve.tracks import finalize_bbh_tracks
from bsi_trap.evolve.trap_context import build_trap_context
from bsi_trap.runtime.remnants import (
    HierarchicalMergerLedger,
    build_hierarchical_mergers,
)
from bsi_trap.initialization.bbh_parameters import BBH_params_initialize
from bsi_trap.initialization.trap_population import (
    TrapPopulation,
    init_bbh,
    init_sbh,
)
from bsi_trap.timeline.bsi import construct_bsi_timeline
from bsi_trap.timeline.evolution import construct_bbh_evolution_timeline
from bsi_trap.timeline.models import EvolutionTimeline, FloatArray, IntArray
from bsi_trap.timeline.pairing import construct_pairing_up_timeline


@dataclass
class ActiveBinary:
    """One committed BBH currently owned by the runtime clock."""

    record_index: int
    generation: int
    population: TrapPopulation


@dataclass(frozen=True)
class TrapEvolutionRecord:
    """Finalized BBH event record ready for storage."""

    record_index: int
    generation: int
    population: TrapPopulation
    outcome: EvolutionResult


@dataclass(frozen=True)
class TrapRuntimeEvent:
    kind: str
    time_s: float
    generation: int
    record_index: int
    live_single_count: int
    active_binary_count: int


@dataclass
class TrapRuntimeState:
    current_time_s: float
    live_singles: Single_BH_properties
    active_binaries: list[ActiveBinary] = field(default_factory=list)
    events: list[TrapRuntimeEvent] = field(default_factory=list)


@dataclass(frozen=True)
class TrapRuntimeRun:
    records: tuple[TrapEvolutionRecord, ...]
    events: tuple[TrapRuntimeEvent, ...]
    final_state: TrapRuntimeState

    @property
    def generation_count(self) -> int:
        return len(self.records)

    @property
    def final_record(self) -> TrapEvolutionRecord:
        if not self.records:
            raise ValueError("TrapRuntimeRun has no finalized records")
        return self.records[-1]

    @property
    def evolved_binary_count(self) -> int:
        return int(
            sum(record.outcome.evolution_timeline.size for record in self.records)
        )

    @property
    def merger_count(self) -> int:
        return int(
            sum(
                np.count_nonzero(record.population.binaries.flags.merger_flag)
                for record in self.records
            )
        )

    @property
    def pairing_event_s(self) -> FloatArray:
        return _concat_float(
            record.population.pairing_timeline.formation_s for record in self.records
        )

    @property
    def merger_event_s(self) -> FloatArray:
        return _concat_float(
            record.outcome.evolution_timeline.bbh_merge_s for record in self.records
        )

    @property
    def merger_generation(self) -> IntArray:
        return _concat_int(
            np.full(
                record.outcome.evolution_timeline.bbh_merge_s.size,
                record.generation,
                dtype=np.int64,
            )
            for record in self.records
        )


def _concat_float(arrays) -> FloatArray:
    items = [np.asarray(array, dtype=np.float64) for array in arrays if array.size > 0]
    if not items:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(items)


def _concat_int(arrays) -> IntArray:
    items = [np.asarray(array, dtype=np.int64) for array in arrays if array.size > 0]
    if not items:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(items)


def _combine_single_populations(
    *populations: Single_BH_properties | None,
) -> Single_BH_properties:
    items = [
        population
        for population in populations
        if population is not None and population.num > 0
    ]
    total = sum(population.num for population in items)
    combined = Single_BH_properties(total)
    if total == 0:
        return combined

    offset = 0
    for population in items:
        next_offset = offset + population.num
        combined.m_msun[offset:next_offset] = population.m_msun
        combined.available_from_s[offset:next_offset] = population.available_from_s
        combined.chi[offset:next_offset] = population.chi
        combined.theta[offset:next_offset] = population.theta
        combined.r_pair_cm[offset:next_offset] = population.r_pair_cm
        combined.generation[offset:next_offset] = population.generation
        offset = next_offset

    order = np.argsort(combined.available_from_s, kind="stable")
    combined.m_msun = combined.m_msun[order]
    combined.available_from_s = combined.available_from_s[order]
    combined.chi = combined.chi[order]
    combined.theta = combined.theta[order]
    combined.r_pair_cm = combined.r_pair_cm[order]
    combined.generation = combined.generation[order]
    return combined


def _empty_outcome() -> EvolutionResult:
    return EvolutionResult(
        evolution_timeline=EvolutionTimeline.empty(),
        hierarchical_mergers=HierarchicalMergerLedger.empty(),
    )


def _record_event(
    state: TrapRuntimeState,
    *,
    kind: str,
    time_s: float,
    generation: int,
    record_index: int,
) -> None:
    state.events.append(
        TrapRuntimeEvent(
            kind=kind,
            time_s=float(time_s),
            generation=int(generation),
            record_index=int(record_index),
            live_single_count=int(state.live_singles.num),
            active_binary_count=len(state.active_binaries),
        )
    )


def _build_embedded_reservoir(source_singles: Single_BH_properties, disk: DiskProfile):
    from bsi_trap.classes.models import Embedded_BH

    embedded_bh = Embedded_BH(source_singles.num)
    embedded_bh.init_ebh(
        source_singles.m_msun,
        source_singles.available_from_s,
        np.full(source_singles.num, disk.r_trap_cm, dtype=np.float64),
    )
    embedded_bh.t_mig_s = np.zeros(source_singles.num, dtype=np.float64)
    embedded_bh.t_reach_s = np.asarray(
        source_singles.available_from_s,
        dtype=np.float64,
    )
    embedded_bh.r_trap_target_cm = np.full(
        source_singles.num,
        disk.r_trap_cm,
        dtype=np.float64,
    )
    embedded_bh.r_pair_cm = np.array(
        source_singles.r_pair_cm,
        dtype=np.float64,
        copy=True,
    )
    embedded_bh.chi = np.array(source_singles.chi, dtype=np.float64, copy=True)
    embedded_bh.theta = np.array(source_singles.theta, dtype=np.float64, copy=True)
    return embedded_bh


def _tag_pair_generations(
    population: TrapPopulation,
    source_singles: Single_BH_properties,
) -> int:
    bbh = population.binaries
    if bbh.num == 0:
        return 0

    pair_generations: list[int] = []
    for first, second in enumerate(population.embedded_bh.trap_flag):
        if second >= 0 and first < second:
            pair_generations.append(
                int(
                    max(
                        source_singles.generation[first],
                        source_singles.generation[int(second)],
                    )
                    + 1
                )
            )

    bbh.generation = np.asarray(pair_generations, dtype=np.int64)

    single_idx = np.flatnonzero(population.embedded_bh.trap_flag == -2)
    population.singles.generation = np.asarray(
        source_singles.generation[single_idx],
        dtype=np.int64,
    )
    return int(np.max(bbh.generation))


def _build_pair_event_population(
    *,
    source_singles: Single_BH_properties,
    current_time_s: float,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    rng_pair_up: np.random.Generator,
    rng_orbit: np.random.Generator,
    rng_spin: np.random.Generator,
) -> TrapPopulation:
    embedded_bh = _build_embedded_reservoir(source_singles, disk)
    end_time_s = agn.lifetime_myr * Myr
    reached_mask = np.isfinite(embedded_bh.t_reach_s) & (
        embedded_bh.t_reach_s < end_time_s
    )
    trap_context = build_trap_context(disk, model_config)
    embedded_bh.trap_width_cm = trap_context.width_cm

    pairing_timeline = construct_pairing_up_timeline(
        embedded_bh=embedded_bh,
        reached_mask=reached_mask,
        trap_context=trap_context,
        model_config=model_config,
        disk=disk,
        smbh_mass_msun=agn.M_SMBH_msun,
        end_time_s=end_time_s,
        rng_pair_up=rng_pair_up,
        max_events=1,
        start_time_s=current_time_s,
    )

    bbh = init_bbh(embedded_bh)
    singles = init_sbh(embedded_bh)
    population = TrapPopulation(
        embedded_bh=embedded_bh,
        binaries=bbh,
        singles=singles,
        pairing_timeline=pairing_timeline,
    )
    _tag_pair_generations(population, source_singles)

    if bbh.num > 0:
        BBH_params_initialize(
            agn=agn,
            disk=disk,
            bbh=bbh,
            rng_orbit=rng_orbit,
            rng_spin=rng_spin,
            model_config=model_config,
            preserve_existing_spins=True,
        )

    return population


def sample_next_pair_event(
    *,
    state: TrapRuntimeState,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    rng_pair_up: np.random.Generator,
    rng_orbit: np.random.Generator,
    rng_spin: np.random.Generator,
) -> TrapPopulation:
    """Return the next committed-pair candidate from the live singles."""
    return _build_pair_event_population(
        source_singles=state.live_singles,
        current_time_s=state.current_time_s,
        model_config=model_config,
        agn=agn,
        disk=disk,
        rng_pair_up=rng_pair_up,
        rng_orbit=rng_orbit,
        rng_spin=rng_spin,
    )


def _binary_is_final(population: TrapPopulation) -> bool:
    bbh = population.binaries
    if bbh.num == 0:
        return True
    return bool(
        np.any(bbh.flags.merger_flag != 0)
        or np.any(bbh.flags.stop_reason_flag != 0)
    )


def advance_bbh_segment(
    *,
    Evolparams: BBHEvolverParams,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    active: ActiveBinary,
    boundary_s: float,
) -> float:
    """Advance an active BBH to a global boundary or to its physical stop."""
    bbh = active.population.binaries
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


def _predict_active_event_s(
    *,
    Evolparams: BBHEvolverParams,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    active_binaries: list[ActiveBinary],
    boundary_s: float,
) -> float | None:
    event_times: list[float] = []
    for active in active_binaries:
        prediction = ActiveBinary(
            record_index=active.record_index,
            generation=active.generation,
            population=deepcopy(active.population),
        )
        event_s = advance_bbh_segment(
            Evolparams=Evolparams,
            model_config=model_config,
            agn=agn,
            disk=disk,
            active=prediction,
            boundary_s=boundary_s,
        )
        if _binary_is_final(prediction.population) and event_s <= boundary_s:
            event_times.append(float(event_s))

    if not event_times:
        return None
    return min(event_times)


def _finalize_active_binary(
    *,
    agn: AGN_properties,
    disk: DiskProfile,
    active: ActiveBinary,
    rng_orbit: np.random.Generator,
) -> TrapEvolutionRecord:
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
    return TrapEvolutionRecord(
        record_index=active.record_index,
        generation=active.generation,
        population=active.population,
        outcome=EvolutionResult(
            evolution_timeline=evolution_timeline,
            hierarchical_mergers=hierarchical_mergers,
        ),
    )


def _advance_active_binaries(
    *,
    state: TrapRuntimeState,
    records: list[TrapEvolutionRecord],
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
            active=active,
            boundary_s=boundary_s,
        )
        if _binary_is_final(active.population) and event_s <= boundary_s:
            record = _finalize_active_binary(
                agn=agn,
                disk=disk,
                active=active,
                rng_orbit=rng_orbit,
            )
            records.append(record)
            state.live_singles = _combine_single_populations(
                state.live_singles,
                record.outcome.hierarchical_mergers.next_generation_singles,
            )
            event_kind = (
                "merge"
                if np.any(active.population.binaries.flags.merger_flag != 0)
                else "stop"
            )
            _record_event(
                state,
                kind=event_kind,
                time_s=event_s,
                generation=active.generation,
                record_index=active.record_index,
            )
            continue
        remaining.append(active)
    state.active_binaries = remaining


def _activate_pair_event(
    *,
    state: TrapRuntimeState,
    population: TrapPopulation,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    rng_orbit: np.random.Generator,
    record_index: int,
) -> None:
    bbh = population.binaries
    if bbh.num == 0:
        return

    if model_config.bsi_enabled:
        schedule_bsi_events(
            model_config=model_config,
            agn=agn,
            disk=disk,
            bbh=bbh,
            sbh=population.singles,
            rng=rng_orbit,
        )

    generation = int(bbh.generation[0])
    state.live_singles = population.singles
    active = ActiveBinary(
        record_index=record_index,
        generation=generation,
        population=population,
    )
    state.active_binaries.append(active)
    _record_event(
        state,
        kind="pair",
        time_s=float(bbh.formation_s[0]),
        generation=generation,
        record_index=record_index,
    )


def _empty_record(
    *,
    record_index: int,
    population: TrapPopulation,
) -> TrapEvolutionRecord:
    return TrapEvolutionRecord(
        record_index=record_index,
        generation=0,
        population=population,
        outcome=_empty_outcome(),
    )


def evolve_trap_runtime(
    *,
    Evolparams: BBHEvolverParams,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    initial_population: TrapPopulation,
    rng_pair_up: np.random.Generator,
    rng_orbit: np.random.Generator,
    rng_spin: np.random.Generator,
) -> TrapRuntimeRun:
    """Evolve the trap population with one top-level physical event clock."""
    if initial_population.binaries.num != 0:
        raise ValueError("Runtime evolution expects trap-arrival singles only")

    end_time_s = float(agn.lifetime_myr * Myr)
    state = TrapRuntimeState(
        current_time_s=0.0,
        live_singles=_combine_single_populations(initial_population.singles),
    )
    records: list[TrapEvolutionRecord] = []
    next_record_index = 1

    while state.current_time_s < end_time_s:
        pair_population = sample_next_pair_event(
            state=state,
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
        boundary_s = end_time_s if pair_time_s is None else pair_time_s
        active_event_s = _predict_active_event_s(
            Evolparams=Evolparams,
            model_config=model_config,
            agn=agn,
            disk=disk,
            active_binaries=state.active_binaries,
            boundary_s=boundary_s,
        )
        if active_event_s is not None and active_event_s < boundary_s:
            boundary_s = active_event_s

        if boundary_s <= state.current_time_s:
            break

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

        if active_event_s is not None and np.isclose(boundary_s, active_event_s):
            continue

        if pair_time_s is not None and np.isclose(boundary_s, pair_time_s):
            _activate_pair_event(
                state=state,
                population=pair_population,
                model_config=model_config,
                agn=agn,
                disk=disk,
                rng_orbit=rng_orbit,
                record_index=next_record_index,
            )
            next_record_index += 1
            continue

        if pair_time_s is None and not state.active_binaries:
            break

    if not records and not state.active_binaries:
        records.append(
            _empty_record(
                record_index=next_record_index,
                population=initial_population,
            )
        )

    return TrapRuntimeRun(
        records=tuple(records),
        events=tuple(state.events),
        final_state=state,
    )
