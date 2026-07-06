from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bsi_trap.classes.models import (
    AGN_properties,
    BBHEvolverParams,
    DiskProfile,
    Embedded_BH,
    Single_BH_properties,
)
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.constant.constant import Myr
from bsi_trap.evolve.gen_evolve import EvolutionResult, make_generation
from bsi_trap.evolve.trap_context import build_trap_context
from bsi_trap.initialization.bbh_parameters import BBH_params_initialize
from bsi_trap.initialization.trap_population import (
    TrapPopulation,
    init_bbh,
    init_sbh,
)
from bsi_trap.timeline.models import FloatArray, IntArray
from bsi_trap.timeline.pairing import construct_pairing_up_timeline


@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    population: TrapPopulation
    outcome: EvolutionResult


@dataclass(frozen=True)
class HierarchicalRun:
    records: tuple[GenerationRecord, ...]

    @property
    def generation_count(self) -> int:
        return len(self.records)

    @property
    def final_record(self) -> GenerationRecord:
        if not self.records:
            raise ValueError("HierarchicalRun has no generation records")
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
    def remnant_event_s(self) -> FloatArray:
        return _concat_float(
            record.outcome.hierarchical_mergers.merger_s for record in self.records
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
    """Return one event-time ordered single-BH reservoir."""
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
        combined.t_ini_s[offset:next_offset] = population.t_ini_s
        combined.chi[offset:next_offset] = population.chi
        combined.theta[offset:next_offset] = population.theta
        combined.r_pair_cm[offset:next_offset] = population.r_pair_cm
        offset = next_offset

    order = np.argsort(combined.t_ini_s, kind="stable")
    combined.m_msun = combined.m_msun[order]
    combined.t_ini_s = combined.t_ini_s[order]
    combined.chi = combined.chi[order]
    combined.theta = combined.theta[order]
    combined.r_pair_cm = combined.r_pair_cm[order]
    return combined


def _singles_from_embedded_arrivals(
    embedded_bh: Embedded_BH,
    end_time_s: float,
) -> Single_BH_properties:
    reached_mask = (
        np.isfinite(embedded_bh.t_reach_s)
        & (embedded_bh.t_reach_s >= 0.0)
        & (embedded_bh.t_reach_s < end_time_s)
        & np.isfinite(embedded_bh.m_msun)
        & (embedded_bh.m_msun > 0.0)
    )
    reached_idx = np.flatnonzero(reached_mask)
    reached_idx = reached_idx[
        np.argsort(embedded_bh.t_reach_s[reached_idx], kind="stable")
    ]

    singles = Single_BH_properties(reached_idx.size)
    if reached_idx.size == 0:
        return singles

    singles.m_msun = np.array(embedded_bh.m_msun[reached_idx], copy=True)
    singles.t_ini_s = np.array(embedded_bh.t_reach_s[reached_idx], copy=True)
    singles.chi = np.array(embedded_bh.chi[reached_idx], copy=True)
    singles.theta = np.array(embedded_bh.theta[reached_idx], copy=True)
    singles.r_pair_cm = np.array(embedded_bh.r_pair_cm[reached_idx], copy=True)
    return singles


def _build_embedded_reservoir(
    source_singles: Single_BH_properties,
    disk: DiskProfile,
) -> Embedded_BH:
    embedded_bh = Embedded_BH(source_singles.num)
    embedded_bh.init_ebh(
        source_singles.m_msun,
        source_singles.t_ini_s,
        np.full(source_singles.num, disk.r_trap_cm, dtype=np.float64),
    )
    embedded_bh.t_mig_s = np.zeros(source_singles.num, dtype=np.float64)
    embedded_bh.t_reach_s = np.asarray(source_singles.t_ini_s, dtype=np.float64)
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


def _build_population_from_single_reservoir(
    *,
    source_singles: Single_BH_properties,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    rng_pair_up: np.random.Generator,
    rng_orbit: np.random.Generator,
    rng_spin: np.random.Generator,
    max_pairing_events: int | None,
) -> TrapPopulation:

    embedded_bh = _build_embedded_reservoir(source_singles, disk)
    end_time_s = agn.lifetime_myr * Myr
    reached_mask = np.isfinite(embedded_bh.t_reach_s) & (
        embedded_bh.t_reach_s < end_time_s
    )

    # build geometry trap context
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
        max_events=max_pairing_events,
    )

    binaries_ite = init_bbh(embedded_bh)
    singles_ite = init_sbh(embedded_bh)

    if binaries_ite.num > 0:
        BBH_params_initialize(
            agn=agn,
            disk=disk,
            bbh=binaries_ite,
            rng_orbit=rng_orbit,
            rng_spin=rng_spin,
            model_config=model_config,
            preserve_existing_spins=True,
        )

    return TrapPopulation(
        embedded_bh=embedded_bh,
        binaries=binaries_ite,
        singles=singles_ite,
        pairing_timeline=pairing_timeline,
    )


def evolve_hierarchical_generations(
    *,
    Evolparams: BBHEvolverParams,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    initial_population: TrapPopulation,
    rng_pair_up: np.random.Generator,
    rng_orbit: np.random.Generator,
    rng_spin: np.random.Generator,
) -> HierarchicalRun:

    end_time_s = agn.lifetime_myr * Myr

    live_singles = _singles_from_embedded_arrivals(
        initial_population.embedded_bh,
        end_time_s,
    )

    # list of record -> records
    records: list[GenerationRecord] = []
    event_index = 1

    while True:
        population = _build_population_from_single_reservoir(
            source_singles=live_singles,
            model_config=model_config,
            agn=agn,
            disk=disk,
            rng_pair_up=rng_pair_up,
            rng_orbit=rng_orbit,
            rng_spin=rng_spin,
            max_pairing_events=1,
        )

        outcome = make_generation(
            Evolparams=Evolparams,
            model_config=model_config,
            agn=agn,
            disk=disk,
            bbh=population.binaries,
            sbh=population.singles,
            rng_orbit=rng_orbit,
        )

        record = GenerationRecord(
            generation=event_index,
            population=population,
            outcome=outcome,
        )

        # break condition 1: no active binaries
        if population.binaries.num == 0:
            if not records:
                records.append(record)
            break

        records.append(record)

        live_singles = _combine_single_populations(
            population.singles,
            outcome.hierarchical_mergers.next_generation_singles,
        )

        # break condition 2: no more than 2 active single bhs
        if live_singles.num < 2:
            break

        # iterating condition: next event
        event_index += 1

    return HierarchicalRun(records=tuple(records))
