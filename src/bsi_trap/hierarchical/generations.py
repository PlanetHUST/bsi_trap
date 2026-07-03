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
from bsi_trap.evolve.gen_evolve import GenerationEvolutionResult, make_generation
from bsi_trap.evolve.pairing import construct_pairing_up_timeline
from bsi_trap.evolve.timeline import FloatArray, IntArray
from bsi_trap.evolve.trap_context import build_trap_context
from bsi_trap.initialization.bbh_parameters import BBH_params_initialize
from bsi_trap.initialization.trap_population import (
    TrapPopulation,
    init_bbh,
    init_sbh,
)


@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    population: TrapPopulation
    outcome: GenerationEvolutionResult


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


def make_remnant_generation_population(
    *,
    remnant_singles: Single_BH_properties,
    generation: int,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    rng_pair_up: np.random.Generator,
    rng_orbit: np.random.Generator,
    rng_spin: np.random.Generator,
) -> TrapPopulation:
    """Build the next trap generation from retained merger remnants."""
    embedded_bh = Embedded_BH(remnant_singles.num)
    embedded_bh.init_ebh(
        remnant_singles.m_msun,
        remnant_singles.t_ini_s,
        np.full(remnant_singles.num, disk.r_trap_cm, dtype=np.float64),
    )
    embedded_bh.t_mig_s = np.zeros(remnant_singles.num, dtype=np.float64)
    embedded_bh.t_reach_s = np.asarray(remnant_singles.t_ini_s, dtype=np.float64)
    embedded_bh.r_trap_target_cm = np.full(
        remnant_singles.num,
        disk.r_trap_cm,
        dtype=np.float64,
    )
    embedded_bh.chi = np.asarray(remnant_singles.chi, dtype=np.float64)
    embedded_bh.theta = np.asarray(remnant_singles.theta, dtype=np.float64)

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
    )
    binaries = init_bbh(embedded_bh)
    singles = init_sbh(embedded_bh)

    if binaries.num > 0:
        BBH_params_initialize(
            agn=agn,
            disk=disk,
            bbh=binaries,
            rng_orbit=rng_orbit,
            rng_spin=rng_spin,
            model_config=model_config,
            preserve_existing_spins=True,
        )

    return TrapPopulation(
        embedded_bh=embedded_bh,
        binaries=binaries,
        singles=singles,
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
    max_generations: int,
) -> HierarchicalRun:
    if max_generations < 1:
        raise ValueError("max_generations must be at least 1")

    records: list[GenerationRecord] = []
    population = initial_population

    for generation in range(1, max_generations + 1):
        outcome = make_generation(
            Evolparams=Evolparams,
            model_config=model_config,
            agn=agn,
            disk=disk,
            bbh=population.binaries,
            sbh=population.singles,
            rng_orbit=rng_orbit,
            rng_spin=rng_spin,
        )
        records.append(
            GenerationRecord(
                generation=generation,
                population=population,
                outcome=outcome,
            )
        )

        if generation == max_generations:
            break

        remnant_singles = outcome.hierarchical_mergers.next_generation_singles
        if remnant_singles.num < 2:
            break

        next_population = make_remnant_generation_population(
            remnant_singles=remnant_singles,
            generation=generation + 1,
            model_config=model_config,
            agn=agn,
            disk=disk,
            rng_pair_up=rng_pair_up,
            rng_orbit=rng_orbit,
            rng_spin=rng_spin,
        )
        if next_population.binaries.num == 0:
            break

        population = next_population

    return HierarchicalRun(records=tuple(records))
