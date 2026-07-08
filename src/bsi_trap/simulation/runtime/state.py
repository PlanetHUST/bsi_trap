from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bsi_trap.simulation.state.models import Embedded_BH, Single_BH_properties
from bsi_trap.simulation.evolution.generation import EvolutionResult
from bsi_trap.simulation.initial_conditions.population import TrapPopulation
from bsi_trap.simulation.timeline.models import FloatArray, IntArray


@dataclass
class ActiveBinary:
    """One committed BBH currently owned by the runtime clock."""

    record_index: int
    generation: int
    population: TrapPopulation


@dataclass(frozen=True)
class EvolutionRecord:
    """Finalized BBH event record ready for storage."""

    record_index: int
    generation: int
    population: TrapPopulation
    outcome: EvolutionResult


@dataclass(frozen=True)
class RuntimeEvent:
    kind: str
    time_s: float
    generation: int = -1
    record_index: int = -1
    live_single_count: int = -1
    active_binary_count: int = -1
    binary_idx: int = -1
    active_index: int = -1


@dataclass
class StateLog:
    current_time_s: float
    live_singles: Single_BH_properties
    embedded_bh: Embedded_BH
    active_binaries: list[ActiveBinary] = field(default_factory=list)
    events: list[RuntimeEvent] = field(default_factory=list)


@dataclass(frozen=True)
class TrapEvolveRun:
    records: tuple[EvolutionRecord, ...]
    events: tuple[RuntimeEvent, ...]
    final_state: StateLog

    @property
    def generation_count(self) -> int:
        return len(self.records)

    @property
    def final_record(self) -> EvolutionRecord:
        if not self.records:
            raise ValueError("TrapEvolveRun has no finalized records")
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
    def bsi_event_s(self) -> FloatArray:
        return np.asarray(
            [event.time_s for event in self.events if event.kind == "bsi"],
            dtype=np.float64,
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


def record_event(
    state: StateLog,
    *,
    kind: str,
    time_s: float,
    generation: int,
    record_index: int,
    binary_idx: int = -1,
    active_index: int = -1,
) -> None:
    state.events.append(
        RuntimeEvent(
            kind=kind,
            time_s=float(time_s),
            generation=int(generation),
            record_index=int(record_index),
            live_single_count=int(state.live_singles.num),
            active_binary_count=len(state.active_binaries),
            binary_idx=int(binary_idx),
            active_index=int(active_index),
        )
    )
