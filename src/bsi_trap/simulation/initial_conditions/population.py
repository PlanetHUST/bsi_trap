from __future__ import annotations

from dataclasses import dataclass

from bsi_trap.simulation.state.models import (
    BBH_properties,
    Embedded_BH,
    Single_BH_properties,
)
from bsi_trap.simulation.timeline.models import PairingTimeline


@dataclass(frozen=True)
class TrapArrivalState:
    """Trap-arrival reservoir before any runtime pairing is committed."""

    embedded_bh: Embedded_BH
    singles: Single_BH_properties


@dataclass(frozen=True)
class TrapPopulation:
    """One committed BBH population plus remaining live singles."""

    embedded_bh: Embedded_BH
    binaries: BBH_properties
    singles: Single_BH_properties
    pairing_timeline: PairingTimeline


def empty_trap_population(arrival_state: TrapArrivalState) -> TrapPopulation:
    """Package an arrival-only run for storage when no BBH ever forms."""
    return TrapPopulation(
        embedded_bh=arrival_state.embedded_bh,
        binaries=BBH_properties(0),
        singles=arrival_state.singles,
        pairing_timeline=PairingTimeline.empty(),
    )
