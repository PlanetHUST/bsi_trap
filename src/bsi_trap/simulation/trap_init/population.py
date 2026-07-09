from __future__ import annotations

from dataclasses import dataclass

from bsi_trap.simulation.state.models import (
    BBH_properties,
    Embedded_BH,
    Single_BH_properties,
)
from bsi_trap.simulation.timeline.models import PairingTimeline
from bsi_trap.simulation.runtime.state import StateLog
from bsi_trap.simulation.state.models import (
    Arrive_properties,
    BBH_snapshot,
    SBH_original_snapshot,
    Remants_snapshot,
)


@dataclass(frozen=True)
class TrapPopulation:
    """One committed BBH population plus remaining live singles."""

    embedded_bh: Embedded_BH
    binaries: BBH_properties
    singles: Single_BH_properties
    pairing_timeline: PairingTimeline


def construct_state_log(
    current_time_s: float,
    *,
    bbh_snapshot: BBH_snapshot | None = None,
    sbh_snapshot: SBH_original_snapshot | None = None,
    rem_snapshot: Remants_snapshot | None = None,
) -> StateLog:

    if bbh_snapshot == None or sbh_snapshot == None or rem_snapshot == None:
        print("initialize statelog")
        current_time_s = 0

    evolve_state = StateLog(current_time_s, sbh_snapshot, bbh_snapshot, rem_snapshot)

    return evolve_state
