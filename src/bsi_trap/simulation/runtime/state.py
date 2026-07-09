from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing import Literal

from bsi_trap.simulation.state.models import (
    BBH_snapshot,
    SBH_original_snapshot,
    Remants_snapshot,
)


@dataclass(frozen=True)
class EventLog_binary:
    # kind: Literal["bsi", "isolate evolve", "merge", "pair"]
    current_time_s: float
    step_timeline: np.ndarray
    delta_a_cm_array: np.ndarray
    delta_e_array: np.ndarray


@dataclass
class StateLog:
    current_time_s: float
    single_snapshot: SBH_original_snapshot | None
    binary_snapshot: BBH_snapshot | None
    remant_snapshot: Remants_snapshot | None
