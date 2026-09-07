"""Compact simulation surface.

The run path is deliberately plain:

configuration -> disk -> embedded population -> trap arrivals -> pair-up timeline
-> BBH evolution
"""

from bsi_trap.simulation.models import (
    ArrivalTimeline,
    BSIRecord,
    BinaryPopulation,
    BinaryState,
    DemoConfig,
    DemoResult,
    DiskProfile,
    EmbeddedPopulation,
    MergerRecord,
    PairingRecord,
    SimulationResult,
    SimulationState,
)
from bsi_trap.simulation.run import run_simulation, validate_config


__all__ = [
    "ArrivalTimeline",
    "BSIRecord",
    "BinaryPopulation",
    "BinaryState",
    "DemoConfig",
    "DemoResult",
    "DiskProfile",
    "EmbeddedPopulation",
    "MergerRecord",
    "PairingRecord",
    "SimulationResult",
    "SimulationState",
    "run_simulation",
    "validate_config",
]
