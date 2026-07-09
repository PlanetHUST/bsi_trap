"""Compact simulation surface.

The run path is deliberately plain:

configuration -> disk -> embedded population -> trap arrivals -> pair-up timeline
-> BBH evolution
"""

from bsi_trap.simulation.models import (
    ArrivalTimeline,
    BinaryPopulation,
    DemoConfig,
    DemoResult,
    DiskProfile,
    EmbeddedPopulation,
    EvolutionTimeline,
    PairingTimeline,
)
from bsi_trap.simulation.run import run_simulation, validate_config


__all__ = [
    "ArrivalTimeline",
    "BinaryPopulation",
    "DemoConfig",
    "DemoResult",
    "DiskProfile",
    "EmbeddedPopulation",
    "EvolutionTimeline",
    "PairingTimeline",
    "run_simulation",
    "validate_config",
]
