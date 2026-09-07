"""BSI-trap compact demo package."""

from bsi_trap.demo import (
    demo_summary,
    run_demo,
)
from bsi_trap.dataoi import format_summary, result_to_dict, write_result_json
from bsi_trap.simulation import DemoConfig, DemoResult, SimulationResult

__all__ = [
    "DemoConfig",
    "DemoResult",
    "SimulationResult",
    "demo_summary",
    "format_summary",
    "result_to_dict",
    "run_demo",
    "write_result_json",
]
