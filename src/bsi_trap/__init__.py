"""BSI-trap compact demo package."""

from bsi_trap.demo import (
    demo_summary,
    run_demo,
)
from bsi_trap.dataoi import format_summary
from bsi_trap.simulation import DemoConfig, DemoResult

__all__ = [
    "DemoConfig",
    "DemoResult",
    "demo_summary",
    "format_summary",
    "run_demo",
]
