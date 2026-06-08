import logging
from pathlib import Path
from typing import Any, Callable, Sequence

from bsi_trap.parallel_compute import ParallelSimulationManager
from bsi_trap.parallel_compute.state import (
    SimulationConfig,
    SimulationResult,
)

logger = logging.getLogger(__name__)


def run_simulation_workflow(
    *,
    models: list[Any],
    smbh_mass_grid: Sequence[float],
    physics_callback: Callable,
    num_samples: int,
    max_workers: int,
    output_dir: Path,
    temp_dir: Path,
    base_seed: int = 42,
    timeout: float | None = None,
) -> list[SimulationResult]:
    """
    High-level simulation workflow.

    This function prepares the simulation configuration and delegates
    parallel execution to ParallelSimulationManager.

    It should not contain physics formulas.
    It should not contain multiprocessing internals.
    """

    logger.info("Preparing simulation workflow")

    config = SimulationConfig(
        num_samples=num_samples,
        max_workers=max_workers,
        output_dir=output_dir,
        temp_dir=temp_dir,
        base_seed=base_seed,
        timeout=timeout,
        models=models,
    )

    manager = ParallelSimulationManager(
        config=config,
        smbh_mass_grid=smbh_mass_grid,
        physics_callback=physics_callback,
    )

    results = manager.run()

    logger.info("Simulation workflow finished")

    return results
