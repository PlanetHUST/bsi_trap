import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

from bsi_trap.parallel_compute.json_io import write_json_atomically
from bsi_trap.parallel_compute.state import (
    SimulationConfig,
    SimulationResult,
    SimulationTask,
)
from bsi_trap.parallel_compute.single_work import process_single_sample_worker
from bsi_trap.storage.combined import combine_all_files
from bsi_trap.storage.contract import HDF5Contract

logger = logging.getLogger(__name__)


class ParallelSimulationManager:
    def __init__(
        self,
        config: SimulationConfig,
        smbh_mass_grid: Sequence[float],
        physics_callback: Callable,
    ) -> None:
        config.validate()

        if len(smbh_mass_grid) != config.num_samples:
            raise ValueError("SMBH mass grid length must equal config.num_samples.")

        self.config = config
        self.smbh_mass_grid = list(smbh_mass_grid)
        self.physics_callback = physics_callback

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)

    def build_tasks(self) -> list[SimulationTask]:
        tasks: list[SimulationTask] = []

        for model_index, model_config in enumerate(self.config.models):
            for sample_index, smbh_mass in enumerate(self.smbh_mass_grid):
                tasks.append(
                    SimulationTask(
                        model_config=model_config,
                        model_index=model_index,
                        sample_index=sample_index,
                        smbh_mass=float(smbh_mass),
                    )
                )

        return tasks

    def run(self) -> list[SimulationResult]:
        tasks = self.build_tasks()
        results: list[SimulationResult] = []

        logger.info("Starting %s simulation tasks", len(tasks))

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(
                    process_single_sample_worker,
                    task,
                    self.config.base_seed,
                    self.config.output_dir,
                    self.config.temp_dir,
                    self.physics_callback,
                )
                for task in tasks
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                logger.info(
                    "Finished model=%s sample=%s success=%s",
                    result.model_index,
                    result.sample_index,
                    result.success,
                )

        results.sort(key=lambda result: (result.model_index, result.sample_index))
        combined_files = combine_all_files(
            HDF5Contract(root_dir=self.config.output_dir)
        )
        self.write_summary(results, combined_files)
        return results

    def write_summary(
        self,
        results: list[SimulationResult],
        combined_files: list[Path],
    ) -> None:
        summary_path = self.config.output_dir / "simulation_summary.json"

        data = {
            "num_results": len(results),
            "num_success": sum(r.success for r in results),
            "num_failed": sum(not r.success for r in results),
            "combined_files": [str(path) for path in combined_files],
            "results": [asdict(r) for r in results],
        }

        write_json_atomically(summary_path, data)
        logger.info("Wrote summary to %s", summary_path)
