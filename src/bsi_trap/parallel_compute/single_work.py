import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from bsi_trap.parallel_compute.json_io import write_json_atomically
from bsi_trap.parallel_compute.state import SimulationResult, SimulationTask
from bsi_trap.parallel_compute.rand_num_generator import make_sample_seed

logger = logging.getLogger(__name__)


def process_single_sample_worker(
    task: SimulationTask,
    base_seed: int,
    output_dir: Path,
    temp_dir: Path,
    physics_callback: Callable,
) -> SimulationResult:
    start = time.perf_counter()
    metadata_file = temp_dir / (
        f"model_{task.model_index}_sample_{task.sample_index}.json"
    )

    try:
        seed = make_sample_seed(base_seed, task.model_index, task.sample_index)

        output_file = output_dir / (
            f"model_{task.model_index}_sample_{task.sample_index}.h5"
        )

        result = physics_callback(
            model_config=task.model_config,
            sample_index=task.sample_index,
            smbh_mass=task.smbh_mass,
            seed=seed,
            output_file=output_file,
        )

        elapsed = time.perf_counter() - start
        result_output_file = getattr(result, "output_file", output_file)
        generation_files = getattr(result, "generation_files", None)
        manifest_file = getattr(result, "manifest_file", None)

        simulation_result = SimulationResult(
            success=True,
            model_index=task.model_index,
            sample_index=task.sample_index,
            smbh_mass=task.smbh_mass,
            output_file=str(result_output_file),
            generation_files=tuple(str(path) for path in generation_files)
            if generation_files
            else None,
            manifest_file=str(manifest_file) if manifest_file else None,
            metadata_file=str(metadata_file),
            execution_time=elapsed,
            merger_count=getattr(result, "merger_count", None),
            evolved_binary_count=getattr(result, "evolved_binary_count", None),
            generation_count=getattr(result, "generation_count", None),
        )
        write_json_atomically(metadata_file, asdict(simulation_result))
        return simulation_result

    except Exception as exc:
        logger.exception(
            "Simulation failed: model=%s sample=%s",
            task.model_index,
            task.sample_index,
        )

        simulation_result = SimulationResult(
            success=False,
            model_index=task.model_index,
            sample_index=task.sample_index,
            smbh_mass=task.smbh_mass,
            metadata_file=str(metadata_file),
            execution_time=time.perf_counter() - start,
            error=str(exc),
        )
        write_json_atomically(metadata_file, asdict(simulation_result))
        return simulation_result
