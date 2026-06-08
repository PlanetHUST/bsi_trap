import logging
import time
from pathlib import Path
from typing import Callable

from bsi_trap.parallel_compute.state import SimulationResult, SimulationTask
from bsi_trap.parallel_compute.rand_num_generator import make_sample_seed

logger = logging.getLogger(__name__)


def process_single_sample_worker(
    task: SimulationTask,
    base_seed: int,
    output_dir: Path,
    physics_callback: Callable,
) -> SimulationResult:
    start = time.perf_counter()

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

        return SimulationResult(
            success=True,
            model_index=task.model_index,
            sample_index=task.sample_index,
            smbh_mass=task.smbh_mass,
            output_file=str(result_output_file),
            execution_time=elapsed,
            merger_count=getattr(result, "merger_count", None),
            generation_count=getattr(result, "generation_count", None),
        )

    except Exception as exc:
        logger.exception(
            "Simulation failed: model=%s sample=%s",
            task.model_index,
            task.sample_index,
        )

        return SimulationResult(
            success=False,
            model_index=task.model_index,
            sample_index=task.sample_index,
            smbh_mass=task.smbh_mass,
            execution_time=time.perf_counter() - start,
            error=str(exc),
        )
