from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from bsi_trap.simulation.state.models import (
    BBHEvolverParams,
    DiskProfile,
)

from bsi_trap.simulation.config.model_config import ModelConfig
from bsi_trap.simulation.initial_conditions.environment import make_environment
from bsi_trap.simulation.initial_conditions.trap_population import (
    make_trap_population,
)
from bsi_trap.simulation.runtime.evolution import evolve_trap_runtime
from bsi_trap.simulation.config.rng import resolve_seed, make_rng_streams
from bsi_trap.storage.contract import HDF5Contract
from bsi_trap.storage.payload import (
    GenerationState,
    generation_data,
    write_sample_manifest,
)
from bsi_trap.storage.save_gen import save_generation_to_file

TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class SampleWorkflowResult:
    output_file: str
    merger_count: int = 0
    evolved_binary_count: int = 0
    generation_count: int = 1
    manifest_file: str | None = None
    generation_files: tuple[str, ...] = ()


def _progress_enabled() -> bool:
    return os.getenv("AGN_PROGRESS", "1").lower() in TRUE_VALUES


def _progress(sample_index: int, model_name: str, message: str) -> None:
    if not _progress_enabled():
        return
    print(
        f"[PID:{os.getpid()}][sample {sample_index:04d}][{model_name}] {message}",
        flush=True,
    )


def _evolver_params(disk: DiskProfile) -> BBHEvolverParams:
    return BBHEvolverParams(
        rlist=disk.radius_cm,
        sigmalist=disk.sigma_cgs,
        hlist=disk.h_over_r,
        cslist=disk.cs_cgs,
    )


def run_physics_sample(
    model_config: ModelConfig,
    sample_index: int,
    smbh_mass: float,
    output_file: Path,
    seed: int | None = None,
) -> SampleWorkflowResult:

    # 1. Resolve the sample-local random streams.
    # Each physical subsystem gets its own stream so changing one stochastic
    # step does not silently reshuffle all later draws.
    master_seed = resolve_seed(seed if seed is not None else model_config.master_seed)
    sample_seed = master_seed if seed is not None else master_seed + sample_index
    rng = make_rng_streams(sample_seed)
    _progress(
        sample_index,
        model_config.simulation_type,
        f"start: Msmbh={smbh_mass:.3e} Msun",
    )

    # 2. Build the AGN environment for this SMBH mass.
    # This step owns the pAGN disk profile, torque profile, and reference trap
    # geometry used by later migration and pairing kernels.
    agn, disk = make_environment(
        model_config=model_config,
        smbh_mass=smbh_mass,
    )
    _progress(
        sample_index,
        model_config.simulation_type,
        f"disk: rtrap={disk.r_trap_cm:.3e} cm",
    )

    # 3. Initialize the trap-arrival population.
    # Reaching the trap only creates live single BHs here; BBH formation is not
    # precomputed at initialization.
    population = make_trap_population(
        model_config=model_config,
        agn=agn,
        disk=disk,
        rng_nsc=rng.nsc,
    )
    _progress(
        sample_index,
        model_config.simulation_type,
        "trap arrival population: "
        f"embedded={population.embedded_bh.num}, "
        f"BBH={population.binaries.num}, "
        f"single={population.singles.num}",
    )

    # 4. Run the unified trap event clock.
    # The runtime owns live singles, active BBHs, current time, and recycled
    # merger remnants. Pairing and weak BSI are sampled one time window at a
    # time, then committed BBHs are advanced only to the next global boundary.
    runtime_run = evolve_trap_runtime(
        Evolparams=_evolver_params(disk),
        model_config=model_config,
        agn=agn,
        disk=disk,
        initial_population=population,
        rng_pair_up=rng.pair_up,
        rng_orbit=rng.dynamics,
        rng_spin=rng.spin,
    )

    # Finalized records are the storage units: each record contains the
    # committed pair population, its BBH evolution timeline, and remnant ledger.
    final_record = runtime_run.final_record
    evolved_binary_count = runtime_run.evolved_binary_count
    merger_count = runtime_run.merger_count
    generation_count = runtime_run.generation_count

    _progress(
        sample_index,
        model_config.simulation_type,
        "evolved trap events: "
        f"count={generation_count}, binaries={evolved_binary_count}, "
        f"mergers={merger_count}",
    )
    _progress(
        sample_index,
        model_config.simulation_type,
        "final merger remnants: "
        f"retained={final_record.outcome.hierarchical_mergers.retained_count}, "
        f"escaped={final_record.outcome.hierarchical_mergers.escaped_count}, "
        f"recycled={final_record.outcome.hierarchical_mergers.recycled_count}",
    )

    # 5. Persist every finalized runtime record.
    # A physical sample can produce multiple records when retained remnants
    # re-enter the live single reservoir and pair again.
    output_files: list[Path] = []
    contract = HDF5Contract(root_dir=output_file.parent)
    final_state: GenerationState | None = None
    for record in runtime_run.records:
        state = GenerationState(
            agn=agn,
            disk=disk,
            population=record.population,
            evolution_timeline=record.outcome.evolution_timeline,
            hierarchical_mergers=record.outcome.hierarchical_mergers,
            generation_index=record.generation,
        )
        saved_path = save_generation_to_file(
            simulation_type=model_config.simulation_type,
            sample_index=sample_index,
            generation=record.record_index,
            generation_data=generation_data(
                model_config=model_config,
                state=state,
            ),
            contract=contract,
        )
        output_files.append(saved_path)
        final_state = state

    if final_state is None:
        raise RuntimeError("No generation state was produced")

    output_file = output_files[-1]

    # 6. Write the per-sample manifest after all generation files exist.
    # The manifest points analysis/plotting code at the final state and the
    # complete list of per-record HDF5 outputs for this SMBH sample.
    manifest_file = write_sample_manifest(
        model_config=model_config,
        state=final_state,
        sample_index=sample_index,
        smbh_mass=smbh_mass,
        output_file=output_file,
        merger_count=merger_count,
        evolved_binary_count=evolved_binary_count,
        generation_count=generation_count,
        generation_files=output_files,
    )
    _progress(
        sample_index,
        model_config.simulation_type,
        f"saved generations: {len(output_files)} files, final={output_file}",
    )

    return SampleWorkflowResult(
        output_file=str(output_file),
        merger_count=merger_count,
        evolved_binary_count=evolved_binary_count,
        generation_count=generation_count,
        manifest_file=str(manifest_file),
        generation_files=tuple(str(path) for path in output_files),
    )
