"""Command-line entry point for the AGN BBH model-channel pipeline."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from experiment_grid import load_experiment_grid
from model_config import (
    BSI_MODEL_NAME,
    NO_DAMP_COPY_MODEL_NAME,
    ModelConfig,
    legacy_model_config,
)
from parallel_manager import (
    ParallelSimulationManager,
    SimulationConfig,
    setup_parallel_environment,
)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"

DEFAULT_TYPES = (
    "with_gas_damp",
    NO_DAMP_COPY_MODEL_NAME,
    "with_gas_lin",
    "with_gas_log",
    BSI_MODEL_NAME,
)
ALL_TYPES = (
    "with_gas_damp",
    NO_DAMP_COPY_MODEL_NAME,
    "without_gas_damp",
    "with_gas_lin",
    "without_gas_lin",
    "with_gas_log",
    "without_gas_log",
    BSI_MODEL_NAME,
)


def runtime_defaults() -> tuple[int, int, bool]:
    """Return sample/worker defaults and whether this looks like a SLURM job."""
    is_hpc = "SLURM_JOB_ID" in os.environ
    if is_hpc:
        workers = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        return 100, workers, True
    return 10, max(1, (os.cpu_count() or 4) - 2), False


def build_parser(default_samples: int, default_workers: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AGN BBH simulations, combine HDF5 output, and optionally plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    run = parser.add_argument_group("Run size")
    run.add_argument("--samples", type=int, default=default_samples)
    run.add_argument("--workers", type=int, default=default_workers)
    run.add_argument("--timeout", type=float, default=3600.0)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--sample-start", type=int, default=0)
    run.add_argument("--sample-stop", type=int, default=None)
    run.add_argument(
        "--test", action="store_true", help="Use 4 samples and <=4 workers"
    )

    models = parser.add_argument_group("Model selection")
    models.add_argument(
        "--experiment-grid", help="YAML model grid, e.g. configs/baseline.yaml"
    )
    models.add_argument(
        "--model-name", action="append", help="Run one named model; repeatable"
    )
    models.add_argument("--list-models", action="store_true")
    models.add_argument("--types", nargs="+", default=list(DEFAULT_TYPES))
    models.add_argument(
        "--all-types", action="store_true", help="Use all built-in gas/no-gas types"
    )

    output = parser.add_argument_group("Output and plotting")
    output.add_argument("--plot", action="store_true", help="Plot after simulation")
    output.add_argument(
        "--plot-only", action="store_true", help="Only plot existing combined HDF5"
    )
    output.add_argument(
        "--keep-output", action="store_true", help="Do not clear output/ first"
    )
    output.add_argument(
        "--no-combine",
        action="store_true",
        help="Do not combine HDF5 outputs after this run; useful for SLURM array chunks.",
    )
    output.add_argument(
        "--combine-only",
        action="store_true",
        help="Only combine existing HDF5 outputs and exit.",
    )

    debug = parser.add_argument_group("Diagnostics")
    debug.add_argument("--debug", action="store_true")
    debug.add_argument("--debug-plots", action="store_true")
    debug.add_argument("--debug-dir", default="output/debug")
    debug.add_argument("--debug-max-points", type=int, default=5000)
    debug.add_argument("--quiet-progress", action="store_true")

    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply convenience modes before models are resolved."""
    if args.test:
        args.samples = 4
        args.workers = min(args.workers, 4)
    if args.all_types:
        args.types = list(ALL_TYPES)
    return args


def resolve_models(args: argparse.Namespace) -> tuple[ModelConfig, ...]:
    """Load explicit models from YAML, or convert legacy type names."""
    if args.experiment_grid:
        models = load_experiment_grid(args.experiment_grid)
    else:
        models = [legacy_model_config(name, base_seed=args.seed) for name in args.types]

    if not args.model_name:
        return tuple(models)

    requested = set(args.model_name)
    selected = [model for model in models if model.model_name in requested]
    missing = requested - {model.model_name for model in selected}
    if missing:
        raise ValueError(f"Unknown model name(s): {', '.join(sorted(missing))}")
    return tuple(selected)


def configure_worker_environment(args: argparse.Namespace) -> None:
    """Pass lightweight debug/progress settings through environment variables."""
    if args.debug or args.debug_plots:
        os.environ["AGN_DEBUG"] = "1"
    if args.debug_plots:
        os.environ["AGN_DEBUG_PLOTS"] = "1"
    os.environ["AGN_DEBUG_DIR"] = args.debug_dir
    os.environ["AGN_DEBUG_MAX_POINTS"] = str(args.debug_max_points)
    os.environ["AGN_PROGRESS"] = "0" if args.quiet_progress else "1"


def is_chunked_run(args: argparse.Namespace) -> bool:
    """Return True when this process owns only a slice of the global samples."""
    return args.sample_start != 0 or (
        args.sample_stop is not None and args.sample_stop != args.samples
    )


def prepare_output(args: argparse.Namespace) -> None:
    """Clear output for fresh full simulation runs, preserving chunked outputs."""
    if args.plot_only or args.combine_only or args.keep_output or is_chunked_run(args):
        if is_chunked_run(args) and not args.keep_output:
            print("Chunked sample range detected; preserving output/ automatically.")
        OUTPUT_DIR.mkdir(exist_ok=True)
        return
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)


def print_run_summary(args: argparse.Namespace, *, is_hpc: bool) -> None:
    models = ", ".join(model.model_name for model in args.model_configs)
    print("\nAGN BBH run")
    print(f"  environment: {'HPC/SLURM' if is_hpc else 'local'}")
    if args.combine_only:
        print("  mode: combine-only")
        print("  source: output/*/*.h5")
        return
    print(f"  samples/model: {args.samples}")
    if is_chunked_run(args):
        stop = args.sample_stop if args.sample_stop is not None else args.samples
        print(f"  sample range: {args.sample_start}:{stop}")
    print(f"  workers: {args.workers}")
    print(f"  models: {models}")
    chunk_samples = (
        args.sample_stop if args.sample_stop is not None else args.samples
    ) - args.sample_start
    print(f"  total tasks: {chunk_samples * len(args.model_configs)}")
    print(f"  seed: {args.seed}")
    if args.experiment_grid:
        print(f"  experiment grid: {args.experiment_grid}")
    print(f"  progress: {'quiet' if args.quiet_progress else 'live'}")
    print(f"  diagnostics: {'on' if args.debug or args.debug_plots else 'off'}")


def run_simulation_pipeline(args: argparse.Namespace) -> None:
    """Run all requested model/sample tasks, then combine HDF5 fragments."""
    from physical_simulation import run_physics_simulation

    # ParallelSimulationManager owns the shared SMBH mass grid. This is the key
    # comparison invariant: sample_index=i means the same SMBH in every model.
    manager = ParallelSimulationManager(
        SimulationConfig(
            num_samples=args.samples,
            num_workers=args.workers,
            model_configs=args.model_configs,
            timeout=args.timeout,
            output_root=str(OUTPUT_DIR),
            seed=args.seed,
            sample_start=args.sample_start,
            sample_stop=args.sample_stop,
        )
    )
    manager.run_all_simulations(run_physics_simulation)

    if args.no_combine:
        print("\nSkipping HDF5 combination (--no-combine).")
        return

    run_combine_pipeline()


def run_combine_pipeline() -> None:
    """Combine existing per-sample HDF5 fragments into per-model files."""
    import storedata

    print("\nCombining HDF5 fragments...")
    storedata.combine_all_files(
        final_output_dir=str(OUTPUT_DIR / "final_combined"),
        source_root=str(OUTPUT_DIR),
    )


def run_plotting_pipeline(args: argparse.Namespace) -> None:
    """Build figures from output/final_combined/*.h5."""
    print("\nGenerating figures...")
    import fig_main

    fig_main.main(
        model_names=[model.model_name for model in args.model_configs],
        model_configs=args.model_configs,
        experiment_grid=args.experiment_grid,
        data_dir=str(OUTPUT_DIR),
    )


def parse_cli() -> tuple[argparse.Namespace, bool]:
    default_samples, default_workers, is_hpc = runtime_defaults()
    parser = build_parser(default_samples, default_workers)
    args = normalize_args(parser.parse_args())
    args.model_configs = resolve_models(args)
    return args, is_hpc


def main() -> None:
    setup_parallel_environment()
    args, is_hpc = parse_cli()

    if args.list_models:
        for model in args.model_configs:
            print(model.model_name)
        return

    try:
        configure_worker_environment(args)
        prepare_output(args)
        print_run_summary(args, is_hpc=is_hpc)

        if args.combine_only:
            run_combine_pipeline()
        elif not args.plot_only:
            run_simulation_pipeline(args)
        if not args.combine_only and (args.plot or args.plot_only):
            run_plotting_pipeline(args)

        print("\nAll tasks completed successfully.")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:
        print(f"\nError: {exc}")
        raise
    finally:
        print("Finished.")


if __name__ == "__main__":
    main()
