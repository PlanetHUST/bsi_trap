import argparse
import logging
from pathlib import Path

import numpy as np

from cli.common import add_log_level_argument, configure_logging_from_args
from bsi_trap.config.model_config import ModelConfig
from workflow.parallel_manager import run_simulation_workflow
from workflow.sample_run import run_physics_sample

logger = logging.getLogger(__name__)


def load_smbh_mass_grid(path: Path) -> list[float]:
    if not path.exists():
        raise FileNotFoundError(f"SMBH mass grid file not found: {path}")

    data = np.loadtxt(path, dtype=float)

    data = np.atleast_1d(data)

    if data.ndim != 1:
        raise ValueError("SMBH mass grid must be a one-dimensional file.")

    return data.tolist()


def add_simulate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--smbh-grid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--temp-dir", type=Path, default=Path("temp"))
    parser.add_argument("--model-name", default="disk_trap_single_bbh")
    parser.add_argument(
        "--gas-mode",
        choices=["with_gas", "without_gas"],
        default="without_gas",
    )
    parser.add_argument(
        "--disk-model",
        choices=["sirko", "thompson"],
        default="sirko",
    )
    parser.add_argument("--agn-lifetime-myr", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--disk-resolution", type=int, default=256)
    parser.add_argument("--bsi-enabled", action="store_true")
    parser.add_argument("--bsi-eta", type=float, default=6.0)
    parser.add_argument("--pair-up-eta", type=float, default=2.5)
    parser.add_argument("--trap-width-scale-heights", type=float, default=1.0)
    parser.add_argument(
        "--compact-object-scale-height-fraction",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--trap-radial-dispersion-scale-heights",
        type=float,
        default=0.25,
    )
    parser.add_argument("--max-generations", type=int, default=4)


def add_simulate_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "simulate",
        help="Run parallel simulations.",
    )
    add_simulate_arguments(parser)
    parser.set_defaults(func=run_simulate_command)


def run_simulate_command(args: argparse.Namespace) -> None:
    logger.info("Running simulation command")

    smbh_mass_grid = load_smbh_mass_grid(args.smbh_grid)

    if len(smbh_mass_grid) != args.num_samples:
        raise ValueError(
            f"SMBH grid has {len(smbh_mass_grid)} entries, "
            f"but --num-samples={args.num_samples}."
        )

    models = [
        ModelConfig(
            model_name=args.model_name,
            gas_mode=args.gas_mode,
            disk_model=args.disk_model,
            bsi_enabled=args.bsi_enabled,
            agn_lifetime_myr=args.agn_lifetime_myr,
            alpha=args.alpha,
            disk_resolution=args.disk_resolution,
            bsi_eta=args.bsi_eta,
            pair_up_eta=args.pair_up_eta,
            trap_width_scale_heights=args.trap_width_scale_heights,
            compact_object_scale_height_fraction=(
                args.compact_object_scale_height_fraction
            ),
            trap_radial_dispersion_scale_heights=(
                args.trap_radial_dispersion_scale_heights
            ),
            max_generations=args.max_generations,
        )
    ]

    run_simulation_workflow(
        models=models,
        smbh_mass_grid=smbh_mass_grid,
        physics_callback=run_physics_sample,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        base_seed=args.base_seed,
    )

    logger.info("Simulation command finished")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BSI-trap simulations.")
    add_log_level_argument(parser)
    add_simulate_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging_from_args(args)
    run_simulate_command(args)


if __name__ == "__main__":
    main()
