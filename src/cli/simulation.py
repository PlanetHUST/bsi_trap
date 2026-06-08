import argparse
import logging
from pathlib import Path

import numpy as np

from bsi_trap.config.model_config import ModelConfig
from workflow.parallel_manager import run_simulation_workflow
from workflow.simulation import run_physics_sample

logger = logging.getLogger(__name__)


def load_smbh_mass_grid(path: Path) -> list[float]:
    if not path.exists():
        raise FileNotFoundError(f"SMBH mass grid file not found: {path}")

    data = np.loadtxt(path, dtype=float)

    data = np.atleast_1d(data)

    if data.ndim != 1:
        raise ValueError("SMBH mass grid must be a one-dimensional file.")

    return data.tolist()


def add_simulation_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "simulate",
        help="Run parallel simulations.",
    )

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
    parser.add_argument("--disk-resolution", type=int, default=256)
    parser.add_argument("--migrator-mass-msun", type=float, default=10.0)

    parser.set_defaults(func=run_simulation_command)


def run_simulation_command(args: argparse.Namespace) -> None:
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
            agn_lifetime_myr=args.agn_lifetime_myr,
            disk_resolution=args.disk_resolution,
            migrator_mass_msun=args.migrator_mass_msun,
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
