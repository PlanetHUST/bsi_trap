#!/usr/bin/env python
"""Run a trap-arrival initialization diagnostic from a source checkout."""

from __future__ import annotations

import argparse

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from bsi_trap.simulation.config.model_config import ModelConfig
from bsi_trap.simulation.config.rng import make_rng_streams, resolve_seed
from bsi_trap.constant.constant import Myr
from bsi_trap.simulation.trap_init.environment import make_environment
from bsi_trap.simulation.trap_init.arrival import sbh_trap_arrival_cal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize one AGN trap-arrival reservoir and print a summary."
    )
    parser.add_argument("--smbh-mass", type=float, default=1.0e6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="trap_population_diagnostic")
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
    parser.add_argument("--preview", type=int, default=5)
    return parser


def _format_time_myr(values_s: np.ndarray, limit: int) -> str:
    values_myr = np.asarray(values_s[:limit], dtype=float) / Myr
    return np.array2string(values_myr, precision=4, separator=", ")


def main() -> None:
    args = build_parser().parse_args()
    seed = resolve_seed(args.seed)
    rng = make_rng_streams(seed)
    config = ModelConfig(
        model_name=args.model_name,
        gas_mode=args.gas_mode,
        disk_model=args.disk_model,
        agn_lifetime_myr=args.agn_lifetime_myr,
        disk_resolution=args.disk_resolution,
        pair_up_eta=args.pair_up_eta,
        trap_width_scale_heights=args.trap_width_scale_heights,
        object_scale_height_fraction=args.compact_object_scale_height_fraction,
        trap_radial_sigma_in_scale_height=args.trap_radial_dispersion_scale_heights,
    )

    agn, disk = make_environment(config, args.smbh_mass)
    arrival_state = sbh_trap_arrival_cal(
        model_config=config,
        agn=agn,
        disk=disk,
        rng_nsc=rng.nsc,
    )

    embedded = arrival_state.embedded_bh
    singles = arrival_state.singles
    reached = int(np.count_nonzero(embedded.trap_flag != -1))
    preview = max(0, args.preview)

    print("Trap arrival diagnostic")
    print(f"  model: {config.model_name}")
    print(f"  seed: {seed}")
    print(f"  SMBH mass [Msun]: {agn.M_SMBH_msun:.6g}")
    print(f"  AGN lifetime [Myr]: {agn.lifetime_myr:.6g}")
    print(f"  disk model: {config.disk_model}")
    print(f"  trap radius [cm]: {disk.r_trap_cm:.6e}")
    print(f"  embedded BHs: {embedded.num}")
    print(f"  reached trap: {reached}")
    print(f"  trap singles: {singles.num}")

    if singles.num > 0 and preview > 0:
        shown = min(preview, singles.num)
        print(f"  first {shown} single masses [Msun]:")
        print(
            f"    {np.array2string(singles.m_msun[:shown], precision=4, separator=', ')}"
        )
        print(f"  first {shown} single available times [Myr]:")
        print(f"    {_format_time_myr(singles.available_from_s, shown)}")


if __name__ == "__main__":
    main()
