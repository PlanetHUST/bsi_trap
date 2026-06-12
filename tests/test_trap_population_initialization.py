from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Direct script execution puts tests/ on sys.path, so expose the src/ package root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bsi_trap.config.model_config import ModelConfig
from bsi_trap.config.rng import make_rng_streams
from bsi_trap.constant.constant import AU, Myr
from bsi_trap.plots.fig_storage import ensure_dir, save_figure
from workflow import simulation


def run_initialization_diagnostic(
    *,
    smbh_mass_msun: float,
    population_size: int,
    seed: int,
    disk_resolution: int,
    agn_lifetime_myr: float,
    output_dir: Path,
) -> None:
    """Run one real initialization task and save population diagnostics."""
    config = ModelConfig(
        model_name="trap-population-diagnostic",
        agn_lifetime_myr=agn_lifetime_myr,
        disk_resolution=disk_resolution,
        master_seed=seed,
    )
    agn, disk = simulation._build_environment(config, smbh_mass_msun)
    rng = make_rng_streams(seed)

    # The current NSC normalization returns zero after integer conversion.
    # Fix the Monte Carlo size explicitly while retaining the real physics path.
    with patch(
        "bsi_trap.main_class_op.embedded_BH_initialize.number_R_NSC",
        return_value=population_size,
    ):
        population = simulation.initialize_trap_population(
            model_config=config,
            agn=agn,
            disk=disk,
            rng_pair_up=rng.pair_up,
            rng_nsc=rng.nsc,
            rng_orbit=rng.dynamics,
            rng_spin=rng.spin,
        )

    _print_population_statistics(
        population=population,
        smbh_mass_msun=smbh_mass_msun,
        population_size=population_size,
        seed=seed,
    )
    figure_paths = _plot_population(population, ensure_dir(output_dir))
    print("\nSaved figures:")
    for path in figure_paths:
        print(f"  {path}")


def _print_population_statistics(
    *,
    population: simulation.TrapPopulation,
    smbh_mass_msun: float,
    population_size: int,
    seed: int,
) -> None:
    ebh = population.embedded_bh
    bbh = population.binaries
    sbh = population.singles
    reached_count = int(np.count_nonzero(ebh.trap_flag != -1))

    print("\nTrap population initialization")
    print(f"  SMBH mass:              {smbh_mass_msun:.3e} Msun")
    print(f"  Monte Carlo population: {population_size}")
    print(f"  random seed:            {seed}")
    print(f"  reached trap:           {reached_count}")
    print(f"  BBHs in trap:           {bbh.num}")
    print(f"  single BHs in trap:     {sbh.num}")

    _print_array_summary("BBH primary mass", bbh.m1_msun, "Msun")
    _print_array_summary("BBH secondary mass", bbh.m2_msun, "Msun")
    _print_array_summary("BBH mass ratio", _safe_ratio(bbh.m2_msun, bbh.m1_msun), "")
    _print_array_summary("BBH semimajor axis", bbh.sma_cm / AU, "AU")
    _print_array_summary("BBH eccentricity", bbh.ecc, "")
    _print_array_summary("BBH trap time", bbh.t_ini_s / Myr, "Myr")
    _print_array_summary("single-BH mass", sbh.m_msun, "Msun")
    _print_array_summary("single-BH trap time", sbh.t_ini_s / Myr, "Myr")


def _print_array_summary(label: str, values: np.ndarray, unit: str) -> None:
    values = np.asarray(values, dtype=float)
    suffix = f" {unit}" if unit else ""
    if values.size == 0:
        print(f"  {label:<23} no objects")
        return
    print(
        f"  {label:<23} mean={np.mean(values):.3e}{suffix}, "
        f"min={np.min(values):.3e}{suffix}, max={np.max(values):.3e}{suffix}"
    )


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    )


def _plot_population(
    population: simulation.TrapPopulation,
    output_dir: Path,
) -> list[Path]:
    bbh = population.binaries
    sbh = population.singles
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(bbh.m1_msun, bins="auto", alpha=0.7, label="BBH primary")
    axes[0].hist(bbh.m2_msun, bins="auto", alpha=0.7, label="BBH secondary")
    axes[0].hist(sbh.m_msun, bins="auto", alpha=0.7, label="Single BH")
    axes[0].set_xlabel(r"Mass ($M_\odot$)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].scatter(bbh.m1_msun, bbh.m2_msun, s=24, alpha=0.7)
    if bbh.num:
        mass_limit = float(np.max(bbh.m1_msun))
        axes[1].plot([0.0, mass_limit], [0.0, mass_limit], "k--", linewidth=1)
    axes[1].set_xlabel(r"Primary mass ($M_\odot$)")
    axes[1].set_ylabel(r"Secondary mass ($M_\odot$)")
    paths.append(save_figure(fig, output_dir, "trap_population_masses"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(bbh.sma_cm / AU, bbh.ecc, s=24, alpha=0.7)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("BBH semimajor axis (AU)")
    axes[0].set_ylabel("BBH eccentricity")

    axes[1].scatter(bbh.chi1, bbh.chi2, s=24, alpha=0.7)
    axes[1].set_xlabel(r"$\chi_1$")
    axes[1].set_ylabel(r"$\chi_2$")
    paths.append(save_figure(fig, output_dir, "trap_population_orbits_spins"))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(bbh.t_ini_s / Myr, bins="auto", alpha=0.7, label="BBH")
    ax.hist(sbh.t_ini_s / Myr, bins="auto", alpha=0.7, label="Single BH")
    ax.set_xlabel("Trap initialization time (Myr)")
    ax.set_ylabel("Count")
    ax.legend()
    paths.append(save_figure(fig, output_dir, "trap_population_times"))

    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and visualize one trap-population initialization."
    )
    parser.add_argument("--smbh-mass-msun", type=float, default=1.0e6)
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--disk-resolution", type=int, default=128)
    parser.add_argument("--agn-lifetime-myr", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report/trap_population_initialization"),
    )
    parser.add_argument(
        "--unit-test",
        action="store_true",
        help="Run the deterministic unit test instead of the real diagnostic.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.unit_test:
        unittest.main(argv=[__file__])
    else:
        run_initialization_diagnostic(
            smbh_mass_msun=args.smbh_mass_msun,
            population_size=args.population_size,
            seed=args.seed,
            disk_resolution=args.disk_resolution,
            agn_lifetime_myr=args.agn_lifetime_myr,
            output_dir=args.output_dir,
        )
