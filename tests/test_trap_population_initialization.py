from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path

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
from bsi_trap.classes.models import DiskProfile, Embedded_BH
from bsi_trap.evolve.pairing import construct_pairing_up_timeline
from bsi_trap.evolve.trap_context import build_trap_context
from bsi_trap.functions.nsc_population import number_enclosed_NSC
from bsi_trap.main_class_op.initial_trap_population import init_bbh, init_sbh
from bsi_trap.plots.fig_storage import ensure_dir, save_figure
from workflow import simulation


def _initialize_population(
    *,
    smbh_mass_msun: float,
    agn_lifetime_myr: float,
    disk_resolution: int,
    seed: int,
):
    config = ModelConfig(
        model_name="trap-population-initialization-test",
        agn_lifetime_myr=agn_lifetime_myr,
        disk_resolution=disk_resolution,
        master_seed=seed,
    )
    agn, disk = simulation._build_environment(config, smbh_mass=smbh_mass_msun)
    rng = make_rng_streams(seed)
    population = simulation.initialize_trap_population(
        model_config=config,
        agn=agn,
        disk=disk,
        rng_pair_up=rng.pair_up,
        rng_nsc=rng.nsc,
        rng_orbit=rng.dynamics,
        rng_spin=rng.spin,
    )
    return config, agn, disk, population


class TestInitialTrapCondition(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.agn, cls.disk, cls.population = _initialize_population(
            smbh_mass_msun=1.0e3,
            agn_lifetime_myr=10.0,
            disk_resolution=64,
            seed=12345,
        )

    def test_uses_population_size_from_enclosed_nsc(self) -> None:
        expected = max(
            1,
            int(
                round(
                    number_enclosed_NSC(
                        self.agn.M_SMBH_msun,
                        self.agn.mave_msun,
                        self.disk.r_out_cm,
                    )
                )
            ),
        )
        self.assertEqual(self.population.embedded_bh.num, expected)

    def test_stores_migration_and_trap_arrival_times(self) -> None:
        ebh = self.population.embedded_bh
        trap_arrival_s = ebh.t_form_s + ebh.t_mig_s
        reached = trap_arrival_s < self.agn.lifetime_myr * Myr

        self.assertEqual(ebh.t_mig_s.shape, (ebh.num,))
        self.assertTrue(np.all(ebh.t_mig_s >= 0.0))
        np.testing.assert_array_equal(reached, ebh.trap_flag != -1)

    def test_reached_population_is_conserved_during_conversion(self) -> None:
        ebh = self.population.embedded_bh
        reached_count = int(np.count_nonzero(ebh.trap_flag != -1))
        converted_count = 2 * self.population.binaries.num + self.population.singles.num
        self.assertEqual(reached_count, converted_count)

    def test_pair_completion_does_not_precede_trap_arrival(self) -> None:
        ebh = self.population.embedded_bh
        paired = ebh.trap_flag >= 0
        trap_arrival_s = ebh.t_form_s + ebh.t_mig_s
        self.assertTrue(np.all(ebh.t_pair_s[paired] >= trap_arrival_s[paired]))


class TestPairingTimeline(unittest.TestCase):
    """Focused tests for the reached-single event timeline and conversion state."""

    @staticmethod
    def _pairing_inputs():
        config = ModelConfig(model_name="pairing-timeline-test")
        disk = DiskProfile(
            radius_cm=np.array([1.0e12, 2.0e12]),
            h_over_r=np.array([0.1, 0.1]),
            sigma_cgs=np.array([1.0e5, 1.0e5]),
            cs_cgs=np.array([1.0e7, 1.0e7]),
            r_in_cm=1.0e12,
            r_out_cm=2.0e12,
            r_trap_cm=1.0e12,
            r_antitrap_cm=2.0e12,
        )
        ebh = Embedded_BH(5)
        ebh.m_msun[:] = [10.0, 20.0, 30.0, 40.0, 50.0]
        ebh.t_reach_s[:] = [4.0, 1.0, 3.0, 2.0, 1.0e23]
        reached_mask = np.array([True, True, True, True, False])
        return config, disk, ebh, reached_mask

    @classmethod
    def _build_pairing_timeline(cls, seed: int):
        config, disk, ebh, reached_mask = cls._pairing_inputs()
        formation_times_s = construct_pairing_up_timeline(
            embedded_bh=ebh,
            reached_mask=reached_mask,
            trap_context=build_trap_context(disk, config),
            model_config=config,
            disk=disk,
            end_time_s=1.0e22,
            rng_pair_up=np.random.default_rng(seed),
        )
        return ebh, formation_times_s

    def test_pairing_uses_only_reached_singles(self) -> None:
        ebh, formation_times_s = self._build_pairing_timeline(seed=7)
        paired = ebh.trap_flag >= 0

        self.assertEqual(formation_times_s.shape, (2,))
        self.assertTrue(np.all(np.diff(formation_times_s) >= 0.0))
        self.assertEqual(np.count_nonzero(paired), 4)
        self.assertEqual(ebh.trap_flag[4], -1)
        self.assertTrue(np.isnan(ebh.t_pair_s[4]))

        for first in np.flatnonzero(paired):
            second = int(ebh.trap_flag[first])
            self.assertEqual(ebh.trap_flag[second], first)
            self.assertEqual(ebh.t_pair_s[first], ebh.t_pair_s[second])
            self.assertGreaterEqual(
                ebh.t_pair_s[first],
                max(ebh.t_reach_s[first], ebh.t_reach_s[second]),
            )

    def test_pairing_is_seed_reproducible_and_converts_population(self) -> None:
        first_ebh, first_times_s = self._build_pairing_timeline(seed=11)
        second_ebh, second_times_s = self._build_pairing_timeline(seed=11)

        np.testing.assert_array_equal(first_times_s, second_times_s)
        np.testing.assert_array_equal(first_ebh.trap_flag, second_ebh.trap_flag)
        np.testing.assert_array_equal(first_ebh.t_pair_s, second_ebh.t_pair_s)

        bbh = init_bbh(first_ebh)
        sbh = init_sbh(first_ebh)
        self.assertEqual(bbh.num, 2)
        self.assertEqual(sbh.num, 0)
        self.assertTrue(np.all(bbh.m1_msun >= bbh.m2_msun))
        self.assertTrue(np.all(np.isin(bbh.t_ini_s, first_times_s)))


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _print_summary(label: str, values: np.ndarray, unit: str = "") -> None:
    values = _finite(values)
    suffix = f" {unit}" if unit else ""
    if values.size == 0:
        print(f"  {label:<26} no objects")
        return
    print(
        f"  {label:<26} n={values.size}, mean={np.mean(values):.3e}{suffix}, "
        f"median={np.median(values):.3e}{suffix}, "
        f"min={np.min(values):.3e}{suffix}, max={np.max(values):.3e}{suffix}"
    )


def _print_population_summary(agn, disk, population) -> None:
    ebh = population.embedded_bh
    bbh = population.binaries
    sbh = population.singles
    trap_arrival_s = ebh.t_form_s + ebh.t_mig_s
    reached = ebh.trap_flag != -1
    paired = ebh.trap_flag >= 0

    print("\nPhysical trap-population initialization")
    print(f"  SMBH mass:                {agn.M_SMBH_msun:.3e} Msun")
    print(f"  enclosed NSC population:  {ebh.num}")
    print(f"  disk outer radius:         {disk.r_out_cm:.3e} cm")
    print(f"  reached migration trap:    {np.count_nonzero(reached)}")
    print(f"  BBHs / single BHs:         {bbh.num} / {sbh.num}")

    print("\nMass summaries")
    _print_summary("embedded BH mass", ebh.m_msun, "Msun")
    _print_summary("BBH primary mass", bbh.m1_msun, "Msun")
    _print_summary("BBH secondary mass", bbh.m2_msun, "Msun")
    _print_summary("single BH mass", sbh.m_msun, "Msun")

    print("\nSpin summaries")
    _print_summary("primary spin magnitude", bbh.chi1)
    _print_summary("secondary spin magnitude", bbh.chi2)
    _print_summary("primary spin tilt", bbh.th1, "rad")
    _print_summary("secondary spin tilt", bbh.th2, "rad")

    print("\nOrbit summaries")
    _print_summary("BBH semimajor axis", bbh.sma_cm / AU, "AU")
    _print_summary("BBH eccentricity", bbh.ecc)

    print("\nTime summaries")
    _print_summary("formation time", ebh.t_form_s / Myr, "Myr")
    _print_summary("migration time", ebh.t_mig_s / Myr, "Myr")
    _print_summary("reached trap arrival", trap_arrival_s[reached] / Myr, "Myr")
    _print_summary("pair completion", ebh.t_pair_s[paired] / Myr, "Myr")
    _print_summary(
        "pairing delay",
        (ebh.t_pair_s[paired] - trap_arrival_s[paired]) / Myr,
        "Myr",
    )


def _hist(ax, values: np.ndarray, *, label: str, **kwargs) -> None:
    values = _finite(values)
    if values.size:
        bins = max(1, min(30, int(np.ceil(np.sqrt(values.size)))))
        ax.hist(values, bins=bins, label=label, **kwargs)


def _plot_population_diagnostics(
    agn,
    population,
    output_dir: Path,
) -> list[Path]:
    ebh = population.embedded_bh
    bbh = population.binaries
    sbh = population.singles
    trap_arrival_s = ebh.t_form_s + ebh.t_mig_s
    reached = ebh.trap_flag != -1
    paired = ebh.trap_flag >= 0
    output_dir = ensure_dir(output_dir)
    paths = []

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    _hist(axes[0], ebh.m_msun, label="Embedded BH", alpha=0.65)
    _hist(axes[0], bbh.m1_msun, label="BBH primary", alpha=0.65)
    _hist(axes[0], bbh.m2_msun, label="BBH secondary", alpha=0.65)
    _hist(axes[0], sbh.m_msun, label="Single at trap", alpha=0.65)
    axes[0].set_xlabel(r"Mass ($M_\odot$)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    _hist(axes[1], bbh.chi1, label=r"$\chi_1$", alpha=0.65)
    _hist(axes[1], bbh.chi2, label=r"$\chi_2$", alpha=0.65)
    axes[1].set_xlabel("Dimensionless spin magnitude")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    _hist(axes[2], bbh.th1, label=r"$\theta_1$", alpha=0.65)
    _hist(axes[2], bbh.th2, label=r"$\theta_2$", alpha=0.65)
    axes[2].set_xlabel("Spin tilt (rad)")
    axes[2].set_ylabel("Count")
    axes[2].legend()
    paths.append(save_figure(fig, output_dir, "population_mass_spin_histograms"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _hist(axes[0], bbh.sma_cm / AU, label="BBH", alpha=0.75)
    axes[0].set_xlabel("BBH semimajor axis (AU)")
    axes[0].set_ylabel("Count")
    if np.any(bbh.sma_cm > 0.0):
        axes[0].set_xscale("log")
    axes[0].legend()

    _hist(axes[1], bbh.ecc, label="BBH", alpha=0.75)
    axes[1].set_xlabel("BBH eccentricity")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    paths.append(save_figure(fig, output_dir, "population_orbit_histograms"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _hist(
        axes[0],
        trap_arrival_s[reached] / Myr,
        label="Reached BHs",
        alpha=0.75,
    )
    axes[0].axvline(
        agn.lifetime_myr,
        color="black",
        linestyle="--",
        label="AGN lifetime",
    )
    axes[0].set_xlabel("Trap arrival time (Myr)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    _hist(axes[1], ebh.t_form_s / Myr, label=r"$t_{\rm form}$", alpha=0.55)
    _hist(axes[1], ebh.t_mig_s / Myr, label=r"$t_{\rm mig}$", alpha=0.55)
    _hist(
        axes[1],
        trap_arrival_s[reached] / Myr,
        label=r"$t_{\rm reach}$",
        histtype="step",
        linewidth=2,
    )
    _hist(
        axes[1],
        ebh.t_pair_s[paired] / Myr,
        label=r"$t_{\rm pair}$",
        histtype="step",
        linewidth=2,
    )
    axes[1].set_xlabel("Timescale (Myr)")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    paths.append(save_figure(fig, output_dir, "population_time_histograms"))

    return paths


def run_diagnostics(
    *,
    smbh_mass_msun: float,
    agn_lifetime_myr: float,
    disk_resolution: int,
    seed: int,
    output_dir: Path,
) -> None:
    _, agn, disk, population = _initialize_population(
        smbh_mass_msun=smbh_mass_msun,
        agn_lifetime_myr=agn_lifetime_myr,
        disk_resolution=disk_resolution,
        seed=seed,
    )
    _print_population_summary(agn, disk, population)
    paths = _plot_population_diagnostics(agn, population, output_dir)
    print("\nSaved figures:")
    for path in paths:
        print(f"  {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test or diagnose physical trap-population initialization."
    )
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--smbh-mass-msun", type=float, default=1.0e6)
    parser.add_argument("--agn-lifetime-myr", type=float, default=10.0)
    parser.add_argument("--disk-resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report/trap_population_initialization"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.diagnostic:
        run_diagnostics(
            smbh_mass_msun=args.smbh_mass_msun,
            agn_lifetime_myr=args.agn_lifetime_myr,
            disk_resolution=args.disk_resolution,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    else:
        unittest.main(argv=[sys.argv[0]])
