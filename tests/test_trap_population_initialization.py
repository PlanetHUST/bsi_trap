from __future__ import annotations

import argparse
import logging
import os
import sys
import unittest
from dataclasses import dataclass
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
from bsi_trap.constant.constant import AU, G, Myr, msun
from bsi_trap.classes.models import (
    AGN_properties,
    BBH_properties,
    DiskProfile,
    Embedded_BH,
    Single_BH_properties,
    TrapContext,
)
from bsi_trap.evolve.pairing import construct_pairing_up_timeline
from bsi_trap.evolve.trap_context import build_trap_context
from bsi_trap.initialization.bbh_parameters import BBH_params_initialize
from bsi_trap.functions.pairing_physics import (
    conjunction_rate,
    effective_capture_impact_parameter,
    is_strong_gravity_encounter,
    pair_up_timescale,
    shear_relative_velocity,
    sbh_cross_section,
)
from bsi_trap.functions.random_sample import sample_embedded_radius
from bsi_trap.initialization.embedded_bh import load_filtered_stellar_population
from bsi_trap.initialization.trap_population import (
    construct_initial_trap_condition,
    init_bbh,
    init_sbh,
)
from bsi_trap.plots.fig_storage import ensure_dir, save_figure
from workflow import simulation

TEST_EMBEDDED_BH_NUM = 2000
TEST_PAIR_UP_ETA = 2.5


@dataclass(frozen=True)
class InitializedTrapPopulation:
    embedded_bh: Embedded_BH
    binaries: BBH_properties
    singles: Single_BH_properties


def _initialize_population(
    *,
    smbh_mass_msun: float,
    agn_lifetime_myr: float,
    disk_resolution: int,
    embedded_bh_num: int,
    pair_up_eta: float,
    seed: int,
) -> tuple[ModelConfig, AGN_properties, DiskProfile, InitializedTrapPopulation]:
    """Build a fixed-size reservoir and exercise only trap initialization."""

    config = ModelConfig(
        model_name="trap-population-initialization-test",
        agn_lifetime_myr=agn_lifetime_myr,
        disk_resolution=disk_resolution,
        pair_up_eta=pair_up_eta,
        master_seed=seed,
    )
    agn, disk = simulation._build_environment(config, smbh_mass=smbh_mass_msun)
    rng = make_rng_streams(seed)

    _, catalog_mass_msun, catalog_tform_myr = load_filtered_stellar_population()
    sample_idx = rng.nsc.integers(
        0,
        catalog_mass_msun.size,
        size=embedded_bh_num,
    )

    ebh = Embedded_BH(embedded_bh_num)
    ebh.init_ebh(
        catalog_mass_msun[sample_idx],
        catalog_tform_myr[sample_idx] * Myr,
        sample_embedded_radius(embedded_bh_num, agn, disk, rng.nsc),
    )
    bbh, sbh = construct_initial_trap_condition(
        model_config=config,
        agn=agn,
        disk=disk,
        ebh=ebh,
        rng_pair_up=rng.pair_up,
    )
    BBH_params_initialize(
        agn=agn,
        disk=disk,
        bbh=bbh,
        rng_orbit=rng.dynamics,
        rng_spin=rng.spin,
        model_config=config,
    )
    return config, agn, disk, InitializedTrapPopulation(ebh, bbh, sbh)


class TestInitialTrapCondition(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.agn, cls.disk, cls.population = _initialize_population(
            smbh_mass_msun=1.0e6,
            agn_lifetime_myr=10.0,
            disk_resolution=64,
            embedded_bh_num=TEST_EMBEDDED_BH_NUM,
            pair_up_eta=TEST_PAIR_UP_ETA,
            seed=12345,
        )

    def test_uses_requested_embedded_bh_count(self) -> None:
        self.assertEqual(self.population.embedded_bh.num, TEST_EMBEDDED_BH_NUM)

    def test_stores_migration_and_trap_arrival_times(self) -> None:
        ebh = self.population.embedded_bh
        trap_arrival_s = ebh.t_form_s + ebh.t_mig_s
        reached = trap_arrival_s < self.agn.lifetime_myr * Myr

        self.assertEqual(ebh.t_mig_s.shape, (ebh.num,))
        self.assertTrue(np.all(ebh.t_mig_s >= 0.0))
        np.testing.assert_array_equal(reached, ebh.trap_flag != -1)

    def test_omits_bhs_without_stable_mass_dependent_traps(self) -> None:
        ebh = self.population.embedded_bh
        omitted = np.isnan(ebh.r_trap_target_cm)

        self.assertGreater(np.count_nonzero(omitted), 0)
        self.assertTrue(np.all(ebh.trap_flag[omitted] == -1))
        self.assertTrue(np.all(np.isinf(ebh.t_mig_s[omitted])))
        self.assertAlmostEqual(
            ebh.trap_width_cm,
            self.config.trap_width_scale_heights
            * self.disk.get_scale_height_cm(self.disk.r_trap_cm),
        )
        reached = ebh.trap_flag != -1
        self.assertTrue(np.all(np.isfinite(ebh.r_pair_cm[reached])))
        self.assertTrue(np.all(np.isnan(ebh.r_pair_cm[~reached])))
        self.assertTrue(
            np.all(
                np.abs(ebh.r_pair_cm[reached] - ebh.r_trap_target_cm[reached])
                <= 0.5 * ebh.trap_width_cm * (1.0 + 1.0e-12)
            )
        )

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

    def test_formed_bbhs_have_valid_initial_orbital_and_spin_properties(self) -> None:
        bbh = self.population.binaries
        hill_radius_cm = self.disk.r_trap_cm * (
            (bbh.m1_msun + bbh.m2_msun) / self.agn.M_SMBH_msun
        ) ** (1.0 / 3.0)

        if bbh.num == 0:
            return
        self.assertTrue(np.all(bbh.t_ini_s < self.agn.lifetime_myr * Myr))
        self.assertTrue(np.all(bbh.sma_cm > 0.0))
        self.assertTrue(np.all(bbh.sma_cm <= hill_radius_cm))
        self.assertTrue(np.all((bbh.ecc >= 0.0) & (bbh.ecc < 1.0)))
        self.assertTrue(np.all((bbh.chi1 >= 0.0) & (bbh.chi1 <= 1.0)))
        self.assertTrue(np.all((bbh.chi2 >= 0.0) & (bbh.chi2 <= 1.0)))
        self.assertTrue(np.all((bbh.th1 >= 0.0) & (bbh.th1 <= np.pi)))
        self.assertTrue(np.all((bbh.th2 >= 0.0) & (bbh.th2 <= np.pi)))


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
        ebh.r_pair_cm[:] = [1.00e12, 1.01e12, 1.02e12, 1.03e12, np.nan]
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
            smbh_mass_msun=1.0e6,
            end_time_s=1.0e22,
            rng_pair_up=np.random.default_rng(seed),
        )
        return ebh, formation_times_s

    def test_pairing_uses_only_reached_singles(self) -> None:
        ebh, formation_times_s = self._build_pairing_timeline(seed=7)
        _, disk, _, _ = self._pairing_inputs()
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
            self.assertGreaterEqual(
                ebh.t_pair_s[first],
                max(ebh.t_reach_s[first], ebh.t_reach_s[second])
                + pair_up_timescale(
                    disk,
                    float(ebh.m_msun[first]),
                    float(ebh.m_msun[second]),
                ),
            )

    def test_pairing_requires_local_radial_overlap(self) -> None:
        config, disk, _, _ = self._pairing_inputs()
        ebh = Embedded_BH(2)
        ebh.m_msun[:] = [10.0, 20.0]
        ebh.t_reach_s[:] = [1.0, 1.0]
        ebh.r_pair_cm[:] = [1.00e12, 1.90e12]

        formation_times_s = construct_pairing_up_timeline(
            embedded_bh=ebh,
            reached_mask=np.array([True, True]),
            trap_context=build_trap_context(disk, config),
            model_config=config,
            disk=disk,
            smbh_mass_msun=1.0e6,
            end_time_s=1.0e22,
            rng_pair_up=np.random.default_rng(7),
        )

        self.assertEqual(formation_times_s.size, 0)
        np.testing.assert_array_equal(ebh.trap_flag, [-2, -2])

    def test_pair_up_timescale_uses_midplane_density(self) -> None:
        _, disk, _, _ = self._pairing_inputs()
        total_mass_msun = 30.0
        sound_speed_cgs = disk.get_cs(disk.r_trap_cm)
        scale_height_cm = disk.get_scale_height_cm(disk.r_trap_cm)
        midplane_density_cgs = disk.get_sigma(disk.r_trap_cm) / (2.0 * scale_height_cm)
        expected_s = (
            3.0
            * sound_speed_cgs**3
            / (4.0 * np.pi * G**2 * midplane_density_cgs * total_mass_msun * msun)
        )

        self.assertAlmostEqual(pair_up_timescale(disk, 10.0, 20.0), expected_s)

    def test_capture_cross_section_uses_hill_radius_and_finite_layer(self) -> None:
        config, disk, _, _ = self._pairing_inputs()
        trap_context = build_trap_context(disk, config)
        total_mass_msun = 30.0
        expected_hill_radius_cm = disk.r_trap_cm * (
            total_mass_msun / (3.0 * 1.0e6)
        ) ** (1.0 / 3.0)
        impact_parameter_cm = effective_capture_impact_parameter(
            model_config=config,
            total_mass_msun=total_mass_msun,
            trap_radius_cm=disk.r_trap_cm,
            smbh_mass_msun=1.0e6,
            trap_width_cm=trap_context.width_cm,
        )
        self.assertAlmostEqual(
            impact_parameter_cm,
            min(config.pair_up_eta * expected_hill_radius_cm, trap_context.width_cm / 2.0),
        )

        thin_layer_context = TrapContext(
            width_cm=trap_context.width_cm,
            scale_height_cm=1.0e10,
            volume_cm3=trap_context.volume_cm3 / 10.0,
        )
        layer_limited_config = ModelConfig(
            model_name="layer-limited-capture-test",
            pair_up_eta=10.0,
        )
        layer_limited_impact_cm = effective_capture_impact_parameter(
            model_config=layer_limited_config,
            total_mass_msun=total_mass_msun,
            trap_radius_cm=disk.r_trap_cm,
            smbh_mass_msun=1.0e6,
            trap_width_cm=thin_layer_context.width_cm,
        )
        expected_area_cm2 = 2.0 * (
            thin_layer_context.scale_height_cm
            * np.sqrt(
                layer_limited_impact_cm**2 - thin_layer_context.scale_height_cm**2
            )
            + layer_limited_impact_cm**2
            * np.arcsin(thin_layer_context.scale_height_cm / layer_limited_impact_cm)
        )
        self.assertAlmostEqual(
            sbh_cross_section(
                model_config=layer_limited_config,
                total_mass_msun=total_mass_msun,
                trap_radius_cm=disk.r_trap_cm,
                smbh_mass_msun=1.0e6,
                trap_context=thin_layer_context,
            ),
            expected_area_cm2,
        )

    def test_local_shear_and_gravity_gate_define_conjunction_candidates(self) -> None:
        first_radius_cm = 1.00e12
        second_radius_cm = 1.01e12
        relative_velocity_cms = shear_relative_velocity(
            1.0e6,
            first_radius_cm,
            second_radius_cm,
        )
        self.assertGreater(relative_velocity_cms, 0.0)
        self.assertGreater(
            conjunction_rate(1.0e6, first_radius_cm, second_radius_cm),
            0.0,
        )
        self.assertTrue(
            is_strong_gravity_encounter(
                10.0,
                20.0,
                relative_velocity_cms,
                5.0e10,
            )
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


def _print_population_summary(
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    population: InitializedTrapPopulation,
) -> None:
    ebh = population.embedded_bh
    bbh = population.binaries
    sbh = population.singles
    trap_arrival_s = ebh.t_form_s + ebh.t_mig_s
    reached = ebh.trap_flag != -1
    paired = ebh.trap_flag >= 0

    print("\nPhysical trap-population initialization")
    print(f"  SMBH mass:                {agn.M_SMBH_msun:.3e} Msun")
    print(f"  initialized embedded BHs: {ebh.num}")
    print(f"  pair-up Hill multiplier:  {model_config.pair_up_eta:.3e}")
    print(f"  configured trap width:    {ebh.trap_width_cm:.3e} cm")
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


def _legend_if_labeled(ax) -> None:
    """Add a legend only when this diagnostic panel has plotted data."""
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


def _plot_population_diagnostics(
    population,
    output_dir: Path,
) -> list[Path]:
    ebh = population.embedded_bh
    bbh = population.binaries
    reached = ebh.trap_flag != -1
    output_dir = ensure_dir(output_dir)
    arrival_times_s = np.sort(ebh.t_reach_s[reached])
    pairing_times_s = np.sort(bbh.t_ini_s)
    event_times_s = np.concatenate((arrival_times_s, pairing_times_s))
    event_kind = np.concatenate(
        (
            np.zeros(arrival_times_s.size, dtype=int),
            np.ones(pairing_times_s.size, dtype=int),
        )
    )
    order = np.lexsort((event_kind, event_times_s))
    event_times_myr = event_times_s[order] / Myr
    single_count = np.cumsum(np.where(event_kind[order] == 0, 1, -2))
    binary_count = np.cumsum(np.where(event_kind[order] == 1, 1, 0))

    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)
    axes[0].step(event_times_myr, single_count, where="post", label="Trap singles")
    axes[0].step(event_times_myr, binary_count, where="post", label="Trap BBHs")
    axes[0].set_ylabel("Objects")
    axes[0].set_title("Initial trap-population timeline")
    _legend_if_labeled(axes[0])

    pairing_times_myr = pairing_times_s / Myr
    if bbh.num:
        axes[1].scatter(pairing_times_myr, bbh.sma_cm / AU, label="BBH")
        axes[1].set_yscale("log")
        axes[2].scatter(pairing_times_myr, bbh.ecc, label="BBH")
        axes[3].scatter(pairing_times_myr, bbh.chi1, label=r"$\chi_1$")
        axes[3].scatter(pairing_times_myr, bbh.chi2, label=r"$\chi_2$")
        axes[4].scatter(pairing_times_myr, bbh.th1, label=r"$\theta_1$")
        axes[4].scatter(pairing_times_myr, bbh.th2, label=r"$\theta_2$")
        for ax in axes[1:]:
            _legend_if_labeled(ax)
    else:
        for ax in axes[1:]:
            ax.text(
                0.5,
                0.5,
                "No BBHs formed during the selected AGN lifetime",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    axes[1].set_ylabel("BBH semimajor axis (AU)")
    axes[2].set_ylabel("BBH eccentricity")
    axes[3].set_ylabel("Spin magnitude")
    axes[4].set_ylabel("Spin tilt (rad)")
    axes[4].set_xlabel("Event time (Myr)")
    return [save_figure(fig, output_dir, "trap_population_timeline")]


def run_diagnostics(
    *,
    smbh_mass_msun: float,
    agn_lifetime_myr: float,
    disk_resolution: int,
    embedded_bh_num: int,
    pair_up_eta: float,
    seed: int,
    output_dir: Path,
) -> None:
    config, agn, disk, population = _initialize_population(
        smbh_mass_msun=smbh_mass_msun,
        agn_lifetime_myr=agn_lifetime_myr,
        disk_resolution=disk_resolution,
        embedded_bh_num=embedded_bh_num,
        pair_up_eta=pair_up_eta,
        seed=seed,
    )
    _print_population_summary(config, agn, disk, population)
    paths = _plot_population_diagnostics(population, output_dir)
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
    parser.add_argument("--embedded-bh-num", type=int, default=1_000)
    parser.add_argument(
        "--pair-up-eta",
        type=float,
        default=TEST_PAIR_UP_ETA,
        help="mutual-Hill-radius capture multiplier",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report/trap_population_timeline"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.diagnostic:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        run_diagnostics(
            smbh_mass_msun=args.smbh_mass_msun,
            agn_lifetime_myr=args.agn_lifetime_myr,
            disk_resolution=args.disk_resolution,
            embedded_bh_num=args.embedded_bh_num,
            pair_up_eta=args.pair_up_eta,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    else:
        unittest.main(argv=[sys.argv[0]])
