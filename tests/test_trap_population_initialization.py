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
from bsi_trap.evolve.timeline import PairingTimeline
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
from bsi_trap.initialization.environment import make_environment
from bsi_trap.initialization.trap_population import (
    construct_initial_trap_population,
    init_bbh,
    init_sbh,
)
from bsi_trap.plots.fig_storage import ensure_dir, save_figure

TEST_EMBEDDED_BH_NUM = 2000
TEST_PAIR_UP_ETA = 2.5


# Test fixtures


@dataclass(frozen=True)
class InitializedTrapPopulation:
    embedded_bh: Embedded_BH
    binaries: BBH_properties
    singles: Single_BH_properties
    pairing_timeline: PairingTimeline


@dataclass(frozen=True)
class PairTimelineData:
    """Derived timestamps for the formed-BBH diagnostic timeline."""

    first_idx: np.ndarray
    second_idx: np.ndarray
    migration_start_s: np.ndarray
    arrival_s: np.ndarray
    encounter_s: np.ndarray
    formation_s: np.ndarray
    pair_up_delay_s: np.ndarray


@dataclass(frozen=True)
class ReachedTrapIndexData:
    """Arrival-ordered local indices for the reached trap population."""

    global_indices: np.ndarray
    local_index_by_global: np.ndarray


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
    if embedded_bh_num <= 0:
        raise ValueError("embedded_bh_num must be positive")
    if pair_up_eta <= 0.0:
        raise ValueError("pair_up_eta must be positive")

    config = ModelConfig(
        model_name="trap-population-initialization-test",
        agn_lifetime_myr=agn_lifetime_myr,
        disk_resolution=disk_resolution,
        pair_up_eta=pair_up_eta,
        master_seed=seed,
    )
    agn, disk = make_environment(config, smbh_mass=smbh_mass_msun)
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
    bbh, sbh, pairing_timeline = construct_initial_trap_population(
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
    return config, agn, disk, InitializedTrapPopulation(
        ebh,
        bbh,
        sbh,
        pairing_timeline,
    )


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

    def test_rejects_invalid_diagnostic_population_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "embedded_bh_num must be positive"):
            _initialize_population(
                smbh_mass_msun=1.0e6,
                agn_lifetime_myr=10.0,
                disk_resolution=64,
                embedded_bh_num=0,
                pair_up_eta=TEST_PAIR_UP_ETA,
                seed=12345,
            )
        with self.assertRaisesRegex(ValueError, "pair_up_eta must be positive"):
            _initialize_population(
                smbh_mass_msun=1.0e6,
                agn_lifetime_myr=10.0,
                disk_resolution=64,
                embedded_bh_num=1,
                pair_up_eta=0.0,
                seed=12345,
            )

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


# Focused pairing-contract tests


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
        pairing_timeline = construct_pairing_up_timeline(
            embedded_bh=ebh,
            reached_mask=reached_mask,
            trap_context=build_trap_context(disk, config),
            model_config=config,
            disk=disk,
            smbh_mass_msun=1.0e6,
            end_time_s=1.0e22,
            rng_pair_up=np.random.default_rng(seed),
        )
        return ebh, pairing_timeline

    def test_pairing_uses_only_reached_singles(self) -> None:
        ebh, pairing_timeline = self._build_pairing_timeline(seed=7)
        _, disk, _, _ = self._pairing_inputs()
        paired = ebh.trap_flag >= 0

        self.assertEqual(pairing_timeline.formation_s.shape, (2,))
        self.assertTrue(np.all(np.diff(pairing_timeline.formation_s) >= 0.0))
        self.assertEqual(np.count_nonzero(paired), 4)
        self.assertEqual(ebh.trap_flag[4], -1)
        self.assertTrue(np.isnan(ebh.t_pair_s[4]))
        np.testing.assert_allclose(
            pairing_timeline.formation_s,
            pairing_timeline.encounter_s + pairing_timeline.pair_delay_s,
        )
        self.assertTrue(
            np.all(pairing_timeline.encounter_s >= pairing_timeline.later_arrival_s)
        )

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

        pairing_timeline = construct_pairing_up_timeline(
            embedded_bh=ebh,
            reached_mask=np.array([True, True]),
            trap_context=build_trap_context(disk, config),
            model_config=config,
            disk=disk,
            smbh_mass_msun=1.0e6,
            end_time_s=1.0e22,
            rng_pair_up=np.random.default_rng(7),
        )

        self.assertEqual(pairing_timeline.size, 0)
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
            min(
                config.pair_up_eta * expected_hill_radius_cm,
                trap_context.width_cm / 2.0,
            ),
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
        first_ebh, first_timeline = self._build_pairing_timeline(seed=11)
        second_ebh, second_timeline = self._build_pairing_timeline(seed=11)

        np.testing.assert_array_equal(
            first_timeline.formation_s,
            second_timeline.formation_s,
        )
        np.testing.assert_array_equal(
            first_timeline.encounter_s,
            second_timeline.encounter_s,
        )
        np.testing.assert_array_equal(first_ebh.trap_flag, second_ebh.trap_flag)
        np.testing.assert_array_equal(first_ebh.t_pair_s, second_ebh.t_pair_s)

        bbh = init_bbh(first_ebh)
        sbh = init_sbh(first_ebh)
        self.assertEqual(bbh.num, 2)
        self.assertEqual(sbh.num, 0)
        self.assertTrue(np.all(bbh.m1_msun >= bbh.m2_msun))
        self.assertTrue(np.all(np.isin(bbh.t_ini_s, first_timeline.formation_s)))

    def test_reached_trap_indices_exclude_unreached_bhs(self) -> None:
        ebh, _ = self._build_pairing_timeline(seed=7)
        indices = _reached_trap_index_data(ebh)

        np.testing.assert_array_equal(indices.global_indices, [1, 3, 2, 0])
        self.assertEqual(indices.local_index_by_global[4], -1)
        for global_index in indices.global_indices:
            partner = ebh.trap_flag[global_index]
            if partner >= 0:
                self.assertGreaterEqual(indices.local_index_by_global[partner], 0)


# Diagnostic data derivation and reporting


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


def _formed_pair_timeline_data(
    ebh: Embedded_BH,
    pairing_timeline: PairingTimeline,
) -> PairTimelineData:
    """Attach migration starts to the explicit pairing event timeline."""
    if pairing_timeline.size == 0:
        empty = np.empty(0, dtype=float)
        return PairTimelineData(
            first_idx=pairing_timeline.first_idx,
            second_idx=pairing_timeline.second_idx,
            migration_start_s=empty,
            arrival_s=empty,
            encounter_s=empty,
            formation_s=empty,
            pair_up_delay_s=empty,
        )

    later_component_idx = np.where(
        ebh.t_reach_s[pairing_timeline.first_idx]
        >= ebh.t_reach_s[pairing_timeline.second_idx],
        pairing_timeline.first_idx,
        pairing_timeline.second_idx,
    )
    return PairTimelineData(
        first_idx=pairing_timeline.first_idx,
        second_idx=pairing_timeline.second_idx,
        migration_start_s=ebh.t_form_s[later_component_idx],
        arrival_s=pairing_timeline.later_arrival_s,
        encounter_s=pairing_timeline.encounter_s,
        formation_s=pairing_timeline.formation_s,
        pair_up_delay_s=pairing_timeline.pair_delay_s,
    )


def _reached_trap_index_data(ebh: Embedded_BH) -> ReachedTrapIndexData:
    """Index only reached trap BHs, in stable arrival-time order."""
    global_indices = np.flatnonzero(ebh.trap_flag != -1)
    global_indices = global_indices[
        np.argsort(ebh.t_reach_s[global_indices], kind="stable")
    ]
    local_index_by_global = np.full(ebh.num, -1, dtype=int)
    local_index_by_global[global_indices] = np.arange(global_indices.size)
    return ReachedTrapIndexData(
        global_indices=global_indices,
        local_index_by_global=local_index_by_global,
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
    stable = np.isfinite(ebh.r_trap_target_cm)
    late = stable & ~reached
    pair_timeline = _formed_pair_timeline_data(ebh, population.pairing_timeline)
    reached_indices = _reached_trap_index_data(ebh)
    single_global_indices = reached_indices.global_indices[
        ebh.trap_flag[reached_indices.global_indices] == -2
    ]
    single_residence_s = agn.lifetime_myr * Myr - ebh.t_reach_s[single_global_indices]

    def print_count(label: str, count: int) -> None:
        print(f"  {label:<26} {count:5d} ({count / ebh.num:6.2%})")

    print("\nPhysical trap-population initialization")
    print(f"  SMBH mass:                {agn.M_SMBH_msun:.3e} Msun")
    print(f"  initialized embedded BHs: {ebh.num}")
    print(f"  pair-up Hill multiplier:  {model_config.pair_up_eta:.3e}")
    print(f"  configured trap width:    {ebh.trap_width_cm:.3e} cm")
    print(f"  disk outer radius:         {disk.r_out_cm:.3e} cm")
    print(f"  reached migration trap:    {np.count_nonzero(reached)}")
    print(f"  BBHs / single BHs:         {bbh.num} / {sbh.num}")

    print("\nPopulation outcomes")
    print_count("stable migration target", int(np.count_nonzero(stable)))
    print_count("no stable target", int(np.count_nonzero(~stable)))
    print_count("stable but late", int(np.count_nonzero(late)))
    print_count("reached trap", int(np.count_nonzero(reached)))
    print_count("paired BH members", int(np.count_nonzero(paired)))
    print_count("trap singles", int(np.count_nonzero(ebh.trap_flag == -2)))
    print(f"  formed BBHs:               {bbh.num:5d}")
    print("\nReached-trap population mapping (arrival-order indices)")
    for first, second, formation_s in zip(
        pair_timeline.first_idx,
        pair_timeline.second_idx,
        pair_timeline.formation_s,
    ):
        print(
            "  trap indices "
            f"{reached_indices.local_index_by_global[first]:4d} <-> "
            f"{reached_indices.local_index_by_global[second]:4d}  "
            f"(embedded {first:4d} <-> {second:4d})  "
            f"forms at {formation_s / Myr:.6f} Myr"
        )
    for global_index in single_global_indices:
        local_index = reached_indices.local_index_by_global[global_index]
        residence_s = agn.lifetime_myr * Myr - ebh.t_reach_s[global_index]
        print(
            f"  trap index {local_index:4d}  (embedded {global_index:4d})  "
            f"single after {ebh.t_reach_s[global_index] / Myr:.6f} Myr; "
            f"residence to AGN end {residence_s / Myr:.6f} Myr"
        )
    if reached_indices.global_indices.size == 0:
        print("  no BHs reached the trap")

    print("\nMass summaries")
    _print_summary("embedded BH mass", ebh.m_msun, "Msun")
    _print_summary("initial disk radius", ebh.R_cm / disk.r_trap_cm, "R_trap")
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
    _print_summary("single trap residence", single_residence_s / Myr, "Myr")
    _print_summary("pair completion", ebh.t_pair_s[paired] / Myr, "Myr")
    _print_summary(
        "later-member arrival",
        pair_timeline.arrival_s / Myr,
        "Myr",
    )
    _print_summary(
        "encounter time",
        pair_timeline.encounter_s / Myr,
        "Myr",
    )
    _print_summary(
        "post-arrival encounter wait",
        (pair_timeline.encounter_s - pair_timeline.arrival_s) / Myr,
        "Myr",
    )
    _print_summary(
        "gas pair-up delay",
        pair_timeline.pair_up_delay_s / Myr,
        "Myr",
    )
    _print_summary(
        "arrival-to-BBH duration",
        (pair_timeline.formation_s - pair_timeline.arrival_s) / Myr,
        "Myr",
    )


def _legend_if_labeled(ax) -> None:
    """Add a legend only when this diagnostic panel has plotted data."""
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


# Diagnostic figure rendering


def _figure_directory(output_dir: Path, category: str) -> Path:
    """Return one responsibility-based directory within a diagnostic run."""
    return ensure_dir(output_dir / category)


def _plot_overview_and_bbh_diagnostics(
    population,
    output_dir: Path,
) -> list[Path]:
    """Write one direct event/property view per figure."""
    ebh = population.embedded_bh
    bbh = population.binaries
    reached = ebh.trap_flag != -1
    output_dir = ensure_dir(output_dir)
    arrival_times_s = np.sort(ebh.t_reach_s[reached])
    pairing_order = np.argsort(bbh.t_ini_s, kind="stable")
    pairing_times_s = bbh.t_ini_s[pairing_order]
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

    pairing_times_myr = pairing_times_s / Myr
    paths: list[Path] = []

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.step(event_times_myr, single_count, where="post", label="Trap singles")
    axis.step(event_times_myr, binary_count, where="post", label="Trap BBHs")
    axis.set_title("Initial trap-population count timeline")
    axis.set_xlabel("Event time (Myr)")
    axis.set_ylabel("Objects")
    _legend_if_labeled(axis)
    fig.tight_layout()
    paths.append(
        save_figure(
            fig,
            _figure_directory(output_dir, "overview"),
            "population_counts_vs_time",
        )
    )

    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
    if bbh.num:
        axes[0].scatter(pairing_times_myr, bbh.sma_cm[pairing_order] / AU)
        axes[0].set_yscale("log")
        axes[1].scatter(pairing_times_myr, bbh.ecc[pairing_order])
        axes[2].scatter(
            pairing_times_myr,
            bbh.chi1[pairing_order],
            label=r"$\chi_1$",
        )
        axes[2].scatter(
            pairing_times_myr,
            bbh.chi2[pairing_order],
            label=r"$\chi_2$",
        )
        axes[3].scatter(
            pairing_times_myr,
            bbh.th1[pairing_order],
            label=r"$\theta_1$",
        )
        axes[3].scatter(
            pairing_times_myr,
            bbh.th2[pairing_order],
            label=r"$\theta_2$",
        )
        _legend_if_labeled(axes[2])
        _legend_if_labeled(axes[3])
    else:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "No BBHs formed during the selected AGN lifetime",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
    axes[0].set_ylabel("Semimajor axis (AU)")
    axes[1].set_ylabel("Eccentricity")
    axes[2].set_ylabel("Spin magnitude")
    axes[3].set_ylabel("Spin tilt (rad)")
    axes[3].set_xlabel("BBH formation time (Myr)")
    fig.suptitle("BBH formation parameters", y=0.995)
    fig.tight_layout()
    paths.append(
        save_figure(
            fig,
            _figure_directory(output_dir, "bbh_formation"),
            "formation_parameters_vs_time",
        )
    )
    return paths


def _plot_trap_state_diagnostics(
    agn: AGN_properties,
    disk: DiskProfile,
    population: InitializedTrapPopulation,
    output_dir: Path,
) -> list[Path]:
    """Write one direct initialization-state view per figure."""
    ebh = population.embedded_bh
    sbh = population.singles
    pair_timeline = _formed_pair_timeline_data(ebh, population.pairing_timeline)
    reached_indices = _reached_trap_index_data(ebh)
    single_global_indices = reached_indices.global_indices[
        ebh.trap_flag[reached_indices.global_indices] == -2
    ]
    paths: list[Path] = []

    fig, embedded_axis = plt.subplots(figsize=(10, 6))
    reservoir = embedded_axis.scatter(
        ebh.R_cm / disk.r_trap_cm,
        ebh.m_msun,
        c=ebh.t_form_s / Myr,
        cmap="viridis",
        alpha=0.7,
        s=20,
    )
    embedded_axis.axvline(1.0, color="0.2", linewidth=0.8, linestyle="--")
    embedded_axis.set_title("Embedded-BH reservoir")
    embedded_axis.set_xlabel("Initial radius / reference trap radius")
    embedded_axis.set_ylabel("BH mass (Msun)")
    colorbar = fig.colorbar(reservoir, ax=embedded_axis)
    colorbar.set_label("Formation time (Myr)")
    fig.tight_layout()
    paths.append(
        save_figure(
            fig,
            _figure_directory(output_dir, "embedded_bh"),
            "initial_reservoir",
        )
    )

    timeline_height = max(5.0, 0.16 * pair_timeline.first_idx.size + 2.5)
    fig, timeline_axis = plt.subplots(figsize=(12, timeline_height))
    if pair_timeline.first_idx.size:
        order = np.argsort(pair_timeline.formation_s)
        rows = np.arange(order.size)
        migration_start_myr = pair_timeline.migration_start_s[order] / Myr
        arrival_myr = pair_timeline.arrival_s[order] / Myr
        encounter_myr = pair_timeline.encounter_s[order] / Myr
        formation_myr = pair_timeline.formation_s[order] / Myr
        timeline_axis.hlines(
            rows,
            migration_start_myr,
            arrival_myr,
            color="tab:orange",
            linewidth=2.0,
            label="Migration",
        )
        timeline_axis.hlines(
            rows,
            arrival_myr,
            encounter_myr,
            color="tab:purple",
            linewidth=2.0,
            label="Encounter wait",
        )
        timeline_axis.hlines(
            rows,
            encounter_myr,
            formation_myr,
            color="tab:blue",
            linewidth=2.0,
            label="Gas pair-up",
        )
        timeline_axis.scatter(
            arrival_myr,
            rows,
            color="tab:orange",
            s=12,
            label="Reached trap",
        )
        timeline_axis.scatter(
            encounter_myr,
            rows,
            color="tab:purple",
            s=12,
            label="Encounter",
        )
        timeline_axis.scatter(
            formation_myr,
            rows,
            color="tab:blue",
            s=12,
            label="BBH formation",
        )
        timeline_axis.set_ylabel("BBH ordered by formation time")
    else:
        timeline_axis.text(
            0.5,
            0.5,
            "No BBHs formed during the selected AGN lifetime",
            ha="center",
            va="center",
            transform=timeline_axis.transAxes,
        )
        timeline_axis.set_ylabel("BBH index")
    timeline_axis.axvline(
        agn.lifetime_myr,
        color="0.2",
        linewidth=0.8,
        linestyle=":",
        label="AGN lifetime",
    )
    timeline_axis.set_title("Migration, encounter, and BBH-formation phases")
    timeline_axis.set_xlabel("Time (Myr)")
    _legend_if_labeled(timeline_axis)
    fig.tight_layout()
    paths.append(
        save_figure(
            fig,
            _figure_directory(output_dir, "pairing"),
            "formation_phase_timeline",
        )
    )

    residence_height = max(5.0, 0.25 * single_global_indices.size + 2.5)
    fig, single_axis = plt.subplots(figsize=(12, residence_height))
    if single_global_indices.size:
        rows = np.arange(single_global_indices.size)
        arrival_myr = ebh.t_reach_s[single_global_indices] / Myr
        end_myr = np.full(single_global_indices.size, agn.lifetime_myr)
        residence_myr = end_myr - arrival_myr
        single_axis.hlines(
            rows,
            arrival_myr,
            end_myr,
            color="tab:orange",
            linewidth=2.0,
            label="Available single-BH residence",
        )
        arrival_points = single_axis.scatter(
            arrival_myr,
            rows,
            c=ebh.m_msun[single_global_indices],
            cmap="plasma",
            s=28,
            label="Trap arrival",
        )
        single_axis.scatter(
            end_myr,
            rows,
            color="0.2",
            marker="|",
            s=90,
            label="AGN end",
        )
        if single_global_indices.size <= 20:
            for row, arrival_time_myr, end_time_myr, duration_myr in zip(
                rows,
                arrival_myr,
                end_myr,
                residence_myr,
            ):
                single_axis.text(
                    float(0.5 * (arrival_time_myr + end_time_myr)),
                    float(row),
                    f"{float(duration_myr):.2f} Myr",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        single_axis.set_yticks(rows)
        single_axis.set_yticklabels(
            reached_indices.local_index_by_global[single_global_indices]
        )
        colorbar = fig.colorbar(arrival_points, ax=single_axis)
        colorbar.set_label("BH mass (Msun)")
        _legend_if_labeled(single_axis)
    else:
        single_axis.set_ylabel("Reached-trap sBH index")
    if sbh.num == 0:
        single_axis.text(
            0.5,
            0.5,
            "No trap singles",
            ha="center",
            va="center",
            transform=single_axis.transAxes,
        )
    single_axis.axvline(
        agn.lifetime_myr,
        color="0.2",
        linewidth=0.8,
        linestyle=":",
    )
    single_axis.set_title("Trap single-BH residence available for later BSI")
    single_axis.set_xlabel("Time (Myr)")
    single_axis.set_ylabel("Arrival-ordered reached-trap sBH index")
    fig.tight_layout()
    paths.append(
        save_figure(
            fig,
            _figure_directory(output_dir, "trap_singles"),
            "residence_timeline",
        )
    )

    bbh = population.binaries
    fig, binary_axis = plt.subplots(figsize=(10, 6))
    binary_axis.scatter(
        bbh.m2_msun,
        bbh.m1_msun,
        c=bbh.t_ini_s / Myr,
        cmap="plasma",
        alpha=0.7,
        s=28,
    )
    binary_axis.set_title("First-generation BBH population")
    binary_axis.set_xlabel("Secondary mass (Msun)")
    binary_axis.set_ylabel("Primary mass (Msun)")
    if bbh.num:
        mass_limits = [
            min(np.min(bbh.m2_msun), np.min(bbh.m1_msun)),
            max(np.max(bbh.m1_msun), np.max(bbh.m2_msun)),
        ]
        binary_axis.plot(
            mass_limits,
            mass_limits,
            color="0.2",
            linewidth=0.8,
            linestyle="--",
        )
        colorbar = fig.colorbar(binary_axis.collections[0], ax=binary_axis)
        colorbar.set_label("BBH formation time (Myr)")
    else:
        binary_axis.text(
            0.5,
            0.5,
            "No BBHs formed",
            ha="center",
            va="center",
            transform=binary_axis.transAxes,
        )
    fig.tight_layout()
    paths.append(
        save_figure(
            fig,
            _figure_directory(output_dir, "bbh_formation"),
            "component_mass_distribution",
        )
    )

    fig, pair_index_axis = plt.subplots(figsize=(8, 8))
    single_local_indices = reached_indices.local_index_by_global[single_global_indices]
    if pair_timeline.first_idx.size:
        first_idx = reached_indices.local_index_by_global[pair_timeline.first_idx]
        second_idx = reached_indices.local_index_by_global[pair_timeline.second_idx]
        formation_myr = pair_timeline.formation_s / Myr
        lower_idx = np.minimum(first_idx, second_idx)
        upper_idx = np.maximum(first_idx, second_idx)
        pair_index_map = pair_index_axis.scatter(
            lower_idx,
            upper_idx,
            c=formation_myr,
            cmap="cividis",
            s=24,
        )
        colorbar = fig.colorbar(pair_index_map, ax=pair_index_axis)
        colorbar.set_label("BBH formation time (Myr)")
    if single_local_indices.size:
        pair_index_axis.scatter(
            single_local_indices,
            single_local_indices,
            color="tab:orange",
            marker="x",
            s=38,
            label="Trap single (no partner)",
        )
        _legend_if_labeled(pair_index_axis)
    else:
        if pair_timeline.first_idx.size == 0:
            pair_index_axis.text(
                0.5,
                0.5,
                "No BHs reached the trap during the selected AGN lifetime",
                ha="center",
                va="center",
                transform=pair_index_axis.transAxes,
            )
    reached_count = reached_indices.global_indices.size
    if reached_count:
        pair_index_axis.plot(
            [0, reached_count - 1],
            [0, reached_count - 1],
            color="0.2",
            linewidth=0.8,
            linestyle="--",
        )
        pair_index_axis.set_xlim(-0.5, reached_count - 0.5)
        pair_index_axis.set_ylim(-0.5, reached_count - 0.5)
    pair_index_axis.set_title("Reached-trap sBH pair-up map")
    pair_index_axis.set_xlabel("Lower arrival-ordered sBH index i")
    pair_index_axis.set_ylabel("Higher partner index j; diagonal x = single")
    pair_index_axis.text(
        0.02,
        0.98,
        "Indices include only BHs that reached the trap.\n"
        "Each upper-triangle point (i, j) means i pairs with j, i < j.",
        ha="left",
        va="top",
        transform=pair_index_axis.transAxes,
    )
    fig.tight_layout()
    paths.append(
        save_figure(
            fig,
            _figure_directory(output_dir, "pairing"),
            "reached_trap_sbh_pair_map",
        )
    )
    return paths


# Diagnostic CLI


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
    paths = _plot_overview_and_bbh_diagnostics(population, output_dir)
    paths.extend(_plot_trap_state_diagnostics(agn, disk, population, output_dir))
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
