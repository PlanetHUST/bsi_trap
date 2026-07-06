from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main, mock

import h5py
import numpy as np

from cli import simulate_command
from bsi_trap.classes.models import (
    AGN_properties,
    BBH_properties,
    DiskProfile,
    Embedded_BH,
    Single_BH_properties,
)
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.evolve.gen_evolve import GenerationEvolutionResult
from bsi_trap.hdf5_load.timeline_catalog import TimelineCatalog, read_timeline_catalog
from bsi_trap.hierarchical.generations import GenerationRecord, HierarchicalRun
from bsi_trap.hierarchical.remnants import HierarchicalMergerLedger
from bsi_trap.initialization.trap_population import TrapPopulation
from bsi_trap.parallel_compute.single_work import process_single_sample_worker
from bsi_trap.parallel_compute.state import SimulationTask
from bsi_trap.plots.figures_def import available_figures, resolve_figure_function
from bsi_trap.plots.timeline_evolution import (
    _bbh_energy_angular_momentum_coordinates,
    _population_curves,
)
from bsi_trap.storage.contract import HDF5Contract
from bsi_trap.storage.payload import (
    GenerationState,
    generation_data,
    write_sample_manifest,
)
from bsi_trap.storage.save_gen import save_generation_to_file
from bsi_trap.timeline.models import BSI_Timeline, EvolutionTimeline, PairingTimeline
from workflow import sample_run
from workflow.analysis_run import run_analysis_workflow
from workflow.plot_run import run_plotting_workflow


def _h5_group(h5f: h5py.File, path: str) -> h5py.Group:
    node = h5f[path]
    if not isinstance(node, h5py.Group):
        raise AssertionError(f"{path} is not an HDF5 group")
    return node


def _h5_array(h5f: h5py.File, path: str) -> np.ndarray:
    node = h5f[path]
    if not isinstance(node, h5py.Dataset):
        raise AssertionError(f"{path} is not an HDF5 dataset")
    return np.asarray(node[()])


def _disk() -> DiskProfile:
    return DiskProfile(
        radius_cm=np.array([1.0e12, 2.0e12]),
        h_over_r=np.array([0.1, 0.1]),
        sigma_cgs=np.array([1.0e5, 1.0e5]),
        cs_cgs=np.array([1.0e7, 1.0e7]),
        r_in_cm=1.0e12,
        r_out_cm=2.0e12,
        r_trap_cm=1.0e12,
        r_antitrap_cm=2.0e12,
    )


def _sample_state() -> GenerationState:
    agn = AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=10.0)
    agn.alpha = 0.01
    disk = _disk()

    ebh = Embedded_BH(2)
    ebh.init_ebh(
        [10.0, 20.0],
        [1.0, 2.0],
        [1.1e12, 1.2e12],
    )
    ebh.t_mig_s[:] = [3.0, 4.0]
    ebh.t_reach_s[:] = [4.0, 6.0]
    ebh.t_pair_s[:] = [8.0, 8.0]
    ebh.r_trap_target_cm[:] = [1.0e12, 1.0e12]
    ebh.r_pair_cm[:] = [1.0e12, 1.01e12]
    ebh.trap_flag[:] = [1, 0]
    ebh.trap_width_cm = 1.0e11

    bbh = BBH_properties(1)
    bbh.idi[:] = [0]
    bbh.m1_msun[:] = [20.0]
    bbh.m2_msun[:] = [10.0]
    bbh.sma_initial_cm[:] = [1.2e11]
    bbh.ecc_initial[:] = [0.2]
    bbh.sma_cm[:] = [1.0e11]
    bbh.ecc[:] = [0.1]
    bbh.t_ini_s[:] = [8.0]
    bbh.t_del_s[:] = [5.0]
    bbh.flags.merger_flag[:] = [1]
    bbh.track_binary_idx = np.array([0, 0], dtype=np.int64)
    bbh.track_step_idx = np.array([0, 1], dtype=np.int64)
    bbh.track_time_s = np.array([8.0, 13.0])
    bbh.track_sma_cm = np.array([1.2e11, 1.0e11])
    bbh.track_ecc = np.array([0.2, 0.1])

    singles = Single_BH_properties(0)
    pairing = PairingTimeline(
        first_idx=np.array([0], dtype=np.intp),
        second_idx=np.array([1], dtype=np.intp),
        later_arrival_s=np.array([6.0]),
        encounter_s=np.array([7.0]),
        pair_delay_s=np.array([1.0]),
        formation_s=np.array([8.0]),
    )
    evolution = EvolutionTimeline(
        binary_idx=np.array([0], dtype=np.intp),
        start_s=np.array([8.0]),
        delay_s=np.array([5.0]),
        end_s=np.array([13.0]),
        merger_s=np.array([13.0]),
        merged=np.array([True]),
        stop_reason=np.array([0], dtype=np.int64),
        bsi_timeline=BSI_Timeline.empty(),
    )
    return GenerationState(
        agn=agn,
        disk=disk,
        population=TrapPopulation(
            embedded_bh=ebh,
            binaries=bbh,
            singles=singles,
            pairing_timeline=pairing,
        ),
        evolution_timeline=evolution,
    )


def _sample_state_with_lifetime(lifetime_myr: float) -> GenerationState:
    state = _sample_state()
    state.agn.lifetime_myr = lifetime_myr
    return state


class TestWorkflow(TestCase):
    def test_parameter_timeline_maps_orbit_to_energy_angular_momentum_plane(
        self,
    ) -> None:
        j_over_jc, y_log_a = _bbh_energy_angular_momentum_coordinates(
            np.array([1.0e-3, 1.0e-2]) * 3.08568e18,
            np.array([0.0, np.sqrt(1.0 - 0.01)]),
        )

        np.testing.assert_allclose(j_over_jc, [1.0, 0.1])
        np.testing.assert_allclose(y_log_a, [0.0, 1.0])

    def test_figure_registry_resolves_current_plot_functions(self) -> None:
        self.assertIn("bbh_sbh_population_timeline", available_figures("timeline"))
        self.assertTrue(
            callable(resolve_figure_function("bbh_sbh_population_timeline"))
        )
        self.assertTrue(callable(resolve_figure_function("bbh_parameter_timeline")))

    def test_population_curve_removes_bbh_at_merger(self) -> None:
        catalog = TimelineCatalog(
            trap_reach_s=np.array([1.0, 2.0, 3.0, 4.0]),
            trap_reach_sample=np.zeros(4, dtype=np.int64),
            pair_formation_s=np.array([5.0]),
            pair_sample=np.zeros(1, dtype=np.int64),
            merger_s=np.array([8.0]),
            merger_sample=np.zeros(1, dtype=np.int64),
            start_s=np.array([5.0]),
            end_s=np.array([8.0]),
            sma_initial_cm=np.array([1.2e11]),
            sma_final_cm=np.array([1.0e11]),
            ecc_initial=np.array([0.2]),
            ecc_final=np.array([0.1]),
            remnant_mass_msun=np.array([28.0]),
            kick_velocity_cgs=np.array([0.0]),
            generation=np.array([1], dtype=np.int64),
            sample_index=np.array([0], dtype=np.int64),
            duty_cycle_s=np.array([10.0]),
            merged=np.array([True]),
            track_binary_idx=np.empty(0, dtype=np.int64),
            track_time_s=np.empty(0),
            track_sma_cm=np.empty(0),
            track_ecc=np.empty(0),
        )

        time_myr, single_count, binary_count = _population_curves(catalog)

        expected_time_myr = (
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0])
            / 3.1557600e13
        )
        np.testing.assert_allclose(time_myr, expected_time_myr)
        np.testing.assert_array_equal(single_count, [0, 1, 2, 3, 4, 2, 2, 2])
        np.testing.assert_array_equal(binary_count, [0, 0, 0, 0, 0, 1, 0, 0])

    def test_simulate_command_passes_evolution_model_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grid_path = root / "smbh.txt"
            grid_path.write_text("1000000\n", encoding="utf-8")
            args = simulate_command.build_parser().parse_args(
                [
                    "--num-samples",
                    "1",
                    "--max-workers",
                    "1",
                    "--smbh-grid",
                    str(grid_path),
                    "--gas-mode",
                    "with_gas",
                    "--bsi-enabled",
                    "--bsi-eta",
                    "4.0",
                    "--pair-up-eta",
                    "3.0",
                    "--trap-width-scale-heights",
                    "1.5",
                    "--compact-object-scale-height-fraction",
                    "0.5",
                    "--trap-radial-dispersion-scale-heights",
                    "0.1",
                    "--alpha",
                    "0.02",
                    "--max-generations",
                    "3",
                ]
            )

            with mock.patch.object(
                simulate_command,
                "run_simulation_workflow",
                return_value=[],
            ) as workflow_mock:
                simulate_command.run_simulate_command(args)

        model = workflow_mock.call_args.kwargs["models"][0]
        self.assertEqual(model.gas_mode, "with_gas")
        self.assertTrue(model.bsi_enabled)
        self.assertEqual(model.bsi_eta, 4.0)
        self.assertEqual(model.pair_up_eta, 3.0)
        self.assertEqual(model.trap_width_scale_heights, 1.5)
        self.assertEqual(model.compact_object_scale_height_fraction, 0.5)
        self.assertEqual(model.trap_radial_dispersion_scale_heights, 0.1)
        self.assertEqual(model.alpha, 0.02)
        self.assertEqual(model.max_generations, 3)

    def test_generation_data_writes_hdf5_contract(self) -> None:
        config = ModelConfig(model_name="workflow-test")
        state = _sample_state()
        state.population.binaries.bsi_timescale_s[:] = [2.0]

        with tempfile.TemporaryDirectory() as tmp:
            contract = HDF5Contract(root_dir=Path(tmp))
            output_path = save_generation_to_file(
                simulation_type=config.simulation_type,
                sample_index=0,
                generation=1,
                generation_data=generation_data(
                    model_config=config,
                    state=state,
                ),
                contract=contract,
            )

            self.assertTrue(output_path.exists())
            with h5py.File(output_path, "r") as h5f:
                self.assertIn(contract.group_bbh, h5f)
                self.assertIn(contract.group_flags, h5f)
                self.assertIn(contract.group_timescales, h5f)
                self.assertIn(contract.group_bbh_tracks, h5f)
                self.assertIn(contract.group_trap_population, h5f)
                self.assertIn(contract.group_hierarchical_mergers, h5f)
                bbh_group = _h5_group(h5f, contract.group_bbh)
                self.assertIn("mrem", bbh_group)
                self.assertIn("vk_cgs", bbh_group)
                np.testing.assert_allclose(
                    _h5_array(h5f, f"{contract.group_bbh}/sma_initial"),
                    np.array([1.2e11]),
                )
                np.testing.assert_allclose(
                    _h5_array(h5f, f"{contract.group_bbh}/ecc_initial"),
                    np.array([0.2]),
                )
                np.testing.assert_allclose(
                    _h5_array(h5f, f"{contract.group_bbh}/sma_final"),
                    np.array([1.0e11]),
                )
                np.testing.assert_allclose(
                    _h5_array(h5f, f"{contract.group_timescales}/tdel_myr"),
                    np.array([5.0]) / 3.1557600e13,
                )
                np.testing.assert_allclose(
                    _h5_array(h5f, f"{contract.group_timescales}/tmerg_myr"),
                    np.array([13.0]) / 3.1557600e13,
                )
                self.assertTrue(
                    np.isnan(
                        _h5_array(h5f, f"{contract.group_timescales}/t_bsi_myr")[0]
                    )
                )
                np.testing.assert_allclose(
                    _h5_array(h5f, f"{contract.group_bbh_tracks}/sma_cm"),
                    np.array([1.2e11, 1.0e11]),
                )
                np.testing.assert_allclose(
                    _h5_array(h5f, f"{contract.group_bbh_tracks}/ecc"),
                    np.array([0.2, 0.1]),
                )

    def test_timeline_catalog_clips_events_to_agn_duty_cycle(self) -> None:
        config = ModelConfig(model_name="workflow-test")
        state = _sample_state_with_lifetime(lifetime_myr=3.0e-13)

        with tempfile.TemporaryDirectory() as tmp:
            output_path = save_generation_to_file(
                simulation_type=config.simulation_type,
                sample_index=0,
                generation=1,
                generation_data=generation_data(
                    model_config=config,
                    state=state,
                ),
                contract=HDF5Contract(root_dir=Path(tmp)),
            )

            catalog = read_timeline_catalog(output_path)

        duty_cycle_s = state.agn.lifetime_myr * 3.1557600e13
        self.assertLessEqual(np.nanmax(catalog.trap_reach_s), duty_cycle_s)
        self.assertLessEqual(np.nanmax(catalog.pair_formation_s), duty_cycle_s)
        self.assertEqual(catalog.merger_s.size, 0)

    def test_sample_manifest_records_main_workflow_metadata(self) -> None:
        config = ModelConfig(model_name="workflow-test")
        state = _sample_state()

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "workflow-test" / "sample_0000_gen_001.h5"
            manifest_path = write_sample_manifest(
                model_config=config,
                state=state,
                sample_index=0,
                smbh_mass=1.0e6,
                output_file=output_path,
                merger_count=1,
                evolved_binary_count=1,
                generation_count=1,
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["merger_count"], 1)
            self.assertEqual(manifest["evolved_binary_count"], 1)
            self.assertEqual(manifest["trap_population"]["binary_count"], 1)
            self.assertEqual(
                manifest["binary_evolution"]["evolved_binary_count"],
                1,
            )
            self.assertEqual(manifest["hierarchical_mergers"]["remnant_count"], 0)
            self.assertEqual(manifest["generation_files"], [str(output_path)])

    def test_worker_writes_temp_metadata(self) -> None:
        config = ModelConfig(model_name="workflow-test")

        def physics_callback(**kwargs):
            output_file = Path(kwargs["output_file"])
            manifest_file = output_file.with_suffix(".manifest.json")
            return SimpleNamespace(
                output_file=str(output_file),
                manifest_file=str(manifest_file),
                merger_count=2,
                evolved_binary_count=4,
                generation_count=1,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = SimulationTask(
                model_config=config,
                model_index=0,
                sample_index=3,
                smbh_mass=1.0e6,
            )
            result = process_single_sample_worker(
                task,
                base_seed=42,
                output_dir=root / "output",
                temp_dir=root / "temp",
                physics_callback=physics_callback,
            )

            self.assertTrue(result.success)
            metadata_file = result.metadata_file
            self.assertIsNotNone(metadata_file)
            assert metadata_file is not None
            metadata = json.loads(Path(metadata_file).read_text(encoding="utf-8"))
            self.assertEqual(metadata["sample_index"], 3)
            self.assertEqual(metadata["merger_count"], 2)
            self.assertEqual(metadata["evolved_binary_count"], 4)

    def test_physics_sample_runs_binary_evolution(self) -> None:
        config = ModelConfig(model_name="workflow-test")
        agn = AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=10.0)
        disk = _disk()
        bbh = BBH_properties(1)
        bbh.m1_msun[:] = [20.0]
        bbh.m2_msun[:] = [10.0]
        bbh.sma_cm[:] = [1.2e11]
        bbh.ecc[:] = [0.2]
        bbh.t_ini_s[:] = [8.0]
        population = TrapPopulation(
            embedded_bh=Embedded_BH(0),
            binaries=bbh,
            singles=Single_BH_properties(0),
            pairing_timeline=PairingTimeline.empty(),
        )

        def fake_evolve_hierarchical_generations(**kwargs):
            evolved_bbh = kwargs["initial_population"].binaries
            np.testing.assert_allclose(evolved_bbh.sma_initial_cm, [1.2e11])
            np.testing.assert_allclose(evolved_bbh.ecc_initial, [0.2])
            evolved_bbh.sma_cm[:] = [9.0e10]
            evolved_bbh.ecc[:] = [0.1]
            evolved_bbh.t_del_s[:] = [5.0]
            evolved_bbh.flags.merger_flag[:] = [1]
            evolution_timeline = EvolutionTimeline(
                binary_idx=np.array([0], dtype=np.intp),
                start_s=np.array([8.0]),
                delay_s=np.array([5.0]),
                end_s=np.array([13.0]),
                merger_s=np.array([13.0]),
                merged=np.array([True]),
                stop_reason=np.array([0], dtype=np.int64),
                bsi_timeline=BSI_Timeline.empty(),
            )
            return HierarchicalRun(
                records=(
                    GenerationRecord(
                        generation=1,
                        population=population,
                        outcome=GenerationEvolutionResult(
                            evolution_timeline=evolution_timeline,
                            hierarchical_mergers=HierarchicalMergerLedger.empty(),
                        ),
                    ),
                )
            )

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "model_0_sample_0.h5"
            manifest_path = Path(tmp) / "manifest.json"
            with (
                mock.patch.object(
                    sample_run,
                    "make_environment",
                    return_value=(agn, disk),
                ),
                mock.patch.object(
                    sample_run,
                    "make_trap_population",
                    return_value=population,
                ),
                mock.patch.object(
                    sample_run,
                    "evolve_hierarchical_generations",
                    side_effect=fake_evolve_hierarchical_generations,
                ) as generation_loop_mock,
                mock.patch.object(
                    sample_run,
                    "save_generation_to_file",
                    return_value=output_path,
                ),
                mock.patch.object(
                    sample_run,
                    "write_sample_manifest",
                    return_value=manifest_path,
                ),
            ):
                result = sample_run.run_physics_sample(
                    model_config=config,
                    sample_index=0,
                    smbh_mass=1.0e6,
                    output_file=output_path,
                    seed=42,
                )

        self.assertEqual(generation_loop_mock.call_count, 1)
        self.assertEqual(result.evolved_binary_count, 1)
        self.assertEqual(result.merger_count, 1)
        self.assertEqual(result.generation_files, (str(output_path),))
        np.testing.assert_allclose(population.binaries.sma_initial_cm, [1.2e11])
        np.testing.assert_allclose(population.binaries.sma_cm, [9.0e10])

    def test_analysis_and_plot_workflows_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ModelConfig(model_name="workflow-test")
            output_path = save_generation_to_file(
                simulation_type=config.simulation_type,
                sample_index=0,
                generation=1,
                generation_data=generation_data(
                    model_config=config,
                    state=_sample_state(),
                ),
                contract=HDF5Contract(root_dir=root / "samples"),
            )
            summary_path = root / "simulation_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "combined_files": ["combined.h5"],
                        "results": [
                            {
                                "success": True,
                                "sample_index": 0,
                                "smbh_mass": 1.0e6,
                                "output_file": str(output_path),
                                "manifest_file": "manifest.json",
                                "metadata_file": "metadata.json",
                                "merger_count": 2,
                                "evolved_binary_count": 3,
                                "generation_count": 1,
                            },
                            {
                                "success": False,
                                "sample_index": 1,
                                "smbh_mass": 2.0e6,
                                "error": "failed",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            analysis = run_analysis_workflow(
                summary_path=summary_path,
                output_dir=root / "analysis",
            )
            self.assertEqual(analysis["merger_count"], 2)
            self.assertEqual(analysis["evolved_binary_count"], 3)
            self.assertEqual(analysis["failed"], 1)
            self.assertEqual(analysis["sample_manifests"], ["manifest.json"])
            self.assertEqual(analysis["worker_metadata"], ["metadata.json"])
            self.assertTrue((root / "analysis" / "analysis_summary.json").exists())

            manifest = run_plotting_workflow(
                summary_path=summary_path,
                output_dir=root / "plots",
            )
            self.assertTrue(manifest.exists())
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest_data["generation_files_read"], 1)
            figure_names = {
                Path(figure_path).name for figure_path in manifest_data["figures"]
            }
            self.assertEqual(
                figure_names,
                {
                    "bbh_sbh_population_timeline.png",
                    "bbh_parameter_timeline.png",
                },
            )
            for figure_path in manifest_data["figures"]:
                self.assertTrue(Path(figure_path).exists())
                self.assertGreater(Path(figure_path).stat().st_size, 0)


if __name__ == "__main__":
    main()
