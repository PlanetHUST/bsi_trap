from __future__ import annotations

from unittest import TestCase, main, mock

import numpy as np

from bsi_trap.classes.models import (
    AGN_properties,
    BBHEvolverParams,
    BBH_properties,
    DiskProfile,
    Embedded_BH,
    Single_BH_properties,
)
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.constant import constant as const
from bsi_trap.constant.stop_reason import (
    StopReasons,
    StopReasonType,
    stop_reason_name,
    stop_reason_type,
)
from bsi_trap.evolve import gen_evolve
from bsi_trap.evolve.bsi import (
    apply_scheduled_bsi_step,
    schedule_bsi_events,
)
from bsi_trap.evolve.gas_harden import ode_with_gas
from bsi_trap.evolve.gw import ode_gw_only
from bsi_trap.evolve.numeric import para_evo
from bsi_trap.evolve.physical_quantity import (
    compute_eccentricity_at_frequency,
    threshold_radius,
)
from bsi_trap.evolve.tracks import append_bbh_track, finalize_bbh_tracks
from bsi_trap.initialization.trap_population import TrapPopulation
from bsi_trap.runtime.evolution import (
    TrapEvolutionRecord,
    evolve_trap_runtime,
)
from bsi_trap.runtime.remnants import HierarchicalMergerLedger
from bsi_trap.timeline.bsi import construct_bsi_timeline
from bsi_trap.timeline.models import (
    BSI_Timeline,
    EvolutionTimeline,
    PairingTimeline,
)


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


def _wide_disk() -> DiskProfile:
    return DiskProfile(
        radius_cm=np.array([1.0e16, 2.0e16]),
        h_over_r=np.array([0.1, 0.1]),
        sigma_cgs=np.array([1.0e5, 1.0e5]),
        cs_cgs=np.array([1.0e7, 1.0e7]),
        r_in_cm=1.0e16,
        r_out_cm=2.0e16,
        r_trap_cm=1.0e16,
        r_antitrap_cm=2.0e16,
    )


class TestEvolutionTimeline(TestCase):
    def test_bsi_timeline_records_applied_events(self) -> None:
        bbh = BBH_properties(3)
        bbh.t_ini_s[:] = [10.0, 20.0, 30.0]
        bbh.flags.bsi_event_flag[:] = [1, 1, 0]

        timeline = construct_bsi_timeline(
            bbh,
            event_delay_s=np.array([1.0, 2.0, 3.0]),
        )

        np.testing.assert_array_equal(timeline.binary_idx, [0, 1])
        np.testing.assert_allclose(timeline.event_s, [11.0, 22.0])
        np.testing.assert_allclose(timeline.weak_perturbation_s, [11.0, 22.0])

    def test_bsi_timeline_is_empty_without_event_delays(self) -> None:
        bbh = BBH_properties(1)
        bbh.t_ini_s[:] = [10.0]
        bbh.flags.bsi_event_flag[:] = [1]

        self.assertEqual(construct_bsi_timeline(bbh).size, 0)

    def test_bsi_step_changes_binary_before_timeline_records_event(self) -> None:
        params = BBHEvolverParams(
            rlist=np.array([1.0]),
            sigmalist=np.array([1.0]),
            hlist=np.array([0.1]),
            cslist=np.array([1.0]),
        )
        bbh = BBH_properties(1)
        bbh.t_ini_s[:] = [10.0]
        bbh.bsi_timescale_s[:] = [0.5]
        a_initial = 1.0e11
        e_initial = 0.1

        a_after, e_after, _, _, stop_reason = apply_scheduled_bsi_step(
            params,
            bbh,
            bbh.flags,
            0,
            a_initial,
            e_initial,
            20.0,
            10.0,
            elapsed_s=0.0,
            step_s=1.0,
            bsi_active=True,
        )

        self.assertIsNone(stop_reason)
        self.assertLess(a_after, a_initial)
        self.assertGreater(e_after, e_initial)
        self.assertEqual(bbh.flags.bsi_event_flag[0], 1)
        self.assertEqual(bbh.bsi_applied_flag[0], 1)
        self.assertEqual(bbh.bsi_weak_count[0], 1)
        self.assertGreater(bbh.bsi_delta_e_total[0], 0.0)
        self.assertGreater(bbh.bsi_delta_theta_total[0], 0.0)

        timeline = construct_bsi_timeline(bbh)
        np.testing.assert_allclose(timeline.event_s, [10.5])
        np.testing.assert_allclose(timeline.weak_perturbation_s, [10.5])

    def test_bsi_scheduler_samples_live_single_reservoir_delays(self) -> None:
        bbh = BBH_properties(2)
        bbh.m1_msun[:] = [30.0, 25.0]
        bbh.m2_msun[:] = [20.0, 15.0]
        bbh.sma_cm[:] = [1.0e10, 2.0e10]
        bbh.t_ini_s[:] = [0.0, 2.0 * const.Myr]

        sbh = Single_BH_properties(3)
        sbh.m_msun[:] = [10.0, 12.0, 14.0]
        sbh.t_ini_s[:] = [1.0, 1.5, 3.0]
        sbh.t_ini_s *= const.Myr
        agn = AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=10.0)

        scheduled = schedule_bsi_events(
            ModelConfig(model_name="bsi-scheduler-test", bsi_enabled=True),
            agn,
            _disk(),
            bbh,
            sbh,
            np.random.default_rng(1),
        )

        self.assertTrue(np.all(scheduled))
        self.assertTrue(np.all(np.isfinite(bbh.bsi_timescale_s)))
        event_s = bbh.t_ini_s + bbh.bsi_timescale_s
        self.assertTrue(np.all(event_s >= bbh.t_ini_s))
        self.assertTrue(np.all(event_s >= np.min(sbh.t_ini_s)))
        self.assertTrue(np.all(event_s <= agn.lifetime_myr * const.Myr))

    def test_bsi_scheduler_preserves_existing_delays(self) -> None:
        bbh = BBH_properties(1)
        bbh.m1_msun[:] = [30.0]
        bbh.m2_msun[:] = [20.0]
        bbh.sma_cm[:] = [1.0e10]
        bbh.t_ini_s[:] = [0.0]
        bbh.bsi_timescale_s[:] = [0.25 * const.Myr]

        sbh = Single_BH_properties(1)
        sbh.m_msun[:] = [10.0]
        sbh.t_ini_s[:] = [0.0]

        scheduled = schedule_bsi_events(
            ModelConfig(model_name="bsi-preserve-test", bsi_enabled=True),
            AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=10.0),
            _disk(),
            bbh,
            sbh,
            np.random.default_rng(1),
        )

        self.assertFalse(scheduled[0])
        np.testing.assert_allclose(bbh.bsi_timescale_s, [0.25 * const.Myr])

    def test_eccentricity_at_frequency_interpolates_bracketed_step(self) -> None:
        m1_g = 30.0 * const.msun
        m2_g = 20.0 * const.msun
        target_freq = 10.0
        target_a = (
            const.G * (m1_g + m2_g) / (np.pi * target_freq) ** 2
        ) ** (1.0 / 3.0)

        previous_a = 2.0 * target_a
        current_a = 0.5 * target_a
        previous_e = 0.8
        current_e = 0.2

        interpolated = compute_eccentricity_at_frequency(
            m1_g,
            m2_g,
            current_a,
            current_e,
            target_freq,
            previous_a=previous_a,
            previous_e=previous_e,
        )

        self.assertGreater(interpolated, current_e)
        self.assertLess(interpolated, previous_e)

    def test_make_generation_returns_absolute_evolution_timeline(self) -> None:
        bbh = BBH_properties(2)
        bbh.m1_msun[:] = [30.0, 30.0]
        bbh.m2_msun[:] = [20.0, 20.0]
        bbh.t_ini_s[:] = [10.0, 20.0]
        sbh = Single_BH_properties(0)
        evol_params = BBHEvolverParams(
            rlist=np.array([1.0]),
            sigmalist=np.array([1.0]),
            hlist=np.array([0.1]),
            cslist=np.array([1.0]),
        )
        agn = AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=10.0)
        config = ModelConfig(model_name="generation-timeline-test")

        def fake_evolve(evol_params, model_config, agn, disk, bbh, index):
            bbh.flags.merger_flag[index] = 1 if index == 0 else 0
            return [5.0, 7.0][index]

        with mock.patch.object(
            gen_evolve,
            "evolve_single_binary",
            side_effect=fake_evolve,
        ):
            outcome = gen_evolve.make_generation(
                evol_params,
                config,
                agn,
                _disk(),
                bbh,
                sbh,
                rng_orbit=np.random.default_rng(1),
            )

        timeline = outcome.evolution_timeline
        np.testing.assert_allclose(bbh.t_del_s, [5.0, 7.0])
        np.testing.assert_allclose(timeline.end_s, [15.0, 27.0])
        np.testing.assert_allclose(timeline.bbh_merge_s, [15.0])
        self.assertEqual(timeline.bsi_timeline.size, 0)
        self.assertEqual(outcome.hierarchical_mergers.size, 1)

    def test_make_generation_handles_empty_binary_population(self) -> None:
        outcome = gen_evolve.make_generation(
            BBHEvolverParams(
                rlist=np.array([1.0]),
                sigmalist=np.array([1.0]),
                hlist=np.array([0.1]),
                cslist=np.array([1.0]),
            ),
            ModelConfig(model_name="empty-generation-test"),
            AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=10.0),
            _disk(),
            BBH_properties(0),
            Single_BH_properties(0),
            rng_orbit=np.random.default_rng(1),
        )

        self.assertEqual(outcome.evolution_timeline.size, 0)
        self.assertEqual(outcome.evolution_timeline.bsi_timeline.size, 0)
        self.assertEqual(outcome.hierarchical_mergers.size, 0)

    def test_stop_reason_catalog_keeps_codes_but_groups_meaning(self) -> None:
        self.assertEqual(stop_reason_name(StopReasons.STOP_GW_ESCAPE), "STOP_GW_ESCAPE")
        self.assertEqual(
            stop_reason_type(StopReasons.STOP_GW_ESCAPE),
            StopReasonType.HIERARCHICAL_REMNANT,
        )
        self.assertEqual(
            stop_reason_type(StopReasons.STOP_NON_PHYSICAL_ORBIT),
            StopReasonType.BINARY_EVOLUTION,
        )

    def test_bbh_track_append_stages_chunks_until_finalization(self) -> None:
        bbh = BBH_properties(2)

        append_bbh_track(
            bbh,
            binary_idx=0,
            time_s=[1.0, 2.0],
            sma_cm=[10.0, 9.0],
            ecc=[0.2, 0.1],
        )
        append_bbh_track(
            bbh,
            binary_idx=1,
            time_s=[3.0],
            sma_cm=[8.0],
            ecc=[0.05],
        )

        self.assertEqual(bbh.track_time_s.size, 0)
        finalize_bbh_tracks(bbh)

        np.testing.assert_array_equal(bbh.track_binary_idx, [0, 0, 1])
        np.testing.assert_array_equal(bbh.track_step_idx, [0, 1, 0])
        np.testing.assert_allclose(bbh.track_time_s, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(bbh.track_sma_cm, [10.0, 9.0, 8.0])
        np.testing.assert_allclose(bbh.track_ecc, [0.2, 0.1, 0.05])

    def test_para_evo_passes_evolver_params_to_rk4(self) -> None:
        evol_params = BBHEvolverParams(
            rlist=np.array([1.0]),
            sigmalist=np.array([1.0]),
            hlist=np.array([0.1]),
            cslist=np.array([1.0]),
        )

        def ode(params, m1, m2, a, e):
            return -1.0e-3 * a, -1.0e-4 * e

        a_new, e_new, h_new = para_evo(
            evol_params,
            1.0,
            1.0,
            10.0,
            0.1,
            1.0,
            ode,
        )

        self.assertTrue(np.isfinite(a_new))
        self.assertTrue(np.isfinite(e_new))
        self.assertTrue(np.isfinite(h_new))
        self.assertLess(a_new, 10.0)

    def test_para_evo_respects_maximum_physical_step(self) -> None:
        evol_params = BBHEvolverParams(
            rlist=np.array([1.0]),
            sigmalist=np.array([1.0]),
            hlist=np.array([0.1]),
            cslist=np.array([1.0]),
        )

        def ode(params, m1, m2, a, e):
            return -1.0e-12 * a, -1.0e-12 * e

        _, _, h_new = para_evo(
            evol_params,
            1.0,
            1.0,
            10.0,
            0.1,
            1.0e12,
            ode,
            max_step_s=10.0,
        )

        self.assertTrue(np.isfinite(h_new))
        self.assertLessEqual(h_new, 10.0)

    def test_para_evo_reports_accepted_substep_states(self) -> None:
        evol_params = BBHEvolverParams(
            rlist=np.array([1.0]),
            sigmalist=np.array([1.0]),
            hlist=np.array([0.1]),
            cslist=np.array([1.0]),
        )
        accepted_states = []

        def ode(params, m1, m2, a, e):
            return -1.0e-3 * a, -1.0e-4 * e

        def record_state(elapsed_s, sma_cm, eccentricity):
            accepted_states.append((elapsed_s, sma_cm, eccentricity))

        a_new, e_new, h_new = para_evo(
            evol_params,
            1.0,
            1.0,
            10.0,
            0.1,
            1.0,
            ode,
            state_callback=record_state,
        )

        self.assertEqual(len(accepted_states), 2)
        self.assertGreater(accepted_states[0][0], 0.0)
        self.assertGreater(accepted_states[1][0], accepted_states[0][0])
        np.testing.assert_allclose(accepted_states[-1][0], h_new)
        np.testing.assert_allclose(accepted_states[-1][1], a_new)
        np.testing.assert_allclose(accepted_states[-1][2], e_new)


class TestEvolutionProcess(TestCase):
    def _evol_params(self) -> BBHEvolverParams:
        disk = _disk()
        return BBHEvolverParams(
            rlist=disk.radius_cm,
            sigmalist=disk.sigma_cgs,
            hlist=disk.h_over_r,
            cslist=disk.cs_cgs,
        )

    def _near_merger_binary(
        self,
        *,
        sma_factor: float = 10.0,
        ecc: float = 0.1,
    ) -> BBH_properties:
        bbh = BBH_properties(1)
        bbh.m1_msun[:] = [30.0]
        bbh.m2_msun[:] = [20.0]
        bbh.ecc[:] = [ecc]
        bbh.t_ini_s[:] = [0.0]
        r_th = threshold_radius(30.0 * const.msun, 20.0 * const.msun)
        bbh.sma_cm[:] = [sma_factor * r_th]
        bbh.sma_initial_cm[:] = bbh.sma_cm
        bbh.ecc_initial[:] = bbh.ecc
        return bbh

    def test_gw_ode_hardens_and_circularizes(self) -> None:
        params = self._evol_params()
        m1 = 30.0 * const.msun
        m2 = 20.0 * const.msun
        a = 1.0e10
        e = 0.3

        da_dt, de_dt = ode_gw_only(params, m1, m2, a, e)

        self.assertLess(da_dt, 0.0)
        self.assertLess(de_dt, 0.0)

    def test_gas_ode_adds_extra_hardening(self) -> None:
        params = self._evol_params()
        agn = AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=10.0)
        disk = _disk()
        m1 = 30.0 * const.msun
        m2 = 20.0 * const.msun
        a = 1.0e10
        e = 0.3

        da_gw, _ = ode_gw_only(params, m1, m2, a, e)
        da_gas, de_gas = ode_with_gas(params, m1, m2, a, e, agn, disk)

        self.assertLess(da_gas, da_gw)
        self.assertTrue(np.isfinite(de_gas))

    def test_without_gas_evolves_binary_to_merger(self) -> None:
        bbh = self._near_merger_binary()
        outcome = gen_evolve.make_generation(
            self._evol_params(),
            ModelConfig(model_name="gw-evolution-test", gas_mode="without_gas"),
            AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=10.0),
            _disk(),
            bbh,
            Single_BH_properties(0),
            rng_orbit=np.random.default_rng(1),
        )
        timeline = outcome.evolution_timeline

        self.assertEqual(timeline.size, 1)
        self.assertEqual(bbh.flags.merger_flag[0], 1)
        self.assertEqual(outcome.hierarchical_mergers.size, 1)
        self.assertGreater(bbh.t_del_s[0], 0.0)
        self.assertGreater(bbh.mrem_msun[0], 0.0)
        self.assertGreaterEqual(bbh.vk_cgs[0], 0.0)
        self.assertLess(bbh.sma_cm[0], bbh.sma_initial_cm[0])
        self.assertLess(bbh.ecc[0], bbh.ecc_initial[0])
        np.testing.assert_allclose(timeline.end_s, bbh.t_ini_s + bbh.t_del_s)
        self.assertGreaterEqual(bbh.track_time_s.size, 3)
        np.testing.assert_allclose(bbh.track_sma_cm[0], bbh.sma_initial_cm[0])
        np.testing.assert_allclose(bbh.track_ecc[0], bbh.ecc_initial[0])
        np.testing.assert_allclose(bbh.track_sma_cm[-1], bbh.sma_cm[0])
        np.testing.assert_allclose(bbh.track_ecc[-1], bbh.ecc[0])
        self.assertTrue(np.all(np.diff(bbh.track_time_s) >= 0.0))
        self.assertTrue(np.all(np.diff(bbh.track_sma_cm) <= 0.0))
        self.assertTrue(np.all(np.diff(bbh.track_ecc) <= 0.0))

    def test_with_gas_uses_evolution_path_and_records_continuation(self) -> None:
        bbh = self._near_merger_binary()
        outcome = gen_evolve.make_generation(
            self._evol_params(),
            ModelConfig(model_name="gas-evolution-test", gas_mode="with_gas"),
            AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=10.0),
            _disk(),
            bbh,
            Single_BH_properties(0),
            rng_orbit=np.random.default_rng(1),
        )
        timeline = outcome.evolution_timeline

        self.assertEqual(timeline.size, 1)
        self.assertEqual(bbh.flags.merger_flag[0], 1)
        self.assertEqual(bbh.flags.continue_ng_flag[0], 1)
        self.assertEqual(outcome.hierarchical_mergers.recycled_count, 1)
        self.assertGreater(bbh.t_del_s[0], 0.0)
        self.assertLess(bbh.sma_cm[0], bbh.sma_initial_cm[0])
        self.assertGreaterEqual(bbh.track_time_s.size, 3)
        np.testing.assert_allclose(bbh.track_sma_cm[0], bbh.sma_initial_cm[0])
        np.testing.assert_allclose(bbh.track_ecc[0], bbh.ecc_initial[0])
        np.testing.assert_allclose(bbh.track_sma_cm[-1], bbh.sma_cm[0])
        np.testing.assert_allclose(bbh.track_ecc[-1], bbh.ecc[0])
        self.assertTrue(np.all(np.diff(bbh.track_time_s) >= 0.0))
        self.assertTrue(np.all(np.diff(bbh.track_sma_cm) <= 0.0))

    def test_weak_bsi_records_discrete_track_jump(self) -> None:
        bbh = self._near_merger_binary()
        bbh.bsi_timescale_s[:] = [0.0]

        gen_evolve.make_generation(
            self._evol_params(),
            ModelConfig(
                model_name="bsi-track-test",
                gas_mode="without_gas",
                bsi_enabled=True,
            ),
            AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=10.0),
            _disk(),
            bbh,
            Single_BH_properties(0),
            rng_orbit=np.random.default_rng(1),
        )

        self.assertEqual(bbh.flags.bsi_event_flag[0], 1)
        self.assertGreaterEqual(bbh.track_time_s.size, 4)
        np.testing.assert_allclose(bbh.track_time_s[:2], [0.0, 0.0])
        self.assertLess(bbh.track_sma_cm[1], bbh.track_sma_cm[0])
        self.assertGreater(bbh.track_ecc[1], bbh.track_ecc[0])

    def test_generation_schedules_bsi_from_live_singles(self) -> None:
        bbh = self._near_merger_binary(sma_factor=1.0e3)
        sbh = Single_BH_properties(1)
        sbh.m_msun[:] = [10.0]
        sbh.t_ini_s[:] = [0.0]

        gen_evolve.make_generation(
            self._evol_params(),
            ModelConfig(
                model_name="live-bsi-generation-test",
                gas_mode="without_gas",
                bsi_enabled=True,
            ),
            AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=10.0),
            _disk(),
            bbh,
            sbh,
            rng_orbit=np.random.default_rng(1),
        )

        self.assertTrue(np.isfinite(bbh.bsi_timescale_s[0]))
        self.assertEqual(bbh.flags.bsi_event_flag[0], 1)
        self.assertGreaterEqual(bbh.bsi_weak_count[0], 1)

    def test_gas_bsi_evolves_smoothly_before_scheduled_event(
        self,
    ) -> None:
        bbh = self._near_merger_binary(sma_factor=1.0e5)
        bbh.bsi_timescale_s[:] = [0.1 * const.Myr]
        agn = AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=1.0)

        gen_evolve.make_generation(
            self._evol_params(),
            ModelConfig(
                model_name="gas-bsi-jump-test",
                gas_mode="with_gas",
                bsi_enabled=True,
            ),
            agn,
            _disk(),
            bbh,
            Single_BH_properties(0),
            rng_orbit=np.random.default_rng(1),
        )

        self.assertEqual(bbh.flags.bsi_event_flag[0], 1)
        self.assertEqual(bbh.flags.merger_flag[0], 1)
        event_time_s = 0.1 * const.Myr
        pre_event_indices = np.flatnonzero(
            (bbh.track_time_s > 0.0) & (bbh.track_time_s < event_time_s)
        )
        self.assertGreater(pre_event_indices.size, 0)
        event_indices = np.flatnonzero(np.isclose(bbh.track_time_s, event_time_s))
        self.assertGreaterEqual(event_indices.size, 1)
        pre_event_idx = pre_event_indices[-1]
        bsi_event_idx = event_indices[0]
        self.assertLess(bbh.track_sma_cm[pre_event_idx], bbh.track_sma_cm[0])
        self.assertLess(
            bbh.track_sma_cm[bsi_event_idx],
            bbh.track_sma_cm[pre_event_idx],
        )
        self.assertGreater(
            bbh.track_ecc[bsi_event_idx],
            bbh.track_ecc[pre_event_idx],
        )

    def test_gw_evolution_stops_at_agn_lifetime_without_huge_delay(self) -> None:
        bbh = self._near_merger_binary(sma_factor=1.0e5)
        agn = AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=1.0e-9)
        outcome = gen_evolve.make_generation(
            self._evol_params(),
            ModelConfig(model_name="gw-duty-cap-test", gas_mode="without_gas"),
            agn,
            _disk(),
            bbh,
            Single_BH_properties(0),
            rng_orbit=np.random.default_rng(1),
        )

        duty_cycle_s = agn.lifetime_myr * const.Myr
        self.assertEqual(bbh.flags.merger_flag[0], 0)
        self.assertEqual(
            bbh.flags.stop_reason_flag[0],
            StopReasons.STOP_AGN_LIFETIME,
        )
        self.assertLessEqual(bbh.t_del_s[0], duty_cycle_s)
        np.testing.assert_allclose(outcome.evolution_timeline.end_s, [duty_cycle_s])

    def test_gas_evolution_stops_at_agn_lifetime_without_huge_delay(self) -> None:
        bbh = self._near_merger_binary(sma_factor=1.0e5)
        agn = AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=1.0e-9)
        outcome = gen_evolve.make_generation(
            self._evol_params(),
            ModelConfig(model_name="gas-duty-cap-test", gas_mode="with_gas"),
            agn,
            _disk(),
            bbh,
            Single_BH_properties(0),
            rng_orbit=np.random.default_rng(1),
        )

        duty_cycle_s = agn.lifetime_myr * const.Myr
        self.assertEqual(bbh.flags.merger_flag[0], 0)
        self.assertEqual(
            bbh.flags.stop_reason_flag[0],
            StopReasons.STOP_AGN_LIFETIME,
        )
        self.assertLessEqual(bbh.t_del_s[0], duty_cycle_s)
        np.testing.assert_allclose(outcome.evolution_timeline.end_s, [duty_cycle_s])
        self.assertLessEqual(bbh.track_time_s.size, 3)
        np.testing.assert_allclose(bbh.track_time_s[-1], duty_cycle_s)

    def test_initial_nonphysical_binary_stops_before_integration(self) -> None:
        bbh = self._near_merger_binary()
        bbh.sma_cm[:] = [-1.0]

        outcome = gen_evolve.make_generation(
            self._evol_params(),
            ModelConfig(model_name="invalid-evolution-test", gas_mode="without_gas"),
            AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=10.0),
            _disk(),
            bbh,
            Single_BH_properties(0),
            rng_orbit=np.random.default_rng(1),
        )
        timeline = outcome.evolution_timeline

        self.assertEqual(timeline.size, 1)
        self.assertEqual(outcome.hierarchical_mergers.size, 0)
        self.assertEqual(bbh.t_del_s[0], 0.0)
        self.assertEqual(bbh.flags.merger_flag[0], 0)
        self.assertEqual(
            bbh.flags.stop_reason_flag[0],
            StopReasons.STOP_NON_PHYSICAL_ORBIT,
        )

    def test_initial_too_tight_binary_gets_explicit_stop_reason(self) -> None:
        bbh = self._near_merger_binary(sma_factor=1.05)

        gen_evolve.make_generation(
            self._evol_params(),
            ModelConfig(model_name="tight-evolution-test", gas_mode="without_gas"),
            AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=10.0),
            _disk(),
            bbh,
            Single_BH_properties(0),
            rng_orbit=np.random.default_rng(1),
        )

        self.assertEqual(bbh.t_del_s[0], 0.0)
        self.assertEqual(bbh.flags.merger_flag[0], 0)
        self.assertEqual(
            bbh.flags.stop_reason_flag[0],
            StopReasons.STOP_INITIAL_BINARY_TOO_TIGHT,
        )

    def test_bsi_perturbation_does_not_stop_reduced_bsi_evolution(self) -> None:
        bbh = self._near_merger_binary()
        bbh.bsi_timescale_s[:] = [0.0]

        gen_evolve.make_generation(
            self._evol_params(),
            ModelConfig(
                model_name="bsi-weak-only-test",
                gas_mode="without_gas",
                bsi_enabled=True,
            ),
            AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=10.0),
            _disk(),
            bbh,
            Single_BH_properties(0),
            rng_orbit=np.random.default_rng(1),
        )

        self.assertGreater(bbh.t_del_s[0], 0.0)
        self.assertEqual(bbh.flags.bsi_event_flag[0], 1)
        self.assertGreaterEqual(bbh.bsi_weak_count[0], 1)

    def test_recoil_escape_stops_hierarchical_recycling(self) -> None:
        bbh = BBH_properties(1)
        bbh.m1_msun[:] = [80.0]
        bbh.m2_msun[:] = [10.0]
        bbh.chi1[:] = [0.99]
        bbh.chi2[:] = [0.99]
        bbh.th1[:] = [np.pi / 2.0]
        bbh.th2[:] = [0.0]
        bbh.sma_cm[:] = [1.0e11]
        bbh.ecc[:] = [0.1]
        bbh.t_ini_s[:] = [0.0]
        bbh.flags.merger_flag[:] = [1]

        with mock.patch.object(
            gen_evolve,
            "evolve_single_binary",
            return_value=1.0,
        ):
            outcome = gen_evolve.make_generation(
                self._evol_params(),
                ModelConfig(model_name="escape-test", gas_mode="without_gas"),
                AGN_properties(M_SMBH_msun=1.0e2, lifetime_myr=10.0),
                _disk(),
                bbh,
                Single_BH_properties(0),
                rng_orbit=np.random.default_rng(1),
            )

        self.assertEqual(outcome.hierarchical_mergers.size, 1)
        self.assertEqual(outcome.hierarchical_mergers.escaped_count, 1)
        self.assertEqual(outcome.hierarchical_mergers.recycled_count, 0)
        self.assertEqual(bbh.flags.GWescape_flag[0], 1)
        self.assertEqual(bbh.flags.continue_ng_flag[0], 0)
        self.assertEqual(
            bbh.flags.stop_reason_flag[0],
            StopReasons.STOP_GW_ESCAPE,
        )
        self.assertEqual(
            outcome.evolution_timeline.stop_reason[0],
            StopReasons.STOP_GW_ESCAPE,
        )


class TestTrapRuntimeLoop(TestCase):
    def _embedded_population(
        self,
        masses_msun: list[float],
        reach_times_s: list[float],
    ) -> TrapPopulation:
        embedded_bh = Embedded_BH(len(masses_msun))
        embedded_bh.init_ebh(
            masses_msun,
            np.zeros(len(masses_msun), dtype=float),
            np.full(len(masses_msun), 1.0e16, dtype=float),
        )
        embedded_bh.t_reach_s[:] = reach_times_s
        embedded_bh.r_trap_target_cm[:] = 1.0e16
        embedded_bh.r_pair_cm[:] = 1.0e16
        embedded_bh.trap_flag[:] = -2
        singles = Single_BH_properties(len(masses_msun))
        singles.m_msun[:] = masses_msun
        singles.t_ini_s[:] = reach_times_s
        singles.r_pair_cm[:] = 1.0e16
        return TrapPopulation(
            embedded_bh=embedded_bh,
            binaries=BBH_properties(0),
            singles=singles,
            pairing_timeline=PairingTimeline.empty(),
        )

    def test_runtime_recomputes_pairing_after_remnant_arrival(
        self,
    ) -> None:
        initial_population = self._embedded_population(
            masses_msun=[30.0, 20.0, 10.0],
            reach_times_s=[0.0, 0.0, 0.0],
        )

        remnant_singles = Single_BH_properties(1)
        remnant_singles.m_msun[:] = [45.0]
        remnant_singles.t_ini_s[:] = [5.0]
        remnant_singles.chi[:] = [0.7]
        remnant_singles.theta[:] = [0.0]
        remnant_singles.generation[:] = [1]

        first_timeline = EvolutionTimeline(
            binary_idx=np.array([0], dtype=np.intp),
            start_s=np.array([1.0]),
            delay_s=np.array([4.0]),
            end_s=np.array([5.0]),
            merger_s=np.array([5.0]),
            merged=np.array([True]),
            stop_reason=np.array([0], dtype=np.int64),
            bsi_timeline=BSI_Timeline.empty(),
        )
        first_ledger = HierarchicalMergerLedger(
            source_binary_idx=np.array([0], dtype=np.intp),
            merger_s=np.array([5.0]),
            remnant_mass_msun=np.array([45.0]),
            remnant_spin=np.array([0.7]),
            kick_velocity_cgs=np.array([1.0e6]),
            outer_semimajor_cm=np.array([1.0e12]),
            retained=np.array([True]),
            escaped=np.array([False]),
            trap_overloaded=np.array([False]),
            next_generation_singles=remnant_singles,
        )
        second_timeline = EvolutionTimeline(
            binary_idx=np.array([0], dtype=np.intp),
            start_s=np.array([6.0]),
            delay_s=np.array([2.0]),
            end_s=np.array([8.0]),
            merger_s=np.array([np.nan]),
            merged=np.array([False]),
            stop_reason=np.array([0], dtype=np.int64),
            bsi_timeline=BSI_Timeline.empty(),
        )

        pairing_inputs: list[np.ndarray] = []

        def fake_pairing(embedded_bh, *args, max_events=None, **kwargs):
            pairing_inputs.append(np.array(embedded_bh.m_msun, copy=True))
            self.assertEqual(max_events, 1)
            if np.array_equal(embedded_bh.m_msun, [30.0, 20.0, 10.0]):
                embedded_bh.trap_flag[:] = [1, 0, -2]
                embedded_bh.t_pair_s[:] = [1.0, 1.0, np.nan]
                return PairingTimeline(
                    first_idx=np.array([0], dtype=np.intp),
                    second_idx=np.array([1], dtype=np.intp),
                    later_arrival_s=np.array([0.0]),
                    encounter_s=np.array([0.5]),
                    pair_delay_s=np.array([0.5]),
                    formation_s=np.array([1.0]),
                )
            if np.array_equal(embedded_bh.m_msun, [10.0]):
                embedded_bh.trap_flag[:] = [-2]
                embedded_bh.t_pair_s[:] = [np.nan]
                return PairingTimeline.empty()
            if embedded_bh.num == 0:
                return PairingTimeline.empty()
            embedded_bh.trap_flag[:] = [1, 0]
            embedded_bh.t_pair_s[:] = [6.0, 6.0]
            return PairingTimeline(
                first_idx=np.array([0], dtype=np.intp),
                second_idx=np.array([1], dtype=np.intp),
                later_arrival_s=np.array([5.0]),
                encounter_s=np.array([5.5]),
                pair_delay_s=np.array([0.5]),
                formation_s=np.array([6.0]),
            )

        def fake_advance(**kwargs):
            active = kwargs["active"]
            bbh = active.population.binaries
            if active.record_index == 1:
                bbh.t_del_s[:] = [4.0]
                bbh.flags.merger_flag[:] = [1]
                return 5.0
            bbh.t_del_s[:] = [2.0]
            bbh.flags.stop_reason_flag[:] = [StopReasons.STOP_AGN_LIFETIME]
            return 8.0

        def fake_finalize(**kwargs):
            active = kwargs["active"]
            if active.record_index == 1:
                return TrapEvolutionRecord(
                    record_index=1,
                    generation=1,
                    population=active.population,
                    outcome=gen_evolve.GenerationEvolutionResult(
                        first_timeline,
                        first_ledger,
                    ),
                )
            return TrapEvolutionRecord(
                record_index=2,
                generation=2,
                population=active.population,
                outcome=gen_evolve.GenerationEvolutionResult(
                    second_timeline,
                    HierarchicalMergerLedger.empty(),
                ),
            )

        with (
            mock.patch(
                "bsi_trap.runtime.evolution.construct_pairing_up_timeline",
                side_effect=fake_pairing,
            ) as pairing_mock,
            mock.patch(
                "bsi_trap.runtime.evolution.advance_bbh_segment",
                side_effect=fake_advance,
            ),
            mock.patch(
                "bsi_trap.runtime.evolution._finalize_active_binary",
                side_effect=fake_finalize,
            ),
        ):
            run = evolve_trap_runtime(
                Evolparams=BBHEvolverParams(
                    rlist=np.array([1.0]),
                    sigmalist=np.array([1.0]),
                    hlist=np.array([0.1]),
                    cslist=np.array([1.0]),
                ),
                model_config=ModelConfig(model_name="global-event-test"),
                agn=AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=10.0),
                disk=_wide_disk(),
                initial_population=initial_population,
                rng_pair_up=np.random.default_rng(1),
                rng_orbit=np.random.default_rng(2),
                rng_spin=np.random.default_rng(3),
            )

        self.assertEqual(run.generation_count, 2)
        self.assertGreaterEqual(pairing_mock.call_count, 3)
        np.testing.assert_allclose(run.pairing_event_s, [1.0, 6.0])
        np.testing.assert_allclose(pairing_inputs[0], [30.0, 20.0, 10.0])
        np.testing.assert_allclose(pairing_inputs[1], [10.0])
        np.testing.assert_allclose(pairing_inputs[2], [10.0, 45.0])
        second_bbh = run.records[1].population.binaries
        np.testing.assert_allclose(second_bbh.m1_msun, [45.0])
        np.testing.assert_allclose(second_bbh.m2_msun, [10.0])
        np.testing.assert_array_equal(second_bbh.generation, [2])

    def test_runtime_records_empty_state_when_no_pair_forms(self) -> None:
        run = evolve_trap_runtime(
            Evolparams=BBHEvolverParams(
                rlist=np.array([1.0]),
                sigmalist=np.array([1.0]),
                hlist=np.array([0.1]),
                cslist=np.array([1.0]),
            ),
            model_config=ModelConfig(model_name="empty-event-test"),
            agn=AGN_properties(M_SMBH_msun=1.0e6, lifetime_myr=10.0),
            disk=_wide_disk(),
            initial_population=self._embedded_population(
                masses_msun=[30.0],
                reach_times_s=[0.0],
            ),
            rng_pair_up=np.random.default_rng(1),
            rng_orbit=np.random.default_rng(2),
            rng_spin=np.random.default_rng(3),
        )

        self.assertEqual(run.generation_count, 1)
        self.assertEqual(run.evolved_binary_count, 0)
        self.assertEqual(run.final_record.population.binaries.num, 0)
        self.assertEqual(run.final_record.population.singles.num, 1)
        self.assertEqual(run.pairing_event_s.size, 0)


if __name__ == "__main__":
    main()
