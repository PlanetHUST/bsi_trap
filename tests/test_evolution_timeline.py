from __future__ import annotations

from unittest import TestCase, main, mock

import numpy as np

from bsi_trap.classes.models import (
    AGN_properties,
    BBHEvolverParams,
    BBH_properties,
    DiskProfile,
    Single_BH_properties,
)
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.evolve import gen_evolve
from bsi_trap.evolve.bsi import construct_bsi_timeline
from bsi_trap.evolve.numeric import para_evo


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


class TestEvolutionTimeline(TestCase):
    def test_bsi_timeline_uses_recorded_flags_and_delays(self) -> None:
        bbh = BBH_properties(3)
        bbh.t_ini_s[:] = [10.0, 20.0, 30.0]
        bbh.flags.bsi_event_flag[:] = [1, 1, 0]
        bbh.flags.bsi_exchange_flag[:] = [0, 1, 0]
        bbh.flags.bsi_ejection_flag[:] = [0, 0, 1]

        timeline = construct_bsi_timeline(
            bbh,
            event_delay_s=np.array([1.0, 2.0, 3.0]),
        )

        np.testing.assert_array_equal(timeline.binary_idx, [0, 1, 2])
        np.testing.assert_allclose(timeline.event_s, [11.0, 22.0, 33.0])
        np.testing.assert_array_equal(
            timeline.event_type,
            np.array(["weak", "exchange", "ejection"], dtype=np.str_),
        )
        np.testing.assert_allclose(timeline.weak_perturbation_s, [11.0])
        np.testing.assert_allclose(timeline.strong_perturbation_s, [22.0, 33.0])

    def test_bsi_timeline_is_empty_without_event_delays(self) -> None:
        bbh = BBH_properties(1)
        bbh.t_ini_s[:] = [10.0]
        bbh.flags.bsi_event_flag[:] = [1]

        self.assertEqual(construct_bsi_timeline(bbh).size, 0)

    def test_make_generation_returns_absolute_evolution_timeline(self) -> None:
        bbh = BBH_properties(2)
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
            timeline = gen_evolve.make_generation(
                evol_params,
                config,
                agn,
                _disk(),
                bbh,
                sbh,
                rng_orbit=np.random.default_rng(1),
                rng_spin=np.random.default_rng(2),
            )

        np.testing.assert_allclose(bbh.t_del_s, [5.0, 7.0])
        np.testing.assert_allclose(timeline.end_s, [15.0, 27.0])
        np.testing.assert_allclose(timeline.bbh_merge_s, [15.0])
        self.assertEqual(timeline.bsi_timeline.size, 0)

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


if __name__ == "__main__":
    main()
