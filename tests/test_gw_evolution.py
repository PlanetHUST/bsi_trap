"""Peters gravitational-wave evolution checks in CGS."""

from __future__ import annotations

import numpy as np
import pytest

from bsi_trap.physics.evolution import (
    binary_integration_stop_radius_cm,
    circular_merger_time_s,
    evolve_binary_peters,
    peters_da_dt_cm_s,
    peters_de_dt_s_inv,
)
from bsi_trap.units import C_CM_S, G_CGS, MSUN_G


def test_circular_merger_time_matches_analytic_formula() -> None:
    m1_msun = 30.0
    m2_msun = 20.0
    sma_cm = 1.0e12
    m1_g = m1_msun * MSUN_G
    m2_g = m2_msun * MSUN_G
    expected_s = (
        5.0
        * C_CM_S**5
        * sma_cm**4
        / (256.0 * G_CGS**3 * m1_g * m2_g * (m1_g + m2_g))
    )
    assert circular_merger_time_s(m1_msun, m2_msun, sma_cm) == pytest.approx(expected_s)


def test_peters_derivative_signs_and_circular_invariance() -> None:
    assert peters_da_dt_cm_s(30.0, 20.0, 1.0e12, 0.4) < 0.0
    assert peters_de_dt_s_inv(30.0, 20.0, 1.0e12, 0.4) < 0.0
    assert peters_de_dt_s_inv(30.0, 20.0, 1.0e12, 0.0) == 0.0


def test_circular_evolution_reaches_documented_stop_without_negative_eccentricity() -> None:
    m1_msun = 30.0
    m2_msun = 20.0
    sma_cm = 1.0e12
    stop_radius_cm = binary_integration_stop_radius_cm(m1_msun, m2_msun)
    full_time_s = circular_merger_time_s(m1_msun, m2_msun, sma_cm)
    result = evolve_binary_peters(m1_msun, m2_msun, sma_cm, 0.0, full_time_s)
    expected_stop_time_s = full_time_s * (1.0 - (stop_radius_cm / sma_cm) ** 4)
    assert result.reached_integration_stop
    assert result.elapsed_time_s == pytest.approx(expected_stop_time_s)
    assert result.final_sma_cm == pytest.approx(stop_radius_cm)
    assert result.final_eccentricity == 0.0


def test_eccentric_integration_converges_and_shrinks() -> None:
    arguments = (30.0, 20.0, 1.0e12, 0.6, 1.0e18)
    coarse = evolve_binary_peters(
        *arguments,
        relative_tolerance=1.0e-7,
        max_step_fraction=0.02,
    )
    fine = evolve_binary_peters(
        *arguments,
        relative_tolerance=1.0e-9,
        max_step_fraction=0.005,
    )
    assert coarse.reached_integration_stop
    assert fine.reached_integration_stop
    assert fine.final_sma_cm < arguments[2]
    assert 0.0 <= fine.final_eccentricity < arguments[3]
    assert coarse.elapsed_time_s == pytest.approx(fine.elapsed_time_s, rel=2.0e-6)


def test_finite_time_budget_returns_unmerged_decreasing_state() -> None:
    result = evolve_binary_peters(30.0, 20.0, 1.0e12, 0.3, 1.0e5)
    assert not result.reached_integration_stop
    assert result.final_sma_cm < result.initial_sma_cm
    assert 0.0 <= result.final_eccentricity <= result.initial_eccentricity
    assert np.isfinite(result.elapsed_time_s)
