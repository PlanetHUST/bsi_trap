"""Migration-torque equation, unit, and donor-parity checks."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

from bsi_trap.physics.disk import build_pagn_disk_profile
from bsi_trap.physics.migration_torque import (
    MigrationParameters,
    compute_torque_profile,
    gap_opening_mask,
    radial_log_slope,
    torque_n_m_to_cgs,
)

IMRI_SRC = Path("/home/planet/Phy/imri/src")


def test_torque_conversion_is_exact() -> None:
    np.testing.assert_array_equal(
        torque_n_m_to_cgs(np.array([-2.0, 0.0, 3.0])),
        np.array([-2.0e7, 0.0, 3.0e7]),
    )


def test_radial_log_slope_matches_power_law() -> None:
    radius_cm = np.geomspace(1.0e12, 1.0e18, 80)
    values = 7.0 * radius_cm ** (-1.5)
    slope = radial_log_slope(radius_cm, values, smoothing=0.0)
    np.testing.assert_allclose(slope[3:-3], -1.5, atol=2.0e-12)


def test_gap_diagnostics_do_not_conflate_attenuation_and_type_ii() -> None:
    mass_ratio = 1.0e-4
    aspect_ratio = np.array([0.02, 0.2])
    alpha = 0.01
    mask = gap_opening_mask(mass_ratio, aspect_ratio, alpha, "kii")
    expected_kii = mass_ratio**2 / (alpha * aspect_ratio**2)
    np.testing.assert_array_equal(mask, expected_kii > 11.0)
    attenuation_k = mass_ratio**2 / (alpha * aspect_ratio**5)
    assert not np.allclose(expected_kii, attenuation_k)


@pytest.mark.skipif(not IMRI_SRC.exists(), reason="IMRI donor checkout unavailable")
def test_torque_profile_matches_imri_reference() -> None:
    sys.path.insert(0, str(IMRI_SRC))
    try:
        from disk_quantity.migration_torque import (
            MigrationParameters as ReferenceParameters,
            compute_torque_profile as compute_reference_torque,
        )
        from disk_quantity.pagn_disk_adapter import build_pagn_disk_profile as build_reference_disk
        from physical_process.disk_migration import select_migration_torque_si

        disk_kwargs = {
            "smbh_mass_msun": 1.0e6,
            "alpha": 0.01,
            "model": "sirko",
            "n_resolution": 96,
            "eddington_ratio": 0.5,
            "b": 0,
        }
        target_disk = build_pagn_disk_profile(**disk_kwargs)
        reference_disk = build_reference_disk(**disk_kwargs)
        target = compute_torque_profile(
            target_disk,
            smbh_mass_msun=1.0e6,
            migrator_mass_msun=10.0,
            smoothing=0.005,
            params=MigrationParameters(alpha=0.01),
            gap_opening_condition="kii",
            consider_type_ii_migration=True,
        )
        reference = compute_reference_torque(
            reference_disk.disk,
            migrator_mass_msun=10.0,
            max_radius_rg=None,
            smoothing=0.005,
            params=ReferenceParameters(alpha=0.01),
        )
        selected_reference = select_migration_torque_si(
            reference,
            10.0 / 1.0e6,
            reference_disk.aspect_ratio,
            0.01,
            "kii",
            True,
        )
    finally:
        sys.path.remove(str(IMRI_SRC))

    np.testing.assert_allclose(target.radius_cm, reference.radius_cm, rtol=1.0e-13)
    comparisons = (
        (target.dln_surface_density_dln_radius, reference.dlog_surface_density_dlog_radius),
        (target.dln_temperature_dln_radius, reference.dlog_temperature_dlog_radius),
        (target.dln_pressure_dln_radius, reference.dlog_pressure_dlog_radius),
        (target.gamma_0_cgs, reference.gamma_0_si * 1.0e7),
        (target.gamma_type_i_cgs, reference.gamma_type_i_si * 1.0e7),
        (target.gamma_thermal_cgs, reference.gamma_th_si * 1.0e7),
        (target.gamma_type_ii_cgs, reference.gamma_type_ii_si * 1.0e7),
        (target.gamma_selected_cgs, selected_reference.torque_si * 1.0e7),
    )
    for target_values, reference_values in comparisons:
        np.testing.assert_allclose(target_values, reference_values, rtol=2.0e-12)
    assert target.gamma_selected_cgs.shape == target_disk.radius_cm.shape
    assert np.all(np.isfinite(target.gamma_selected_cgs))
