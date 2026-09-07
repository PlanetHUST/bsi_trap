"""pAGN adapter and disk-domain checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from bsi_trap.physics.disk import (
    build_pagn_disk_profile,
    find_toomre_q_outer_radius_cm,
)
from bsi_trap.units import G_CGS, MSUN_G

IMRI_ADAPTER = Path("/home/planet/Phy/imri/src/disk_quantity/pagn_disk_adapter.py")


def test_q_boundary_is_first_outward_stable_to_unstable_crossing() -> None:
    radius_cm = np.array([1.0, 10.0, 100.0, 1000.0])
    toomre_q = np.array([2.0, 1.5, 0.5, 0.2])
    boundary_cm = find_toomre_q_outer_radius_cm(radius_cm, toomre_q)
    expected_log_radius = 1.0 - np.log10(1.5) / (
        np.log10(0.5) - np.log10(1.5)
    )
    assert boundary_cm == pytest.approx(10.0**expected_log_radius)


def test_sirko_profile_is_valid_cgs_disk() -> None:
    smbh_mass_msun = 1.0e6
    profile = build_pagn_disk_profile(
        smbh_mass_msun=smbh_mass_msun,
        alpha=0.01,
        model="sirko",
        n_resolution=96,
        eddington_ratio=0.5,
        b=0,
    )

    assert np.all(np.diff(profile.radius_cm) > 0.0)
    assert profile.radius_cm.shape == profile.temperature_k.shape
    assert np.all(profile.density_g_cm3 > 0.0)
    assert np.all(profile.surface_density_g_cm2 > 0.0)
    assert np.all(profile.sound_speed_cm_s > 0.0)
    assert np.all(profile.aspect_ratio > 0.0)
    np.testing.assert_allclose(
        profile.surface_density_g_cm2,
        2.0 * profile.density_g_cm3 * profile.scale_height_cm,
        rtol=2.0e-15,
    )
    expected_rg_cm = G_CGS * smbh_mass_msun * MSUN_G / (2.99792458e10**2)
    assert profile.gravitational_radius_cm == pytest.approx(
        expected_rg_cm,
        rel=2.0e-12,
    )
    assert profile.schwarzschild_radius_cm == pytest.approx(
        2.0 * profile.gravitational_radius_cm,
        rel=1.0e-12,
    )


@pytest.mark.skipif(not IMRI_ADAPTER.exists(), reason="IMRI donor checkout unavailable")
def test_sirko_profile_matches_imri_reference() -> None:
    spec = importlib.util.spec_from_file_location("imri_pagn_adapter", IMRI_ADAPTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    kwargs = {
        "smbh_mass_msun": 1.0e6,
        "alpha": 0.01,
        "model": "sirko",
        "n_resolution": 96,
        "eddington_ratio": 0.5,
        "b": 0,
    }
    target = build_pagn_disk_profile(**kwargs)
    reference = module.build_pagn_disk_profile(**kwargs)

    pairs = (
        (target.radius_cm, reference.radius_cm),
        (target.temperature_k, reference.temperature_k),
        (target.surface_density_g_cm2, reference.surface_density_g_cm2),
        (target.density_g_cm3, reference.density_g_cm3),
        (target.aspect_ratio, reference.aspect_ratio),
        (target.toomre_q, reference.toomre_q),
    )
    for target_values, reference_values in pairs:
        np.testing.assert_allclose(target_values, reference_values, rtol=1.0e-13)
    assert target.outer_radius_cm == pytest.approx(
        reference.outer_radius_cm,
        rel=1.0e-13,
    )
