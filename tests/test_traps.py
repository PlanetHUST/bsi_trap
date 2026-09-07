"""Stable migration-trap and anti-trap classification checks."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

from bsi_trap.physics.disk import build_pagn_disk_profile
from bsi_trap.physics.migration_torque import (
    classify_torque_traps,
    compute_torque_profile,
)

IMRI_SRC = Path("/home/planet/Phy/imri/src")


def test_linear_stable_crossing_is_trap() -> None:
    radius_cm = np.linspace(1.0, 9.0, 9)
    torque_cgs = -(radius_cm - 5.0)
    traps = classify_torque_traps(radius_cm, torque_cgs, 1.0, min_separation_cells=0)
    np.testing.assert_allclose(traps.stable_radius_cm, [5.0])
    assert traps.anti_trap_radius_cm.size == 0


def test_linear_unstable_crossing_is_anti_trap() -> None:
    radius_cm = np.linspace(1.0, 9.0, 9)
    torque_cgs = radius_cm - 5.0
    traps = classify_torque_traps(radius_cm, torque_cgs, 1.0, min_separation_cells=0)
    np.testing.assert_allclose(traps.anti_trap_radius_cm, [5.0])
    assert traps.stable_radius_cm.size == 0


def test_no_crossing_returns_empty_arrays() -> None:
    traps = classify_torque_traps(
        np.arange(1.0, 8.0),
        -np.ones(7),
        1.0,
    )
    assert traps.stable_radius_cm.size == 0
    assert traps.anti_trap_radius_cm.size == 0


def test_multiple_crossings_are_ordered_and_classified() -> None:
    radius_cm = np.linspace(1.0, 10.0, 181)
    torque_cgs = -((radius_cm - 3.0) * (radius_cm - 6.0) * (radius_cm - 8.0))
    traps = classify_torque_traps(radius_cm, torque_cgs, 1.0, min_separation_cells=0)
    np.testing.assert_allclose(traps.stable_radius_cm, [3.0, 8.0], atol=0.025)
    np.testing.assert_allclose(traps.anti_trap_radius_cm, [6.0], atol=0.025)


@pytest.mark.skipif(not IMRI_SRC.exists(), reason="IMRI donor checkout unavailable")
def test_physical_trap_classification_matches_imri_reference() -> None:
    sys.path.insert(0, str(IMRI_SRC))
    try:
        from disk_quantity.migration_torque import classify_torque_traps as classify_reference

        disk = build_pagn_disk_profile(
            1.0e6,
            0.01,
            model="sirko",
            n_resolution=128,
            eddington_ratio=0.5,
        )
        profile = compute_torque_profile(
            disk,
            1.0e6,
            10.0,
            smoothing=0.005,
        )
        target = classify_torque_traps(
            profile.radius_cm,
            profile.gamma_selected_cgs,
            disk.gravitational_radius_cm,
            min_separation_cells=10.0,
        )
        reference = classify_reference(
            profile.radius_rg,
            profile.gamma_selected_cgs,
            min_separation_cells=10.0,
        )
    finally:
        sys.path.remove(str(IMRI_SRC))

    np.testing.assert_allclose(target.stable_radius_rg, reference.stable_radius_rg)
    np.testing.assert_allclose(target.anti_trap_radius_rg, reference.anti_trap_radius_rg)
