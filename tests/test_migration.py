"""Signed torque migration direction, basin, and timing checks."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bsi_trap.physics.migration import (
    MigrationStopReason,
    migrate_object,
    migration_time_between_radii_s,
    radial_velocity_cm_s,
)
from bsi_trap.units import C_CM_S, G_CGS, MSUN_G, MYR_S

IMRI_SRC = Path("/home/planet/Phy/imri/src")


def _disk_and_torque(
    radius_cm: np.ndarray,
    torque_cgs: np.ndarray,
    *,
    migrator_mass_msun: float = 10.0,
    smbh_mass_msun: float = 1.0e6,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    gravitational_radius_cm = G_CGS * smbh_mass_msun * MSUN_G / C_CM_S**2
    disk = SimpleNamespace(
        inner_radius_cm=float(radius_cm[0]),
        outer_radius_cm=float(radius_cm[-1]),
        gravitational_radius_cm=gravitational_radius_cm,
    )
    torque = SimpleNamespace(
        migrator_mass_msun=migrator_mass_msun,
        radius_cm=radius_cm,
        gamma_selected_cgs=torque_cgs,
    )
    return disk, torque


def test_torque_sign_sets_radial_direction() -> None:
    velocity = radial_velocity_cm_s(
        np.array([1.0e40, -1.0e40]),
        10.0 * MSUN_G,
        np.array([1.0e15, 1.0e15]),
        1.0e6 * MSUN_G,
    )
    assert velocity[0] > 0.0
    assert velocity[1] < 0.0


@pytest.mark.parametrize("initial_radius_cm", [3.0e15, 7.0e15])
def test_stable_zero_attracts_from_both_sides(initial_radius_cm: float) -> None:
    radius_cm = np.linspace(1.0e15, 9.0e15, 401)
    torque_cgs = -1.0e33 * (radius_cm - 5.0e15)
    disk, torque = _disk_and_torque(radius_cm, torque_cgs)
    result = migrate_object(
        initial_radius_cm,
        10.0,
        disk,
        torque,
        1.0e30,
        arrival_tolerance_cells=0.25,
        min_trap_separation_cells=0.0,
    )
    assert result.stop_reason is MigrationStopReason.STABLE_TRAP
    assert result.reached_stable_trap
    assert result.trap_radius_cm == pytest.approx(5.0e15)
    assert result.final_radius_cm == pytest.approx(5.0e15)
    assert result.elapsed_time_s >= 0.0


def test_object_starting_on_stable_zero_is_already_arrived() -> None:
    radius_cm = np.linspace(1.0e15, 9.0e15, 401)
    torque_cgs = -1.0e33 * (radius_cm - 5.0e15)
    disk, torque = _disk_and_torque(radius_cm, torque_cgs)
    result = migrate_object(5.0e15, 10.0, disk, torque, 1.0e10)
    assert result.stop_reason is MigrationStopReason.STABLE_TRAP
    assert result.reached_stable_trap
    assert result.elapsed_time_s == 0.0


def test_anti_trap_repels_to_opposite_disk_edges() -> None:
    radius_cm = np.linspace(1.0e15, 9.0e15, 401)
    torque_cgs = 1.0e33 * (radius_cm - 5.0e15)
    disk, torque = _disk_and_torque(radius_cm, torque_cgs)
    inward = migrate_object(3.0e15, 10.0, disk, torque, 1.0e30)
    outward = migrate_object(7.0e15, 10.0, disk, torque, 1.0e30)
    assert inward.stop_reason is MigrationStopReason.DISK_INNER_EDGE
    assert inward.final_radius_cm == pytest.approx(radius_cm[0])
    assert outward.stop_reason is MigrationStopReason.DISK_OUTER_EDGE
    assert outward.final_radius_cm == pytest.approx(radius_cm[-1])


def test_lifetime_stop_stays_inside_disk() -> None:
    radius_cm = np.linspace(1.0e15, 9.0e15, 401)
    disk, torque = _disk_and_torque(radius_cm, -np.full(radius_cm.shape, 1.0e42))
    result = migrate_object(8.0e15, 10.0, disk, torque, 1.0)
    assert result.stop_reason is MigrationStopReason.AGN_LIFETIME
    assert disk.inner_radius_cm < result.final_radius_cm <= 8.0e15
    assert result.elapsed_time_s == 1.0


@pytest.mark.skipif(not IMRI_SRC.exists(), reason="IMRI donor checkout unavailable")
def test_inward_timing_matches_imri_for_identical_physical_masses() -> None:
    sys.path.insert(0, str(IMRI_SRC))
    try:
        from physical_process import constants as reference_constants
        from physical_process.disk_migration import migration_time_from_torque

        smbh_mass_msun = 1.0e6
        migrator_mass_msun = 10.0
        radius_cm = np.geomspace(1.0e15, 9.0e15, 300)
        torque_cgs = -np.full(radius_cm.shape, 1.0e45)
        disk, torque = _disk_and_torque(
            radius_cm,
            torque_cgs,
            migrator_mass_msun=migrator_mass_msun,
            smbh_mass_msun=smbh_mass_msun,
        )
        target = migrate_object(
            radius_cm[-1],
            migrator_mass_msun,
            disk,
            torque,
            1.0e30,
        )
        target_path_time_s = migration_time_between_radii_s(
            radius_cm,
            torque_cgs,
            migrator_mass_msun * MSUN_G,
            radius_cm[-1],
            radius_cm[0],
            smbh_mass_msun * MSUN_G,
        )
        reference_smbh_msun = (
            smbh_mass_msun
            * MSUN_G
            * G_CGS
            / (reference_constants.msun * reference_constants.G)
        )
        reference_time_myr = migration_time_from_torque(
            radius_cm,
            torque_cgs,
            migrator_mass_msun * MSUN_G,
            radius_cm[-1],
            radius_cm[0],
            reference_smbh_msun,
        )
    finally:
        sys.path.remove(str(IMRI_SRC))

    assert target.stop_reason is MigrationStopReason.DISK_INNER_EDGE
    assert target.elapsed_time_s == pytest.approx(reference_time_myr * MYR_S, rel=2.0e-15)
    assert target_path_time_s == pytest.approx(reference_time_myr * MYR_S, rel=2.0e-15)
