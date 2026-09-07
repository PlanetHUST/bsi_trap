# pyright: reportMissingImports=false
"""Compare target disk, torque, traps, and migration against the IMRI donor."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

IMRI_SRC = Path("/home/planet/Phy/imri/src")
SMBH_MASSES_MSUN = (1.0e6, 1.0e7)
MIGRATOR_MASSES_MSUN = (10.0, 30.0)
PAGN_MODEL = "sirko"
PAGN_ALPHA = 0.01
PAGN_EDDINGTON_RATIO = 0.5
PAGN_B = 0
DISK_RESOLUTION = 128
TORQUE_SMOOTHING = 0.005
GAP_OPENING_CONDITION = "kii"
MIN_TRAP_SEPARATION_CELLS = 10.0
DISK_RELATIVE_TOLERANCE = 2.0e-12
TORQUE_RELATIVE_TOLERANCE = 5.0e-11
MIGRATION_RELATIVE_TOLERANCE = 5.0e-12


def _max_relative_difference(target: np.ndarray, reference: np.ndarray) -> float:
    target_values = np.asarray(target, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    if target_values.shape != reference_values.shape:
        raise AssertionError(
            f"shape mismatch: target={target_values.shape}, reference={reference_values.shape}"
        )
    if target_values.size == 0:
        return 0.0
    scale = np.maximum(np.abs(reference_values), np.finfo(float).tiny)
    return float(np.max(np.abs(target_values - reference_values) / scale))


def _longest_negative_run(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(valid & np.isfinite(values) & (values < 0.0))
    if indices.size < 2:
        raise AssertionError("no two-cell inward torque segment for migration regression")
    runs = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    run = max(runs, key=lambda candidate: candidate.size)
    if run.size < 2:
        raise AssertionError("inward torque segment is too short for integration")
    return run


def main() -> None:
    if not IMRI_SRC.exists():
        raise FileNotFoundError(f"IMRI donor source not found: {IMRI_SRC}")
    sys.path.insert(0, str(IMRI_SRC))
    try:
        from disk_quantity.migration_torque import (
            MigrationParameters as ReferenceMigrationParameters,
            classify_torque_traps as classify_reference_traps,
            compute_torque_profile as compute_reference_torque,
        )
        from disk_quantity.pagn_disk_adapter import (
            build_pagn_disk_profile as build_reference_disk,
        )
        from physical_process import constants as reference_constants
        from physical_process.disk_migration import (
            migration_time_from_torque as reference_migration_time,
            select_migration_torque_si,
        )

        from bsi_trap.physics.disk import build_pagn_disk_profile
        from bsi_trap.physics.migration import migration_time_between_radii_s
        from bsi_trap.physics.migration_torque import (
            MigrationParameters,
            classify_torque_traps,
            compute_torque_profile,
        )
        from bsi_trap.units import C_CM_S, G_CGS, MSUN_G, MYR_S

        worst_disk = 0.0
        worst_torque = 0.0
        worst_trap_absolute_rg = 0.0
        worst_migration = 0.0
        for smbh_mass_msun in SMBH_MASSES_MSUN:
            disk_kwargs = {
                "smbh_mass_msun": smbh_mass_msun,
                "alpha": PAGN_ALPHA,
                "model": PAGN_MODEL,
                "n_resolution": DISK_RESOLUTION,
                "eddington_ratio": PAGN_EDDINGTON_RATIO,
                "b": PAGN_B,
            }
            target_disk = build_pagn_disk_profile(**disk_kwargs)
            reference_disk = build_reference_disk(**disk_kwargs)
            disk_pairs = (
                (target_disk.radius_cm, reference_disk.radius_cm),
                (target_disk.temperature_k, reference_disk.temperature_k),
                (target_disk.surface_density_g_cm2, reference_disk.surface_density_g_cm2),
                (target_disk.density_g_cm3, reference_disk.density_g_cm3),
                (target_disk.aspect_ratio, reference_disk.aspect_ratio),
                (target_disk.toomre_q, reference_disk.toomre_q),
            )
            disk_difference = max(
                _max_relative_difference(target, reference)
                for target, reference in disk_pairs
            )
            disk_difference = max(
                disk_difference,
                abs(target_disk.outer_radius_cm - reference_disk.outer_radius_cm)
                / reference_disk.outer_radius_cm,
            )
            worst_disk = max(worst_disk, disk_difference)

            for migrator_mass_msun in MIGRATOR_MASSES_MSUN:
                target_torque = compute_torque_profile(
                    target_disk,
                    smbh_mass_msun,
                    migrator_mass_msun,
                    smoothing=TORQUE_SMOOTHING,
                    params=MigrationParameters(alpha=PAGN_ALPHA),
                    gap_opening_condition=GAP_OPENING_CONDITION,
                    consider_type_ii_migration=True,
                )
                reference_torque = compute_reference_torque(
                    reference_disk.disk,
                    migrator_mass_msun,
                    None,
                    smoothing=TORQUE_SMOOTHING,
                    params=ReferenceMigrationParameters(alpha=PAGN_ALPHA),
                )
                selection = select_migration_torque_si(
                    reference_torque,
                    migrator_mass_msun / smbh_mass_msun,
                    reference_disk.aspect_ratio,
                    PAGN_ALPHA,
                    GAP_OPENING_CONDITION,
                    True,
                )
                torque_pairs = (
                    (target_torque.gamma_0_cgs, reference_torque.gamma_0_si * 1.0e7),
                    (target_torque.gamma_type_i_cgs, reference_torque.gamma_type_i_si * 1.0e7),
                    (target_torque.gamma_thermal_cgs, reference_torque.gamma_th_si * 1.0e7),
                    (target_torque.gamma_type_ii_cgs, reference_torque.gamma_type_ii_si * 1.0e7),
                    (target_torque.gamma_selected_cgs, selection.torque_si * 1.0e7),
                )
                torque_difference = max(
                    _max_relative_difference(target, reference)
                    for target, reference in torque_pairs
                )
                worst_torque = max(worst_torque, torque_difference)

                target_traps = classify_torque_traps(
                    target_torque.radius_cm,
                    target_torque.gamma_selected_cgs,
                    target_disk.gravitational_radius_cm,
                    min_separation_cells=MIN_TRAP_SEPARATION_CELLS,
                )
                reference_traps = classify_reference_traps(
                    reference_torque.radius_rg,
                    selection.torque_si,
                    min_separation_cells=MIN_TRAP_SEPARATION_CELLS,
                )
                for target_roots, reference_roots in (
                    (target_traps.stable_radius_rg, reference_traps.stable_radius_rg),
                    (target_traps.anti_trap_radius_rg, reference_traps.anti_trap_radius_rg),
                ):
                    if target_roots.shape != reference_roots.shape:
                        raise AssertionError("target/reference trap counts differ")
                    if target_roots.size:
                        worst_trap_absolute_rg = max(
                            worst_trap_absolute_rg,
                            float(np.max(np.abs(target_roots - reference_roots))),
                        )

                valid_domain = (
                    (target_torque.radius_cm >= target_disk.inner_radius_cm)
                    & (target_torque.radius_cm <= target_disk.outer_radius_cm)
                )
                inward_run = _longest_negative_run(
                    target_torque.gamma_selected_cgs,
                    valid_domain,
                )
                final_radius_cm = float(target_torque.radius_cm[inward_run[0]])
                initial_radius_cm = float(target_torque.radius_cm[inward_run[-1]])
                smbh_mass_g = target_disk.gravitational_radius_cm * C_CM_S**2 / G_CGS
                target_time_s = migration_time_between_radii_s(
                    target_torque.radius_cm,
                    target_torque.gamma_selected_cgs,
                    migrator_mass_msun * MSUN_G,
                    initial_radius_cm,
                    final_radius_cm,
                    smbh_mass_g,
                )
                reference_smbh_mass_msun = (
                    smbh_mass_g
                    * G_CGS
                    / (reference_constants.msun * reference_constants.G)
                )
                reference_time_s = reference_migration_time(
                    reference_torque.radius_cm,
                    selection.torque_si * 1.0e7,
                    migrator_mass_msun * MSUN_G,
                    initial_radius_cm,
                    final_radius_cm,
                    reference_smbh_mass_msun,
                ) * reference_constants.Myr
                migration_difference = abs(target_time_s - reference_time_s) / reference_time_s
                worst_migration = max(worst_migration, migration_difference)
                print(
                    "case: "
                    f"smbh_mass_msun={smbh_mass_msun:.6g} "
                    f"migrator_mass_msun={migrator_mass_msun:.6g} "
                    f"disk_rel={disk_difference:.3e} "
                    f"torque_rel={torque_difference:.3e} "
                    f"migration_rel={migration_difference:.3e} "
                    f"migration_time_myr={target_time_s / MYR_S:.6e}"
                )

        if worst_disk > DISK_RELATIVE_TOLERANCE:
            raise AssertionError(f"disk regression exceeded tolerance: {worst_disk:.3e}")
        if worst_torque > TORQUE_RELATIVE_TOLERANCE:
            raise AssertionError(f"torque regression exceeded tolerance: {worst_torque:.3e}")
        if worst_trap_absolute_rg > 1.0e-7:
            raise AssertionError(
                f"trap regression exceeded absolute tolerance [rg]: {worst_trap_absolute_rg:.3e}"
            )
        if worst_migration > MIGRATION_RELATIVE_TOLERANCE:
            raise AssertionError(
                f"migration regression exceeded tolerance: {worst_migration:.3e}"
            )
        print(f"max_disk_relative_difference: {worst_disk:.6e}")
        print(f"max_torque_relative_difference: {worst_torque:.6e}")
        print(f"max_trap_absolute_difference_rg: {worst_trap_absolute_rg:.6e}")
        print(f"max_migration_relative_difference: {worst_migration:.6e}")
        print("REFERENCE_REGRESSION_OK")
    finally:
        sys.path.remove(str(IMRI_SRC))


if __name__ == "__main__":
    main()
