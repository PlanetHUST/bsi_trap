"""Generate deterministic disk, torque/trap, and migration diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

SMBH_MASS_MSUN = 1.0e6
MIGRATOR_MASS_MSUN = 10.0
PAGN_MODEL = "sirko"
PAGN_ALPHA = 0.01
PAGN_EDDINGTON_RATIO = 0.5
PAGN_B = 0
DISK_RESOLUTION = 256
TORQUE_SMOOTHING = 0.005
GAP_OPENING_CONDITION = "kii"
CONSIDER_TYPE_II = True
MIN_TRAP_SEPARATION_CELLS = 10.0
ARRIVAL_TOLERANCE_CELLS = 0.25
MAX_MIGRATION_TIME_S = 10.0 * 3.15576e13
INITIAL_RADIUS_COUNT = 12
OUTPUT_DIR = Path("report/scientific_diagnostics")


def main(output_dir: Path = OUTPUT_DIR) -> None:
    import matplotlib.pyplot as plt

    from bsi_trap.physics.disk import build_pagn_disk_profile
    from bsi_trap.physics.migration import migrate_object
    from bsi_trap.physics.migration_torque import (
        MigrationParameters,
        classify_torque_traps,
        compute_torque_profile,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    disk = build_pagn_disk_profile(
        smbh_mass_msun=SMBH_MASS_MSUN,
        alpha=PAGN_ALPHA,
        model=PAGN_MODEL,
        n_resolution=DISK_RESOLUTION,
        eddington_ratio=PAGN_EDDINGTON_RATIO,
        b=PAGN_B,
    )
    torque = compute_torque_profile(
        disk,
        smbh_mass_msun=SMBH_MASS_MSUN,
        migrator_mass_msun=MIGRATOR_MASS_MSUN,
        smoothing=TORQUE_SMOOTHING,
        params=MigrationParameters(alpha=PAGN_ALPHA),
        gap_opening_condition=GAP_OPENING_CONDITION,
        consider_type_ii_migration=CONSIDER_TYPE_II,
    )
    traps = classify_torque_traps(
        torque.radius_cm,
        torque.gamma_selected_cgs,
        disk.gravitational_radius_cm,
        min_separation_cells=MIN_TRAP_SEPARATION_CELLS,
    )

    radius_rg = disk.radius_cm / disk.gravitational_radius_cm
    disk_figure, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True)
    disk_fields = (
        (disk.surface_density_g_cm2, r"$\Sigma$ [g cm$^{-2}$]"),
        (disk.density_g_cm3, r"$\rho$ [g cm$^{-3}$]"),
        (disk.temperature_k, "T [K]"),
        (disk.aspect_ratio, "H/R"),
        (disk.toomre_q, "Toomre Q"),
    )
    for axis, (values, label) in zip(axes.flat, disk_fields):
        axis.loglog(radius_rg, values)
        axis.set_ylabel(label)
        axis.grid(True, which="both", alpha=0.25)
    axes.flat[-1].axis("off")
    for axis in axes[-1, :1]:
        axis.set_xlabel(r"R/$r_g$")
    disk_figure.tight_layout()
    disk_path = output_dir / "disk_profiles.png"
    disk_figure.savefig(disk_path, dpi=160)
    plt.close(disk_figure)

    torque_figure, torque_axis = plt.subplots(figsize=(9, 5))
    torque_axis.plot(torque.radius_rg, torque.gamma_type_i_cgs, label="Type I")
    torque_axis.plot(torque.radius_rg, torque.gamma_thermal_cgs, label="thermal")
    torque_axis.plot(torque.radius_rg, torque.gamma_type_ii_cgs, label="Type II")
    torque_axis.plot(
        torque.radius_rg,
        torque.gamma_selected_cgs,
        color="black",
        linewidth=1.5,
        label="selected",
    )
    torque_axis.scatter(traps.stable_radius_rg, np.zeros(traps.stable_radius_rg.size), marker="o", label="stable")
    torque_axis.scatter(traps.anti_trap_radius_rg, np.zeros(traps.anti_trap_radius_rg.size), marker="x", label="anti-trap")
    torque_axis.axvline(
        disk.outer_radius_cm / disk.gravitational_radius_cm,
        color="0.4",
        linestyle="--",
        label="Q outer boundary",
    )
    torque_axis.set(xscale="log", yscale="symlog", xlabel=r"R/$r_g$", ylabel="torque [dyn cm]")
    torque_axis.grid(True, which="both", alpha=0.25)
    torque_axis.legend()
    torque_figure.tight_layout()
    torque_path = output_dir / "torque_and_traps.png"
    torque_figure.savefig(torque_path, dpi=160)
    plt.close(torque_figure)

    initial_radius_cm = np.geomspace(
        disk.inner_radius_cm * 1.01,
        disk.outer_radius_cm / 1.01,
        INITIAL_RADIUS_COUNT,
    )
    migration_results = [
        migrate_object(
            float(radius_cm),
            MIGRATOR_MASS_MSUN,
            disk,
            torque,
            MAX_MIGRATION_TIME_S,
            arrival_tolerance_cells=ARRIVAL_TOLERANCE_CELLS,
            min_trap_separation_cells=MIN_TRAP_SEPARATION_CELLS,
        )
        for radius_cm in initial_radius_cm
    ]
    migration_figure, migration_axis = plt.subplots(figsize=(9, 5))
    migration_axis.plot(
        initial_radius_cm / disk.gravitational_radius_cm,
        np.asarray([result.elapsed_time_s for result in migration_results]) / 3.15576e13,
        marker="o",
    )
    migration_axis.set(
        xscale="log",
        yscale="log",
        xlabel=r"initial R/$r_g$",
        ylabel="elapsed migration time [Myr]",
    )
    migration_axis.grid(True, which="both", alpha=0.25)
    migration_figure.tight_layout()
    migration_path = output_dir / "migration_outcomes.png"
    migration_figure.savefig(migration_path, dpi=160)
    plt.close(migration_figure)

    print(f"disk_model: {PAGN_MODEL}")
    print(f"smbh_mass_msun: {SMBH_MASS_MSUN}")
    print(f"migrator_mass_msun: {MIGRATOR_MASS_MSUN}")
    print(f"gap_opening_condition: {GAP_OPENING_CONDITION}")
    print(f"stable_trap_radius_rg: {traps.stable_radius_rg.tolist()}")
    print(f"anti_trap_radius_rg: {traps.anti_trap_radius_rg.tolist()}")
    print("migration_stop_reasons:", [result.stop_reason.value for result in migration_results])
    print(f"wrote: {disk_path}")
    print(f"wrote: {torque_path}")
    print(f"wrote: {migration_path}")


if __name__ == "__main__":
    main()
