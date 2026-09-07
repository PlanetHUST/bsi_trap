"""Deterministic one-dimensional migration driven by the selected torque."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from bsi_trap.physics.migration_torque import (
    TorqueProfile,
    classify_torque_traps,
)
from bsi_trap.simulation.models import DiskProfile
from bsi_trap.units import C_CM_S, G_CGS, MSUN_G


class MigrationStopReason(str, Enum):
    STABLE_TRAP = "STABLE_TRAP"
    DISK_INNER_EDGE = "DISK_INNER_EDGE"
    DISK_OUTER_EDGE = "DISK_OUTER_EDGE"
    AGN_LIFETIME = "AGN_LIFETIME"
    ZERO_TORQUE = "ZERO_TORQUE"
    INVALID_TORQUE = "INVALID_TORQUE"
    NO_ACCESSIBLE_TRAP = "NO_ACCESSIBLE_TRAP"


@dataclass(frozen=True)
class MigrationResult:
    bh_id: int | None
    initial_radius_cm: float
    final_radius_cm: float
    elapsed_time_s: float
    arrival_time_s: float
    reached_stable_trap: bool
    trap_radius_cm: float | None
    stop_reason: MigrationStopReason


def radial_velocity_cm_s(
    torque_cgs: np.ndarray | float,
    migrator_mass_g: float,
    radius_cm: np.ndarray | float,
    smbh_mass_g: float,
) -> np.ndarray:
    """Return dR/dt = 2 Gamma/(m R Omega) in cm/s."""
    radius_values_cm = np.asarray(radius_cm, dtype=float)
    torque_values_cgs = np.asarray(torque_cgs, dtype=float)
    omega_s_inv = np.sqrt(G_CGS * smbh_mass_g / radius_values_cm**3)
    return 2.0 * torque_values_cgs / (
        migrator_mass_g * radius_values_cm * omega_s_inv
    )


def _invalid_result(
    bh_id: int | None,
    initial_radius_cm: float,
    reason: MigrationStopReason,
) -> MigrationResult:
    return MigrationResult(
        bh_id=bh_id,
        initial_radius_cm=float(initial_radius_cm),
        final_radius_cm=float(initial_radius_cm),
        elapsed_time_s=0.0,
        arrival_time_s=np.inf,
        reached_stable_trap=False,
        trap_radius_cm=None,
        stop_reason=reason,
    )


def _local_arrival_tolerance_cm(
    radius_cm: np.ndarray,
    trap_radius_cm: float,
    tolerance_cells: float,
) -> float:
    insertion = int(np.searchsorted(radius_cm, trap_radius_cm))
    widths: list[float] = []
    if insertion > 0:
        widths.append(trap_radius_cm - float(radius_cm[insertion - 1]))
    if insertion < radius_cm.size:
        widths.append(float(radius_cm[insertion]) - trap_radius_cm)
    positive_widths = [width for width in widths if width > 0.0]
    local_width_cm = min(positive_widths) if positive_widths else np.spacing(trap_radius_cm)
    return max(
        float(tolerance_cells) * local_width_cm,
        trap_radius_cm * 1.0e-12,
    )


def _travel_path_cm(
    radius_grid_cm: np.ndarray,
    initial_radius_cm: float,
    target_radius_cm: float,
) -> np.ndarray:
    lower_cm = min(initial_radius_cm, target_radius_cm)
    upper_cm = max(initial_radius_cm, target_radius_cm)
    interior = radius_grid_cm[
        (radius_grid_cm > lower_cm) & (radius_grid_cm < upper_cm)
    ]
    if target_radius_cm > initial_radius_cm:
        return np.concatenate(([initial_radius_cm], interior, [target_radius_cm]))
    return np.concatenate(([initial_radius_cm], interior[::-1], [target_radius_cm]))


def _radius_at_elapsed_time(
    path_radius_cm: np.ndarray,
    interval_time_s: np.ndarray,
    elapsed_time_s: float,
) -> float:
    cumulative_s = np.concatenate(([0.0], np.cumsum(interval_time_s)))
    interval_index = int(np.searchsorted(cumulative_s, elapsed_time_s, side="right") - 1)
    interval_index = min(max(interval_index, 0), interval_time_s.size - 1)
    interval_start_s = cumulative_s[interval_index]
    interval_duration_s = interval_time_s[interval_index]
    if interval_duration_s <= 0.0:
        return float(path_radius_cm[interval_index])
    fraction = (elapsed_time_s - interval_start_s) / interval_duration_s
    return float(
        path_radius_cm[interval_index]
        + fraction
        * (path_radius_cm[interval_index + 1] - path_radius_cm[interval_index])
    )


def _path_interval_times_s(
    path_radius_cm: np.ndarray,
    path_torque_cgs: np.ndarray,
    migrator_mass_g: float,
    smbh_mass_g: float,
) -> np.ndarray:
    speed_cm_s = np.abs(
        radial_velocity_cm_s(
            path_torque_cgs,
            migrator_mass_g,
            path_radius_cm,
            smbh_mass_g,
        )
    )
    inverse_speed_s_cm = 1.0 / speed_cm_s
    return 0.5 * (
        inverse_speed_s_cm[:-1] + inverse_speed_s_cm[1:]
    ) * np.abs(np.diff(path_radius_cm))


def migration_time_between_radii_s(
    radius_grid_cm: np.ndarray,
    torque_cgs: np.ndarray,
    migrator_mass_g: float,
    initial_radius_cm: float,
    final_radius_cm: float,
    smbh_mass_g: float,
) -> float:
    """Integrate a sign-consistent torque path between two specified radii."""
    radius_values_cm = np.asarray(radius_grid_cm, dtype=float)
    torque_values_cgs = np.asarray(torque_cgs, dtype=float)
    if radius_values_cm.ndim != 1 or torque_values_cgs.ndim != 1:
        raise ValueError("radius and torque arrays must be one-dimensional")
    if radius_values_cm.shape != torque_values_cgs.shape or radius_values_cm.size < 2:
        raise ValueError("radius and torque arrays need matching nontrivial shapes")
    if migrator_mass_g <= 0.0 or smbh_mass_g <= 0.0:
        raise ValueError("migrator and SMBH masses [g] must be positive")
    if initial_radius_cm == final_radius_cm:
        return 0.0
    order = np.argsort(radius_values_cm)
    radius_values_cm = radius_values_cm[order]
    torque_values_cgs = torque_values_cgs[order]
    unique_radius_cm, unique_indices = np.unique(radius_values_cm, return_index=True)
    radius_values_cm = unique_radius_cm
    torque_values_cgs = torque_values_cgs[unique_indices]
    if (
        final_radius_cm < radius_values_cm[0]
        or initial_radius_cm < radius_values_cm[0]
        or final_radius_cm > radius_values_cm[-1]
        or initial_radius_cm > radius_values_cm[-1]
    ):
        raise ValueError("migration path lies outside the torque radius domain")
    path_radius_cm = _travel_path_cm(
        radius_values_cm,
        initial_radius_cm,
        final_radius_cm,
    )
    path_torque_cgs = np.interp(
        path_radius_cm,
        radius_values_cm,
        torque_values_cgs,
    )
    direction = np.sign(final_radius_cm - initial_radius_cm)
    if np.any(~np.isfinite(path_torque_cgs)) or np.any(direction * path_torque_cgs <= 0.0):
        raise ValueError("specified migration path crosses zero or opposing torque")
    interval_time_s = _path_interval_times_s(
        path_radius_cm,
        path_torque_cgs,
        migrator_mass_g,
        smbh_mass_g,
    )
    if not np.all(np.isfinite(interval_time_s) & (interval_time_s >= 0.0)):
        raise ValueError("migration path produced invalid interval times")
    return float(np.sum(interval_time_s))


def migrate_object(
    initial_radius_cm: float,
    mass_msun: float,
    disk: DiskProfile,
    torque_profile: TorqueProfile,
    max_time_s: float,
    *,
    start_time_s: float = 0.0,
    bh_id: int | None = None,
    arrival_tolerance_cells: float = 0.25,
    min_trap_separation_cells: float = 10.0,
) -> MigrationResult:
    """Evolve one circular orbit through its torque field to a trap or edge.

    A stable zero is considered reached within ``arrival_tolerance_cells`` of
    the local torque-grid spacing. This avoids requiring exact arrival at a
    fixed point where the idealized migration time can diverge.
    """
    scalar_inputs = (
        initial_radius_cm,
        mass_msun,
        max_time_s,
        start_time_s,
        arrival_tolerance_cells,
    )
    if not all(np.isfinite(value) for value in scalar_inputs):
        raise ValueError("migration scalar inputs must be finite")
    if initial_radius_cm <= 0.0 or mass_msun <= 0.0:
        raise ValueError("initial radius [cm] and mass [Msun] must be positive")
    if max_time_s < 0.0 or start_time_s < 0.0:
        raise ValueError("migration times [s] must be nonnegative")
    if arrival_tolerance_cells <= 0.0:
        raise ValueError("arrival tolerance [grid cells] must be positive")
    if not np.isclose(torque_profile.migrator_mass_msun, mass_msun, rtol=1.0e-12):
        raise ValueError("torque profile belongs to a different migrator mass")

    radius_grid_cm = np.asarray(torque_profile.radius_cm, dtype=float)
    torque_grid_cgs = np.asarray(torque_profile.gamma_selected_cgs, dtype=float)
    if radius_grid_cm.ndim != 1 or torque_grid_cgs.ndim != 1:
        raise ValueError("torque profile arrays must be one-dimensional")
    if radius_grid_cm.shape != torque_grid_cgs.shape or radius_grid_cm.size < 2:
        raise ValueError("torque radius and values need matching nontrivial shapes")
    order = np.argsort(radius_grid_cm)
    radius_grid_cm = radius_grid_cm[order]
    torque_grid_cgs = torque_grid_cgs[order]
    unique_radius_cm, unique_indices = np.unique(radius_grid_cm, return_index=True)
    radius_grid_cm = unique_radius_cm
    torque_grid_cgs = torque_grid_cgs[unique_indices]
    if not np.all(np.isfinite(radius_grid_cm) & (radius_grid_cm > 0.0)):
        raise ValueError("torque radius grid [cm] must be finite and positive")

    domain_inner_cm = max(float(radius_grid_cm[0]), disk.inner_radius_cm)
    domain_outer_cm = min(float(radius_grid_cm[-1]), disk.outer_radius_cm)
    domain_tolerance_cm = max(initial_radius_cm * 1.0e-12, np.spacing(initial_radius_cm))
    if (
        initial_radius_cm < domain_inner_cm - domain_tolerance_cm
        or initial_radius_cm > domain_outer_cm + domain_tolerance_cm
    ):
        raise ValueError("initial radius lies outside the valid disk domain")
    initial_radius_cm = float(np.clip(initial_radius_cm, domain_inner_cm, domain_outer_cm))
    domain_mask = (radius_grid_cm >= domain_inner_cm) & (radius_grid_cm <= domain_outer_cm)
    if not np.all(np.isfinite(torque_grid_cgs[domain_mask])):
        return _invalid_result(bh_id, initial_radius_cm, MigrationStopReason.INVALID_TORQUE)

    initial_torque_cgs = float(
        np.interp(initial_radius_cm, radius_grid_cm, torque_grid_cgs)
    )
    traps = classify_torque_traps(
        radius_grid_cm,
        torque_grid_cgs,
        disk.gravitational_radius_cm,
        min_separation_cells=min_trap_separation_cells,
    )
    stable_radius_cm = traps.stable_radius_cm[
        (traps.stable_radius_cm >= domain_inner_cm)
        & (traps.stable_radius_cm <= domain_outer_cm)
    ]
    if initial_torque_cgs == 0.0:
        for stable_trap_radius_cm in stable_radius_cm:
            tolerance_cm = _local_arrival_tolerance_cm(
                radius_grid_cm,
                float(stable_trap_radius_cm),
                arrival_tolerance_cells,
            )
            if abs(initial_radius_cm - stable_trap_radius_cm) <= tolerance_cm:
                return MigrationResult(
                    bh_id=bh_id,
                    initial_radius_cm=initial_radius_cm,
                    final_radius_cm=float(stable_trap_radius_cm),
                    elapsed_time_s=0.0,
                    arrival_time_s=start_time_s,
                    reached_stable_trap=True,
                    trap_radius_cm=float(stable_trap_radius_cm),
                    stop_reason=MigrationStopReason.STABLE_TRAP,
                )
        return _invalid_result(bh_id, initial_radius_cm, MigrationStopReason.ZERO_TORQUE)
    direction = 1 if initial_torque_cgs > 0.0 else -1
    if direction > 0:
        candidates = stable_radius_cm[stable_radius_cm >= initial_radius_cm]
        trap_radius_cm = float(candidates[0]) if candidates.size else None
        boundary_radius_cm = domain_outer_cm
        boundary_reason = MigrationStopReason.DISK_OUTER_EDGE
    else:
        candidates = stable_radius_cm[stable_radius_cm <= initial_radius_cm]
        trap_radius_cm = float(candidates[-1]) if candidates.size else None
        boundary_radius_cm = domain_inner_cm
        boundary_reason = MigrationStopReason.DISK_INNER_EDGE

    if trap_radius_cm is not None:
        tolerance_cm = _local_arrival_tolerance_cm(
            radius_grid_cm,
            trap_radius_cm,
            arrival_tolerance_cells,
        )
        if abs(initial_radius_cm - trap_radius_cm) <= tolerance_cm:
            return MigrationResult(
                bh_id=bh_id,
                initial_radius_cm=initial_radius_cm,
                final_radius_cm=trap_radius_cm,
                elapsed_time_s=0.0,
                arrival_time_s=start_time_s,
                reached_stable_trap=True,
                trap_radius_cm=trap_radius_cm,
                stop_reason=MigrationStopReason.STABLE_TRAP,
            )
        target_radius_cm = trap_radius_cm - direction * tolerance_cm
        target_radius_cm = float(
            np.clip(target_radius_cm, domain_inner_cm, domain_outer_cm)
        )
        completion_reason = MigrationStopReason.STABLE_TRAP
    else:
        target_radius_cm = boundary_radius_cm
        completion_reason = boundary_reason

    if target_radius_cm == initial_radius_cm:
        return MigrationResult(
            bh_id=bh_id,
            initial_radius_cm=initial_radius_cm,
            final_radius_cm=initial_radius_cm,
            elapsed_time_s=0.0,
            arrival_time_s=np.inf,
            reached_stable_trap=False,
            trap_radius_cm=None,
            stop_reason=boundary_reason,
        )

    path_radius_cm = _travel_path_cm(
        radius_grid_cm,
        initial_radius_cm,
        target_radius_cm,
    )
    path_torque_cgs = np.interp(path_radius_cm, radius_grid_cm, torque_grid_cgs)
    if np.any(~np.isfinite(path_torque_cgs)):
        return _invalid_result(bh_id, initial_radius_cm, MigrationStopReason.INVALID_TORQUE)
    if np.any(direction * path_torque_cgs <= 0.0):
        return _invalid_result(bh_id, initial_radius_cm, MigrationStopReason.ZERO_TORQUE)

    migrator_mass_g = mass_msun * MSUN_G
    smbh_mass_g = disk.gravitational_radius_cm * C_CM_S**2 / G_CGS
    interval_time_s = _path_interval_times_s(
        path_radius_cm,
        path_torque_cgs,
        migrator_mass_g,
        smbh_mass_g,
    )
    if not np.all(np.isfinite(interval_time_s) & (interval_time_s >= 0.0)):
        return _invalid_result(bh_id, initial_radius_cm, MigrationStopReason.INVALID_TORQUE)
    total_time_s = float(np.sum(interval_time_s))
    if total_time_s > max_time_s:
        final_radius_cm = _radius_at_elapsed_time(
            path_radius_cm,
            interval_time_s,
            max_time_s,
        )
        return MigrationResult(
            bh_id=bh_id,
            initial_radius_cm=initial_radius_cm,
            final_radius_cm=final_radius_cm,
            elapsed_time_s=float(max_time_s),
            arrival_time_s=np.inf,
            reached_stable_trap=False,
            trap_radius_cm=trap_radius_cm,
            stop_reason=MigrationStopReason.AGN_LIFETIME,
        )

    reached_trap = completion_reason is MigrationStopReason.STABLE_TRAP
    return MigrationResult(
        bh_id=bh_id,
        initial_radius_cm=initial_radius_cm,
        final_radius_cm=(
            trap_radius_cm if reached_trap and trap_radius_cm is not None else target_radius_cm
        ),
        elapsed_time_s=total_time_s,
        arrival_time_s=start_time_s + total_time_s if reached_trap else np.inf,
        reached_stable_trap=reached_trap,
        trap_radius_cm=trap_radius_cm,
        stop_reason=completion_reason,
    )


__all__ = [
    "MigrationResult",
    "MigrationStopReason",
    "migrate_object",
    "migration_time_between_radii_s",
    "radial_velocity_cm_s",
]
