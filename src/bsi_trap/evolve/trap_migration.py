"""Mass-dependent stable-trap selection and migration for embedded BHs."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np

from bsi_trap.classes.models import AGN_properties, DiskProfile, Embedded_BH
from bsi_trap.disk_profile.migration_torque import (
    classify_torque_crossings,
    find_traps,
)
from bsi_trap.functions.migration_timescale import migration_timescale_torque


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrapMigrationResult:
    """Mass-dependent migration properties required to initialize one trap."""

    migration_times_s: np.ndarray
    target_radii_cm: np.ndarray
    stable_mask: np.ndarray


def find_catalog_median_reference_trap(
    pagn_disk: Any,
    catalog_mass_msun: np.ndarray,
) -> tuple[float, float, Any, float]:
    """Return a shared trap centered on the catalog's median BH mass."""
    reference_mass_msun = float(np.median(catalog_mass_msun))
    r_trap_cm, r_antitrap_cm, torque_profile = find_traps(
        pagn_disk,
        migrator_mass_msun=reference_mass_msun,
    )
    return r_trap_cm, r_antitrap_cm, torque_profile, reference_mass_msun


def resolve_mass_dependent_trap_migration(
    agn: AGN_properties,
    disk: DiskProfile,
    ebh: Embedded_BH,
) -> TrapMigrationResult:
    """Compute migration targets and widths from the initialized BH masses.

    Each distinct BH mass is evaluated against its own torque profile. Objects
    with no stable torque crossing retain infinite migration time and are
    excluded from trap pairing; the numerical fallback from ``find_traps`` is
    never used as their physical target.
    """
    if agn.pagn_disk is None:
        logger.warning(
            "No pAGN disk is attached; using the shared reference trap for all "
            "embedded BHs. Mass-dependent trap filtering is unavailable."
        )
        migration_times_s = migration_timescale_torque(
            m_msun=ebh.m_msun,
            R_ini_cm=ebh.R_cm,
            AGN=agn,
            Disk=disk,
        )
        target_radii_cm = np.full(ebh.num, disk.r_trap_cm, dtype=float)
        stable_mask = np.ones(ebh.num, dtype=bool)
    else:
        migration_times_s = np.full(ebh.num, np.inf, dtype=float)
        target_radii_cm = np.full(ebh.num, np.nan, dtype=float)
        stable_mask = np.zeros(ebh.num, dtype=bool)

        unique_masses_msun, inverse_mass_index = np.unique(
            ebh.m_msun,
            return_inverse=True,
        )
        mass_order = np.argsort(inverse_mass_index)
        group_sizes = np.bincount(inverse_mass_index)
        group_starts = np.concatenate(([0], np.cumsum(group_sizes[:-1])))

        for mass_msun, group_start, group_size in zip(
            unique_masses_msun,
            group_starts,
            group_sizes,
        ):
            mass_indices = mass_order[group_start : group_start + group_size]
            _, _, torque_profile = find_traps(
                agn.pagn_disk,
                migrator_mass_msun=float(mass_msun),
            )
            stable_radii_cm, _ = classify_torque_crossings(torque_profile)
            if stable_radii_cm.size == 0:
                continue

            target_radius_cm = float(np.max(stable_radii_cm))
            target_radii_cm[mass_indices] = target_radius_cm
            stable_mask[mass_indices] = True
            migration_times_s[mass_indices] = migration_timescale_torque(
                m_msun=ebh.m_msun[mass_indices],
                R_ini_cm=ebh.R_cm[mass_indices],
                AGN=agn,
                Disk=disk,
                torque_profile=torque_profile,
                target_radius_cm=target_radius_cm,
            )

    omitted_mask = ~stable_mask
    logger.info(
        "Mass-dependent trap selection: initialized=%d, stable=%d, "
        "omitted_no_stable_trap=%d, center_cm=%.6e",
        ebh.num,
        int(np.count_nonzero(stable_mask)),
        int(np.count_nonzero(omitted_mask)),
        disk.r_trap_cm,
    )
    if np.any(omitted_mask):
        omitted_masses_msun = ebh.m_msun[omitted_mask]
        logger.info(
            "Omitted embedded BH masses without stable traps: n=%d, "
            "min=%.6g Msun, max=%.6g Msun",
            omitted_masses_msun.size,
            np.min(omitted_masses_msun),
            np.max(omitted_masses_msun),
        )
    return TrapMigrationResult(
        migration_times_s=migration_times_s,
        target_radii_cm=target_radii_cm,
        stable_mask=stable_mask,
    )
