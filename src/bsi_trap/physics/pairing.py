"""Explicit phenomenological pairing of active singles near one torque trap."""

from __future__ import annotations

import numpy as np

from bsi_trap.simulation.models import (
    ArrivalTimeline,
    DiskProfile,
    EmbeddedPopulation,
    SimulationState,
)
from bsi_trap.units import G_CGS, MSUN_G


def hill_radius_cm(
    orbital_radius_cm: float,
    m1_msun: float,
    m2_msun: float,
    smbh_mass_msun: float,
) -> float:
    """Return R_H = R [(m1+m2)/(3 M_SMBH)]^(1/3) [cm]."""
    if min(orbital_radius_cm, m1_msun, m2_msun, smbh_mass_msun) <= 0.0:
        raise ValueError("Hill-radius inputs must be positive")
    return float(
        orbital_radius_cm
        * ((m1_msun + m2_msun) / (3.0 * smbh_mass_msun)) ** (1.0 / 3.0)
    )


def pair_delay_s(
    disk: DiskProfile,
    formation_radius_cm: float,
    m1_msun: float,
    m2_msun: float,
    *,
    delay_scale: float = 1.0,
) -> float:
    """Return the labelled phenomenological gas-damping pair delay [s]."""
    if delay_scale <= 0.0:
        raise ValueError("pair delay scale must be positive")
    disk_index = disk.closest_index(formation_radius_cm)
    sound_speed_cm_s = float(disk.sound_speed_cm_s[disk_index])
    density_g_cm3 = float(disk.density_g_cm3[disk_index])
    total_mass_g = (m1_msun + m2_msun) * MSUN_G
    delay_s = (
        3.0
        * sound_speed_cm_s**3
        / (4.0 * np.pi * G_CGS**2 * density_g_cm3 * total_mass_g)
    )
    return float(delay_scale * delay_s)


def select_pair_candidate_ids(
    state: SimulationState,
    arrivals: ArrivalTimeline,
    disk: DiskProfile,
    trigger_bh_id: int,
    *,
    region_scale_heights: float,
) -> tuple[int, int] | None:
    """Select two physical IDs in the same local trap region.

    This is deliberately a phenomenological eligibility rule, not a detailed
    gas-capture model. Candidate indices are always translated through the
    active-ID array before becoming component IDs.
    """
    if region_scale_heights <= 0.0:
        raise ValueError("pairing region scale [H] must be positive")
    active_ids = np.flatnonzero(state.active_single_mask)
    if active_ids.size < 2 or trigger_bh_id not in active_ids:
        return None
    trigger_trap_radius_cm = float(arrivals.trap_radius_cm[trigger_bh_id])
    if not np.isfinite(trigger_trap_radius_cm):
        return None
    disk_index = disk.closest_index(trigger_trap_radius_cm)
    region_width_cm = (
        region_scale_heights
        * float(disk.aspect_ratio[disk_index])
        * trigger_trap_radius_cm
    )
    active_trap_radius_cm = arrivals.trap_radius_cm[active_ids]
    local_active_ids = active_ids[
        np.isfinite(active_trap_radius_cm)
        & (np.abs(active_trap_radius_cm - trigger_trap_radius_cm) <= region_width_cm)
    ]
    if local_active_ids.size < 2:
        return None
    local_positions = state.rng.choice(local_active_ids.size, size=2, replace=False)
    bh1_id = int(local_active_ids[int(local_positions[0])])
    bh2_id = int(local_active_ids[int(local_positions[1])])
    return bh1_id, bh2_id


def formation_radius_cm(
    arrivals: ArrivalTimeline,
    bh1_id: int,
    bh2_id: int,
) -> float:
    """Use the midpoint of two compatible mass-dependent trap assignments."""
    radii_cm = arrivals.trap_radius_cm[[bh1_id, bh2_id]]
    if not np.all(np.isfinite(radii_cm) & (radii_cm > 0.0)):
        raise ValueError("both pairing candidates need finite trap radii [cm]")
    return float(np.mean(radii_cm))


__all__ = [
    "formation_radius_cm",
    "hill_radius_cm",
    "pair_delay_s",
    "select_pair_candidate_ids",
]
