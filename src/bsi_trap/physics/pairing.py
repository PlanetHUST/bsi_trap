"""One-pass trap pairing logic for the compact demo."""

from __future__ import annotations

import numpy as np

from bsi_trap.simulation.models import (
    ArrivalTimeline,
    DemoConfig,
    DiskProfile,
    EmbeddedPopulation,
    PairingTimeline,
)
from bsi_trap.units import G, MSUN, MYR


def pair_delay_s(disk: DiskProfile, m1_msun: float, m2_msun: float) -> float:
    total_mass_g = (m1_msun + m2_msun) * MSUN
    density_g_cm3 = disk.trap_sigma_g_cm2 / (2.0 * disk.trap_scale_height_cm)
    return float(
        3.0
        * disk.trap_cs_cm_s**3
        / (4.0 * np.pi * G**2 * density_g_cm3 * total_mass_g)
    )


def construct_pairing_timeline(
    config: DemoConfig,
    disk: DiskProfile,
    population: EmbeddedPopulation,
    arrivals: ArrivalTimeline,
) -> tuple[PairingTimeline, np.ndarray]:
    """Pair reached singles in arrival order for one demo generation."""
    end_s = config.agn_lifetime_myr * MYR
    paired_mask = np.zeros(population.size, dtype=bool)
    first_idx: list[int] = []
    second_idx: list[int] = []
    later_arrival_s: list[float] = []
    pair_delay_values_s: list[float] = []
    formation_s: list[float] = []

    for cursor in range(0, arrivals.order.size - 1, 2):
        first = int(arrivals.order[cursor])
        second = int(arrivals.order[cursor + 1])
        later_arrival = float(max(arrivals.arrival_s[first], arrivals.arrival_s[second]))
        delay = (
            pair_delay_s(
                disk,
                float(population.mass_msun[first]),
                float(population.mass_msun[second]),
            )
            * config.pair_delay_scale
        )
        formation = later_arrival + delay
        if formation > end_s:
            continue
        paired_mask[first] = True
        paired_mask[second] = True
        first_idx.append(first)
        second_idx.append(second)
        later_arrival_s.append(later_arrival)
        pair_delay_values_s.append(delay)
        formation_s.append(formation)

    timeline = PairingTimeline(
        first_idx=np.asarray(first_idx, dtype=np.intp),
        second_idx=np.asarray(second_idx, dtype=np.intp),
        later_arrival_s=np.asarray(later_arrival_s, dtype=float),
        pair_delay_s=np.asarray(pair_delay_values_s, dtype=float),
        formation_s=np.asarray(formation_s, dtype=float),
    )
    return timeline, paired_mask


__all__ = ["construct_pairing_timeline", "pair_delay_s"]
