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
        3.0 * disk.trap_cs_cm_s**3 / (4.0 * np.pi * G**2 * density_g_cm3 * total_mass_g)
    )


def cal_pair_value(
    config: DemoConfig,
    disk: DiskProfile,
    first_mass_msun: float,
    second_mass_msun: float,
    rng: np.random.Generator,
) -> float:
    smbh_mass_g = config.smbh_mass_msun * MSUN
    r_trap_cm = disk.r_trap_cm
    radius_hill = (
        np.sqrt((first_mass_msun + second_mass_msun) * MSUN / smbh_mass_g) * r_trap_cm
    )
    return rng.uniform(0, 1 / radius_hill) * radius_hill


def pairing_kernal(
    current_time_s: float,
    config: DemoConfig,
    disk: DiskProfile,
    population: EmbeddedPopulation,
    arrivals: ArrivalTimeline,
    rng: np.random.Generator,
) -> tuple[PairingTimeline, np.ndarray]:
    end_s = config.agn_lifetime_myr * MYR
    arrived_inds = []

    pair_flag = False
    pair_value = 0

    while (not pair_flag) & (current_time_s < end_s):
        next_arrival_ind = arrivals.select_ind(current_time_s)
        arrived_inds.append(next_arrival_ind)
        if len(arrived_inds) < 2:
            current_time_s = arrivals.arrival_s[next_arrival_ind]
            continue
        first_one = rng.choice(arrived_inds)
        second_one = rng.choice(arrived_inds)
        pair_value = cal_pair_value(
            config,
            disk,
            arrivals.mass_msun[first_one],
            arrivals.mass_msun[second_one],
            rng,
        )
        pair_flag = pair_value > 0.9
        current_time_s += pair_delay_s(
            disk,
        )


__all__ = ["pairing_kernal", "pair_delay_s"]
