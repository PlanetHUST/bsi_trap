from __future__ import annotations

import numpy as np

from bsi_trap.simulation.state.models import (
    BBH_properties,
    Embedded_BH,
    Single_BH_properties,
)


def validate_trap_state(ebh: Embedded_BH, end_time_s: float) -> None:
    """Verify that pairing flags, completion times, and population counts agree."""
    paired = ebh.trap_flag >= 0
    singles = ebh.trap_flag == -2
    unreached = ebh.trap_flag == -1
    if not np.all(paired | singles | unreached):
        raise ValueError("Embedded-BH trap flags contain an unknown state")
    if np.any(np.isfinite(ebh.t_pair_s[singles | unreached])):
        raise ValueError("Only paired BHs may have a pair-completion time")

    for first in np.flatnonzero(paired):
        second = int(ebh.trap_flag[first])
        if second >= ebh.num or ebh.trap_flag[second] != first:
            raise ValueError("Embedded-BH pairing flags must be symmetric indices")
        formation_time_s = ebh.t_pair_s[first]
        if not np.isfinite(formation_time_s) or formation_time_s > end_time_s:
            raise ValueError("Paired BHs must form before the trap end time")
        if not np.isclose(formation_time_s, ebh.t_pair_s[second]):
            raise ValueError("Both members of a BBH pair must share one formation time")
        if formation_time_s < max(ebh.t_reach_s[first], ebh.t_reach_s[second]):
            raise ValueError("BBH formation cannot precede either component arrival")


def init_bbh(ebh: Embedded_BH) -> BBH_properties:
    """Create one BBH entry per symmetric pairing recorded on ``ebh``."""
    first_idx: list[int] = []
    second_idx: list[int] = []

    for first, second in enumerate(ebh.trap_flag):
        if second < 0:
            continue
        if second >= ebh.num or ebh.trap_flag[second] != first:
            raise ValueError("Embedded-BH pairing flags must be symmetric indices")
        if first < second:
            first_idx.append(first)
            second_idx.append(int(second))

    first_idx_array = np.asarray(first_idx, dtype=int)
    second_idx_array = np.asarray(second_idx, dtype=int)
    bbh = BBH_properties(first_idx_array.size)
    if bbh.num == 0:
        return bbh

    first_mass_msun = ebh.m_msun[first_idx_array]
    second_mass_msun = ebh.m_msun[second_idx_array]
    first_is_primary = first_mass_msun >= second_mass_msun

    primary_idx = np.where(first_is_primary, first_idx_array, second_idx_array)
    secondary_idx = np.where(first_is_primary, second_idx_array, first_idx_array)
    formation_times_s = ebh.t_pair_s[first_idx_array]

    if np.any(~np.isfinite(formation_times_s)):
        raise ValueError("Every selected BBH pair must have a finite formation time")
    if not np.allclose(formation_times_s, ebh.t_pair_s[second_idx_array]):
        raise ValueError("Both members of a BBH pair must share one formation time")

    bbh.idi = primary_idx
    bbh.m1_msun = ebh.m_msun[primary_idx]
    bbh.m2_msun = ebh.m_msun[secondary_idx]
    bbh.formation_s = formation_times_s

    primary_has_spin = np.isfinite(ebh.chi[primary_idx]) & np.isfinite(
        ebh.theta[primary_idx]
    )
    secondary_has_spin = np.isfinite(ebh.chi[secondary_idx]) & np.isfinite(
        ebh.theta[secondary_idx]
    )
    bbh.chi1[primary_has_spin] = ebh.chi[primary_idx][primary_has_spin]
    bbh.th1[primary_has_spin] = ebh.theta[primary_idx][primary_has_spin]
    bbh.chi2[secondary_has_spin] = ebh.chi[secondary_idx][secondary_has_spin]
    bbh.th2[secondary_has_spin] = ebh.theta[secondary_idx][secondary_has_spin]
    return bbh


def init_sbh(ebh: Embedded_BH) -> Single_BH_properties:
    """Create the single-BH population from reached, unpaired trap objects."""
    single_idx = np.flatnonzero(ebh.trap_flag == -2)
    sbh = Single_BH_properties(single_idx.size)
    sbh.m_msun = np.array(ebh.m_msun[single_idx], copy=True)
    sbh.available_from_s = np.array(ebh.t_reach_s[single_idx], copy=True)
    sbh.chi = np.array(ebh.chi[single_idx], copy=True)
    sbh.theta = np.array(ebh.theta[single_idx], copy=True)
    sbh.r_pair_cm = np.array(ebh.r_pair_cm[single_idx], copy=True)
    return sbh
