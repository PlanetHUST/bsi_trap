from __future__ import annotations

import numpy as np

from bsi_trap.simulation.state.models import Single_BH_properties


def combine_single_populations(
    *populations: Single_BH_properties | None,
) -> Single_BH_properties:
    items = [
        population
        for population in populations
        if population is not None and population.num > 0
    ]

    total = sum(population.num for population in items)
    combined = Single_BH_properties(total)
    if total == 0:
        return combined

    offset = 0
    for population in items:
        next_offset = offset + population.num
        combined.m_msun[offset:next_offset] = population.m_msun
        combined.available_from_s[offset:next_offset] = population.available_from_s
        combined.chi[offset:next_offset] = population.chi
        combined.theta[offset:next_offset] = population.theta
        combined.r_pair_cm[offset:next_offset] = population.r_pair_cm
        combined.generation[offset:next_offset] = population.generation
        offset = next_offset

    order = np.argsort(combined.available_from_s, kind="stable")
    combined.m_msun = combined.m_msun[order]
    combined.available_from_s = combined.available_from_s[order]
    combined.chi = combined.chi[order]
    combined.theta = combined.theta[order]
    combined.r_pair_cm = combined.r_pair_cm[order]
    combined.generation = combined.generation[order]
    return combined


def copy_singles_by_index(
    source_singles: Single_BH_properties,
    indices: np.ndarray,
) -> Single_BH_properties:
    indices = np.asarray(indices, dtype=np.intp)
    singles = Single_BH_properties(indices.size)
    if indices.size == 0:
        return singles

    singles.m_msun = np.asarray(source_singles.m_msun[indices], dtype=np.float64)
    singles.available_from_s = np.asarray(
        source_singles.available_from_s[indices],
        dtype=np.float64,
    )
    singles.chi = np.asarray(source_singles.chi[indices], dtype=np.float64)
    singles.theta = np.asarray(source_singles.theta[indices], dtype=np.float64)
    singles.r_pair_cm = np.asarray(source_singles.r_pair_cm[indices], dtype=np.float64)
    singles.generation = np.asarray(source_singles.generation[indices], dtype=np.int64)
    return singles


def next_single_arrival_s(
    singles: Single_BH_properties,
    current_time_s: float,
    end_time_s: float,
) -> float | None:
    future_arrivals = singles.available_from_s[
        (singles.available_from_s > current_time_s)
        & (singles.available_from_s < end_time_s)
    ]
    if future_arrivals.size == 0:
        return None
    return float(np.min(future_arrivals))
