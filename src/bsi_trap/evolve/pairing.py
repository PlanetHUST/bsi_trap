"""Event-driven construction of the initial BBH pairing timeline."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bsi_trap.classes.models import DiskProfile, Embedded_BH, TrapContext
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.functions.pairing_physics import pair_up_timescale, sbh_cross_section


IndexArray = npt.NDArray[np.intp]
FloatArray = npt.NDArray[np.float64]
PairCandidates = tuple[IndexArray, IndexArray, FloatArray, FloatArray]


def _eligible_pair_candidates(
    embedded_bh: Embedded_BH,
    waiting_idx: list[int],
    disk: DiskProfile,
    model_config: ModelConfig,
    current_time_s: float,
    end_time_s: float,
) -> PairCandidates | None:
    """Return candidate indices, cross sections, and formation delays.

    A candidate is retained only when its pair-up delay could still finish
    before the specified end time if its interaction occurred immediately.
    """
    waiting: IndexArray = np.asarray(waiting_idx, dtype=np.intp)
    left, right = np.triu_indices(waiting.size, k=1)
    first_idx = waiting[left]
    second_idx = waiting[right]

    total_mass_msun: FloatArray = np.asarray(
        embedded_bh.m_msun[first_idx] + embedded_bh.m_msun[second_idx],
        dtype=np.float64,
    )
    cross_sections_cm2: FloatArray = np.fromiter(
        (sbh_cross_section(model_config, float(mass)) for mass in total_mass_msun),
        dtype=np.float64,
        count=total_mass_msun.size,
    )
    pair_delays_s: FloatArray = np.fromiter(
        (
            pair_up_timescale(
                disk,
                float(embedded_bh.m_msun[first]),
                float(embedded_bh.m_msun[second]),
            )
            for first, second in zip(first_idx, second_idx)
        ),
        dtype=np.float64,
        count=first_idx.size,
    )

    can_complete = current_time_s + pair_delays_s <= end_time_s
    if not np.any(can_complete):
        return None

    return (
        first_idx[can_complete],
        second_idx[can_complete],
        cross_sections_cm2[can_complete],
        pair_delays_s[can_complete],
    )


def _sample_next_interaction(
    embedded_bh: Embedded_BH,
    waiting_idx: list[int],
    disk: DiskProfile,
    model_config: ModelConfig,
    trap_context: TrapContext,
    current_time_s: float,
    time_limit_s: float,
    end_time_s: float,
    rng_pair_up: np.random.Generator,
) -> tuple[float, int, int, float] | None:
    """Sample one interaction before ``time_limit_s`` or return ``None``.

    For each candidate pair, the interaction rate is ``sigma * v / V``.
    Summing these rates is equivalent to using the instantaneous single-BH
    density together with the mean cross section over the candidate pool.
    """
    candidates = _eligible_pair_candidates(
        embedded_bh=embedded_bh,
        waiting_idx=waiting_idx,
        disk=disk,
        model_config=model_config,
        current_time_s=current_time_s,
        end_time_s=end_time_s,
    )
    if candidates is None:
        return None

    first_idx, second_idx, cross_sections_cm2, pair_delays_s = candidates
    relative_velocity_cms = disk.get_cs(disk.r_trap_cm)
    interaction_rate_s_inv = (
        cross_sections_cm2.sum() * relative_velocity_cms / trap_context.volume_cm3
    )
    interaction_time_s = current_time_s + rng_pair_up.exponential(
        1.0 / interaction_rate_s_inv
    )
    if interaction_time_s >= time_limit_s:
        return None

    pair_position = int(
        rng_pair_up.choice(
            cross_sections_cm2.size,
            p=cross_sections_cm2 / cross_sections_cm2.sum(),
        )
    )
    return (
        float(interaction_time_s),
        int(first_idx[pair_position]),
        int(second_idx[pair_position]),
        float(pair_delays_s[pair_position]),
    )


def construct_pairing_up_timeline(
    embedded_bh: Embedded_BH,
    reached_mask: npt.ArrayLike,
    trap_context: TrapContext,
    model_config: ModelConfig,
    disk: DiskProfile,
    end_time_s: float,
    rng_pair_up: np.random.Generator,
) -> FloatArray:
    """Construct first-generation BBH formation times up to ``end_time_s``.

    The function mutates only initial-pairing state on ``embedded_bh``:

    - ``trap_flag`` is ``-1`` for objects outside the reached population,
      ``-2`` for trap singles, and stores a symmetric partner index for BBHs.
    - ``t_pair_s`` stores the BBH formation time for both members of each pair.

    Returned times have one entry per BBH and are sorted chronologically.
    """
    reached_mask = np.asarray(reached_mask, dtype=bool)
    if reached_mask.shape != (embedded_bh.num,):
        raise ValueError(
            f"reached_mask must have shape ({embedded_bh.num},), "
            f"got {reached_mask.shape}"
        )
    if not np.isfinite(end_time_s) or end_time_s <= 0.0:
        raise ValueError("end_time_s must be finite and positive")

    reached_times_s = embedded_bh.t_reach_s[reached_mask]
    if np.any(~np.isfinite(reached_times_s)) or np.any(reached_times_s < 0.0):
        raise ValueError("Reached BH arrival times must be finite and nonnegative")
    if np.any(embedded_bh.m_msun[reached_mask] <= 0.0):
        raise ValueError("Reached BH masses must be positive")
    if disk.get_cs(disk.r_trap_cm) <= 0.0:
        raise ValueError("The trap sound speed must be positive")

    active_mask = reached_mask & (embedded_bh.t_reach_s < end_time_s)
    embedded_bh.trap_flag.fill(-1)
    embedded_bh.trap_flag[active_mask] = -2
    embedded_bh.t_pair_s.fill(np.nan)

    arrival_idx = np.flatnonzero(active_mask)
    arrival_idx = arrival_idx[
        np.argsort(embedded_bh.t_reach_s[arrival_idx], kind="stable")
    ]
    if arrival_idx.size == 0:
        return np.array([], dtype=float)

    waiting_idx: list[int] = []
    formation_times_s: list[float] = []
    current_time_s = float(embedded_bh.t_reach_s[arrival_idx[0]])

    def process_until(time_limit_s: float) -> None:
        """Consume all completed pair-selection events before one time boundary."""
        nonlocal current_time_s

        while len(waiting_idx) >= 2 and current_time_s < time_limit_s:
            event = _sample_next_interaction(
                embedded_bh=embedded_bh,
                waiting_idx=waiting_idx,
                disk=disk,
                model_config=model_config,
                trap_context=trap_context,
                current_time_s=current_time_s,
                time_limit_s=time_limit_s,
                end_time_s=end_time_s,
                rng_pair_up=rng_pair_up,
            )
            if event is None:
                return

            interaction_time_s, first, second, pair_delay_s = event
            formation_time_s = interaction_time_s + pair_delay_s
            current_time_s = interaction_time_s

            # The delay is checked again at the sampled interaction time. A
            # late interaction leaves both objects single for later events.
            if formation_time_s > end_time_s:
                continue

            embedded_bh.trap_flag[first] = second
            embedded_bh.trap_flag[second] = first
            embedded_bh.t_pair_s[first] = formation_time_s
            embedded_bh.t_pair_s[second] = formation_time_s
            waiting_idx.remove(first)
            waiting_idx.remove(second)
            formation_times_s.append(formation_time_s)

    for index in arrival_idx:
        arrival_time_s = float(embedded_bh.t_reach_s[index])
        process_until(arrival_time_s)
        current_time_s = arrival_time_s
        waiting_idx.append(int(index))

    process_until(end_time_s)
    return np.sort(np.asarray(formation_times_s, dtype=float))
