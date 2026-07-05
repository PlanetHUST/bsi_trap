"""Event-driven construction of the initial BBH pairing timeline."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bsi_trap.classes.models import DiskProfile, Embedded_BH, TrapContext
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.evolve.timeline import PairingTimeline
from bsi_trap.functions.pairing_physics import (
    conjunction_rate,
    effective_capture_impact_parameter,
    is_strong_gravity_encounter,
    pair_up_timescale,
    shear_relative_velocity,
)

IndexArray = npt.NDArray[np.intp]
FloatArray = npt.NDArray[np.float64]
PairCandidates = tuple[IndexArray, IndexArray, FloatArray, FloatArray]


def _eligible_pair_candidates(
    embedded_bh: Embedded_BH,
    waiting_idx: list[int],
    disk: DiskProfile,
    model_config: ModelConfig,
    smbh_mass_msun: float,
    trap_context: TrapContext,
    current_time_s: float,
    end_time_s: float,
) -> PairCandidates | None:
    """Return locally eligible pair indices, conjunction rates, and delays.

    A candidate is retained only when its pair-up delay could still finish
    before the specified end time if its interaction occurred immediately.
    """

    waiting: IndexArray = np.asarray(waiting_idx, dtype=np.intp)

    # generate pair index without duplicates
    first_part_index, second_part_index = np.triu_indices(waiting.size, k=1)

    first_idx = waiting[first_part_index]
    second_idx = waiting[second_part_index]

    first_radius_cm: FloatArray = embedded_bh.r_pair_cm[first_idx]
    second_radius_cm: FloatArray = embedded_bh.r_pair_cm[second_idx]
    mean_radius_cm: FloatArray = 0.5 * (first_radius_cm + second_radius_cm)

    radial_separation_cm: FloatArray = np.abs(first_radius_cm - second_radius_cm)
    impact_parameters_cm: FloatArray = np.fromiter(
        (
            effective_capture_impact_parameter(
                model_config=model_config,
                total_mass_msun=float(
                    embedded_bh.m_msun[first] + embedded_bh.m_msun[second]
                ),
                trap_radius_cm=float(radius),
                smbh_mass_msun=smbh_mass_msun,
                trap_width_cm=trap_context.width_cm,
            )
            for first, second, radius in zip(first_idx, second_idx, mean_radius_cm)
        ),
        dtype=np.float64,
        count=first_idx.size,
    )

    relative_velocities_cms: FloatArray = np.fromiter(
        (
            shear_relative_velocity(smbh_mass_msun, float(first), float(second))
            for first, second in zip(first_radius_cm, second_radius_cm)
        ),
        dtype=np.float64,
        count=first_idx.size,
    )

    conjunction_rates_s_inv: FloatArray = np.fromiter(
        (
            conjunction_rate(smbh_mass_msun, float(first), float(second))
            for first, second in zip(first_radius_cm, second_radius_cm)
        ),
        dtype=np.float64,
        count=first_idx.size,
    )

    pair_delays_s: FloatArray = np.fromiter(
        (
            pair_up_timescale(
                disk,
                float(embedded_bh.m_msun[first]),
                float(embedded_bh.m_msun[second]),
                trap_radius_cm=float(radius),
            )
            for first, second, radius in zip(first_idx, second_idx, mean_radius_cm)
        ),
        dtype=np.float64,
        count=first_idx.size,
    )

    local_pairs_mask = radial_separation_cm < impact_parameters_cm

    strong_pairs_mask = np.fromiter(
        (
            is_strong_gravity_encounter(
                float(embedded_bh.m_msun[first]),
                float(embedded_bh.m_msun[second]),
                float(relative_velocity),
                float(impact_parameter),
            )
            for first, second, relative_velocity, impact_parameter in zip(
                first_idx,
                second_idx,
                relative_velocities_cms,
                impact_parameters_cm,
            )
        ),
        dtype=bool,
        count=first_idx.size,
    )
    can_complete_mask = current_time_s + pair_delays_s <= end_time_s

    eligible_mask = (
        local_pairs_mask
        & strong_pairs_mask
        & can_complete_mask
        & (conjunction_rates_s_inv > 0.0)
    )

    if not np.any(eligible_mask):
        return None

    return (
        first_idx[eligible_mask],
        second_idx[eligible_mask],
        conjunction_rates_s_inv[eligible_mask],
        pair_delays_s[eligible_mask],
    )


def _sample_next_encounter(
    embedded_bh: Embedded_BH,
    waiting_idx: list[int],
    disk: DiskProfile,
    model_config: ModelConfig,
    smbh_mass_msun: float,
    trap_context: TrapContext,
    current_time_s: float,
    time_limit_s: float,
    end_time_s: float,
    rng_pair_up: np.random.Generator,
) -> tuple[float, int, int, float] | None:
    """Sample the next phase-averaged local conjunction before ``time_limit_s``.

    The total rate is ``sum(lambda_ij)``, where each candidate's rate is its
    circular-orbit conjunction frequency after local-overlap and
    strong-gravity checks. The sampled interval starts at the current timeline
    boundary, which is never earlier than either selected BH arrival time.
    """
    candidates = _eligible_pair_candidates(
        embedded_bh=embedded_bh,
        waiting_idx=waiting_idx,
        disk=disk,
        model_config=model_config,
        smbh_mass_msun=smbh_mass_msun,
        trap_context=trap_context,
        current_time_s=current_time_s,
        end_time_s=end_time_s,
    )
    if candidates is None:
        return None

    first_idx, second_idx, conjunction_rates_s_inv, pair_delays_s = candidates
    encounter_rate_s_inv = conjunction_rates_s_inv.sum()

    # safe check
    if not np.isfinite(encounter_rate_s_inv) or encounter_rate_s_inv <= 0.0:
        raise ValueError("Pair encounter rate must be finite and positive")

    # Possion distribution encounter assumption
    encounter_abs_time_s = current_time_s + rng_pair_up.exponential(
        1.0 / encounter_rate_s_inv
    )

    if encounter_abs_time_s >= time_limit_s:
        return None

    pair_position = int(
        rng_pair_up.choice(
            conjunction_rates_s_inv.size,
            p=conjunction_rates_s_inv / conjunction_rates_s_inv.sum(),
        )
    )

    return (
        float(encounter_abs_time_s),
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
    smbh_mass_msun: float,
    end_time_s: float,
    rng_pair_up: np.random.Generator,
    max_events: int | None = None,
) -> PairingTimeline:
    """Construct BBH pairing events up to ``end_time_s``.

    ``trap_flag`` is ``-1`` outside the reached population, ``-2`` for trap
    singles, and a non-negative symmetric partner index for paired BBHs.
    ``t_pair_s`` stores BBH formation time. If ``max_events`` is set, timeline
    construction stops after that many committed pairings.
    """

    ####### 1. Prepare for trap radius #######

    # Get single BH reaching mask, and reaching timeline
    reached_mask = np.asarray(reached_mask, dtype=bool)
    reached_times_s = embedded_bh.t_reach_s[reached_mask]

    # Safe check
    if reached_mask.shape != (embedded_bh.num,):
        raise ValueError(
            f"reached_mask must have shape ({embedded_bh.num},), "
            f"got {reached_mask.shape}"
        )
    if not np.isfinite(end_time_s) or end_time_s <= 0.0:
        raise ValueError("end_time_s must be finite and positive")
    if not np.isfinite(smbh_mass_msun) or smbh_mass_msun <= 0.0:
        raise ValueError("SMBH mass must be finite and positive")
    if max_events is not None and max_events < 1:
        raise ValueError("max_events must be at least 1 when specified")

    if np.any(~np.isfinite(reached_times_s)) or np.any(reached_times_s < 0.0):
        raise ValueError("Reached BH arrival times must be finite and nonnegative")
    if np.any(embedded_bh.m_msun[reached_mask] <= 0.0):
        raise ValueError("Reached BH masses must be positive")

    # Select reaching time below dutycycle
    active_mask = reached_mask & (embedded_bh.t_reach_s < end_time_s)

    embedded_bh.r_pair_cm[~active_mask] = np.nan

    actual_task_mask = active_mask & ~np.isfinite(embedded_bh.r_pair_cm)

    # Set random initial radius in the trap
    if np.any(actual_task_mask):
        radial_scale_cm = (
            model_config.trap_radial_dispersion_scale_heights
            * disk.get_scale_height_cm(disk.r_trap_cm)
        )

        # safe check
        if radial_scale_cm <= 0.0:
            raise ValueError("trap_radial_dispersion_scale_heights must be positive")

        centers_cm = np.where(
            np.isfinite(embedded_bh.r_trap_target_cm[actual_task_mask]),
            embedded_bh.r_trap_target_cm[actual_task_mask],
            disk.r_trap_cm,
        )

        # Normal distribution center offset
        offsets_cm = rng_pair_up.normal(0.0, radial_scale_cm, actual_task_mask.sum())
        offsets_cm = np.clip(
            offsets_cm,
            -0.5 * trap_context.width_cm,
            0.5 * trap_context.width_cm,
        )
        embedded_bh.r_pair_cm[actual_task_mask] = centers_cm + offsets_cm

    active_radii_cm = embedded_bh.r_pair_cm[active_mask]

    if np.any(~np.isfinite(active_radii_cm)) or np.any(active_radii_cm <= 0.0):
        raise ValueError("Reached BH pairing radii must be finite and positive")

    embedded_bh.trap_flag.fill(-1)
    embedded_bh.trap_flag[active_mask] = -2
    embedded_bh.t_pair_s.fill(np.nan)

    arrival_idx = np.flatnonzero(active_mask)
    arrival_idx = arrival_idx[
        np.argsort(embedded_bh.t_reach_s[arrival_idx], kind="stable")
    ]

    if arrival_idx.size == 0:
        return PairingTimeline.empty()

    waiting_idx: list[int] = []
    first_indices: list[int] = []
    second_indices: list[int] = []
    later_arrivals_s: list[float] = []
    encounter_times_s: list[float] = []
    pair_delays_s: list[float] = []
    formation_times_s: list[float] = []

    # Step current time for constructing timeline
    current_time_s = float(embedded_bh.t_reach_s[arrival_idx[0]])

    def process_until(time_limit_s: float) -> bool:

        nonlocal current_time_s

        while len(waiting_idx) >= 2 and current_time_s < time_limit_s:

            event_parms = _sample_next_encounter(
                embedded_bh=embedded_bh,
                waiting_idx=waiting_idx,
                disk=disk,
                model_config=model_config,
                smbh_mass_msun=smbh_mass_msun,
                trap_context=trap_context,
                current_time_s=current_time_s,
                time_limit_s=time_limit_s,
                end_time_s=end_time_s,
                rng_pair_up=rng_pair_up,
            )

            if event_parms is None:
                return False

            encounter_abs_time_s, first_ind, second_ind, pair_delay_s = event_parms
            bbh_formation_time_s = encounter_abs_time_s + pair_delay_s

            current_time_s = encounter_abs_time_s

            # check for dutycycle
            if bbh_formation_time_s > end_time_s:
                continue

            embedded_bh.trap_flag[first_ind] = second_ind
            embedded_bh.trap_flag[second_ind] = first_ind
            embedded_bh.t_pair_s[first_ind] = bbh_formation_time_s
            embedded_bh.t_pair_s[second_ind] = bbh_formation_time_s

            waiting_idx.remove(first_ind)
            waiting_idx.remove(second_ind)

            first_indices.append(first_ind)
            second_indices.append(second_ind)
            later_arrivals_s.append(
                max(
                    float(embedded_bh.t_reach_s[first_ind]),
                    float(embedded_bh.t_reach_s[second_ind]),
                )
            )
            encounter_times_s.append(encounter_abs_time_s)
            pair_delays_s.append(pair_delay_s)
            formation_times_s.append(bbh_formation_time_s)
            if max_events is not None and len(formation_times_s) >= max_events:
                return True
        return False

    for index in arrival_idx:
        arrival_time_s = float(embedded_bh.t_reach_s[index])
        if process_until(arrival_time_s):
            break
        current_time_s = arrival_time_s
        waiting_idx.append(int(index))

    if max_events is None or len(formation_times_s) < max_events:
        process_until(end_time_s)

    if not formation_times_s:
        return PairingTimeline.empty()

    formation_array = np.asarray(formation_times_s, dtype=float)
    order = np.argsort(formation_array, kind="stable")
    return PairingTimeline(
        first_idx=np.asarray(first_indices, dtype=np.intp)[order],
        second_idx=np.asarray(second_indices, dtype=np.intp)[order],
        later_arrival_s=np.asarray(later_arrivals_s, dtype=float)[order],
        encounter_s=np.asarray(encounter_times_s, dtype=float)[order],
        pair_delay_s=np.asarray(pair_delays_s, dtype=float)[order],
        formation_s=formation_array[order],
    )
