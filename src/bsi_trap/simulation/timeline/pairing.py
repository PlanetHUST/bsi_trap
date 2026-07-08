"""Event-driven construction of local BBH pairing timelines."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bsi_trap.simulation.state.models import (
    DiskProfile,
    Single_BH_properties,
    TrapContext,
)
from bsi_trap.simulation.config.model_config import ModelConfig
from bsi_trap.simulation.timeline.models import PairingTimeline
from bsi_trap.physics.pairing import (
    conjunction_rate,
    effective_capture_impact_parameter,
    is_strong_gravity_encounter,
    pair_up_timescale,
    shear_relative_velocity,
)

IndexArray = npt.NDArray[np.intp]
FloatArray = npt.NDArray[np.float64]
PairCandidates = tuple[IndexArray, IndexArray, FloatArray, FloatArray]


def _validate_pairing_window(
    *,
    current_time_s: float,
    step_end_s: float,
    smbh_mass_msun: float,
) -> None:
    if not np.isfinite(current_time_s) or current_time_s < 0.0:
        raise ValueError("current_time_s must be finite and nonnegative")
    if not np.isfinite(step_end_s) or step_end_s < current_time_s:
        raise ValueError("step_end_s must be finite and no earlier than current_time_s")
    if not np.isfinite(smbh_mass_msun) or smbh_mass_msun <= 0.0:
        raise ValueError("SMBH mass must be finite and positive")


def _ensure_single_pairing_radii(
    singles: Single_BH_properties,
    active_mask: npt.NDArray[np.bool_],
    disk: DiskProfile,
    model_config: ModelConfig,
    trap_context: TrapContext,
    rng_pair_up: np.random.Generator,
) -> None:
    missing_mask = active_mask & ~np.isfinite(singles.r_pair_cm)
    if np.any(missing_mask):
        radial_sigma_cm = (
            model_config.trap_radial_sigma_in_scale_height
            * disk.get_scale_height_cm(disk.r_trap_cm)
        )
        if radial_sigma_cm <= 0.0:
            raise ValueError("trap_radial_sigma_in_scale_height must be positive")

        offsets_cm = rng_pair_up.normal(0.0, radial_sigma_cm, missing_mask.sum())
        offsets_cm = np.clip(
            offsets_cm,
            -0.5 * trap_context.width_cm,
            0.5 * trap_context.width_cm,
        )
        singles.r_pair_cm[missing_mask] = disk.r_trap_cm + offsets_cm

    active_radii_cm = singles.r_pair_cm[active_mask]
    if np.any(~np.isfinite(active_radii_cm)) or np.any(active_radii_cm <= 0.0):
        raise ValueError("Active single-BH pairing radii must be finite and positive")


def _eligible_single_pair_candidates(
    singles: Single_BH_properties,
    waiting_idx: list[int],
    disk: DiskProfile,
    model_config: ModelConfig,
    smbh_mass_msun: float,
    trap_context: TrapContext,
    current_time_s: float,
    step_end_s: float,
) -> PairCandidates | None:
    waiting: IndexArray = np.asarray(waiting_idx, dtype=np.intp)
    if waiting.size < 2:
        return None

    first_part_index, second_part_index = np.triu_indices(waiting.size, k=1)
    first_idx = waiting[first_part_index]
    second_idx = waiting[second_part_index]

    first_radius_cm: FloatArray = singles.r_pair_cm[first_idx]
    second_radius_cm: FloatArray = singles.r_pair_cm[second_idx]
    mean_radius_cm: FloatArray = 0.5 * (first_radius_cm + second_radius_cm)
    radial_separation_cm: FloatArray = np.abs(first_radius_cm - second_radius_cm)

    impact_parameters_cm: FloatArray = np.fromiter(
        (
            effective_capture_impact_parameter(
                model_config=model_config,
                total_mass_msun=float(
                    singles.m_msun[first] + singles.m_msun[second]
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
                float(singles.m_msun[first]),
                float(singles.m_msun[second]),
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
                float(singles.m_msun[first]),
                float(singles.m_msun[second]),
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
    can_complete_mask = current_time_s + pair_delays_s <= step_end_s
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


def _sample_next_single_encounter(
    singles: Single_BH_properties,
    waiting_idx: list[int],
    disk: DiskProfile,
    model_config: ModelConfig,
    smbh_mass_msun: float,
    trap_context: TrapContext,
    current_time_s: float,
    step_end_s: float,
    rng_pair_up: np.random.Generator,
) -> tuple[float, int, int, float] | None:
    candidates = _eligible_single_pair_candidates(
        singles=singles,
        waiting_idx=waiting_idx,
        disk=disk,
        model_config=model_config,
        smbh_mass_msun=smbh_mass_msun,
        trap_context=trap_context,
        current_time_s=current_time_s,
        step_end_s=step_end_s,
    )
    if candidates is None:
        return None

    first_idx, second_idx, conjunction_rates_s_inv, pair_delays_s = candidates
    encounter_rate_s_inv = conjunction_rates_s_inv.sum()
    if not np.isfinite(encounter_rate_s_inv) or encounter_rate_s_inv <= 0.0:
        raise ValueError("Pair encounter rate must be finite and positive")

    encounter_abs_time_s = current_time_s + rng_pair_up.exponential(
        1.0 / encounter_rate_s_inv
    )
    if encounter_abs_time_s >= step_end_s:
        return None

    pair_position = int(
        rng_pair_up.choice(
            conjunction_rates_s_inv.size,
            p=conjunction_rates_s_inv / conjunction_rates_s_inv.sum(),
        )
    )
    pair_delay_s = float(pair_delays_s[pair_position])
    if encounter_abs_time_s + pair_delay_s > step_end_s:
        return None

    return (
        float(encounter_abs_time_s),
        int(first_idx[pair_position]),
        int(second_idx[pair_position]),
        pair_delay_s,
    )


def construct_pairing_step_timeline(
    *,
    singles: Single_BH_properties,
    trap_context: TrapContext,
    model_config: ModelConfig,
    disk: DiskProfile,
    smbh_mass_msun: float,
    current_time_s: float,
    step_end_s: float,
    rng_pair_up: np.random.Generator,
) -> PairingTimeline:
    """Sample at most one pair formation inside one runtime time step.

    The input indices are relative to ``singles``.  Future singles remain in
    the reservoir, but only objects with ``available_from_s <= current_time_s``
    are eligible during the current step.
    """

    _validate_pairing_window(
        current_time_s=current_time_s,
        step_end_s=step_end_s,
        smbh_mass_msun=smbh_mass_msun,
    )
    if step_end_s == current_time_s or singles.num < 2:
        return PairingTimeline.empty()

    if np.any(~np.isfinite(singles.available_from_s)) or np.any(
        singles.available_from_s < 0.0
    ):
        raise ValueError("Single-BH availability times must be finite and nonnegative")
    if np.any(~np.isfinite(singles.m_msun)) or np.any(singles.m_msun <= 0.0):
        raise ValueError("Single-BH masses must be finite and positive")

    active_mask = singles.available_from_s <= current_time_s
    if np.count_nonzero(active_mask) < 2:
        return PairingTimeline.empty()
    _ensure_single_pairing_radii(
        singles,
        active_mask,
        disk,
        model_config,
        trap_context,
        rng_pair_up,
    )
    waiting_idx = [int(index) for index in np.flatnonzero(active_mask)]
    event_params = _sample_next_single_encounter(
        singles=singles,
        waiting_idx=waiting_idx,
        disk=disk,
        model_config=model_config,
        smbh_mass_msun=smbh_mass_msun,
        trap_context=trap_context,
        current_time_s=current_time_s,
        step_end_s=step_end_s,
        rng_pair_up=rng_pair_up,
    )
    if event_params is None:
        return PairingTimeline.empty()

    encounter_s, first_idx, second_idx, pair_delay_s = event_params
    formation_s = encounter_s + pair_delay_s
    later_arrival_s = max(
        float(singles.available_from_s[first_idx]),
        float(singles.available_from_s[second_idx]),
    )
    return PairingTimeline(
        first_idx=np.asarray([first_idx], dtype=np.intp),
        second_idx=np.asarray([second_idx], dtype=np.intp),
        later_arrival_s=np.asarray([later_arrival_s], dtype=np.float64),
        encounter_s=np.asarray([encounter_s], dtype=np.float64),
        pair_delay_s=np.asarray([pair_delay_s], dtype=np.float64),
        formation_s=np.asarray([formation_s], dtype=np.float64),
    )
