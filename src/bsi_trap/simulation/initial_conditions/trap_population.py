from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bsi_trap.simulation.state.models import (
    AGN_properties,
    BBH_properties,
    DiskProfile,
    Embedded_BH,
    Single_BH_properties,
)
from bsi_trap.simulation.config.model_config import ModelConfig
from bsi_trap.constant.constant import Myr
from bsi_trap.physics.pairing import (
    conjunction_rate,
    effective_capture_impact_parameter,
    is_strong_gravity_encounter,
    pair_up_timescale,
    shear_relative_velocity,
)
from bsi_trap.simulation.initial_conditions.embedded_bh import embedded_BH_initialize
from bsi_trap.physics.trap.migration import (
    resolve_migration,
)
from bsi_trap.physics.trap.context import build_trap_context
from bsi_trap.simulation.timeline.models import PairingTimeline

IndexArray = npt.NDArray[np.intp]
FloatArray = npt.NDArray[np.float64]
PairCandidates = tuple[IndexArray, IndexArray, FloatArray, FloatArray]


@dataclass(frozen=True)
class TrapPopulation:
    """Trap-arrival state plus optional committed BBHs for diagnostics."""

    embedded_bh: Embedded_BH
    binaries: BBH_properties
    singles: Single_BH_properties
    pairing_timeline: PairingTimeline


def make_trap_population(
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    rng_nsc: np.random.Generator,
) -> TrapPopulation:
    """Build the embedded-BH trap-arrival reservoir.

    Reaching the trap only makes an object available as a single BH. Pairing is
    committed later by the runtime event loop on the shared event clock.
    """

    embedded_bh = embedded_BH_initialize(agn, disk, rng_nsc)

    prepare_trap_arrival_state(
        model_config=model_config,
        agn=agn,
        disk=disk,
        ebh=embedded_bh,
    )
    return TrapPopulation(
        embedded_bh=embedded_bh,
        binaries=BBH_properties(0),
        singles=init_sbh(embedded_bh),
        pairing_timeline=PairingTimeline.empty(),
    )


def _validate_embedded(ebh: Embedded_BH) -> None:
    """Check the array contract required before migration and pairing."""
    expected_shape = (ebh.num,)
    for name in ("m_msun", "t_form_s", "R_cm"):
        values = getattr(ebh, name)
        if values.shape != expected_shape:
            raise ValueError(f"Embedded-BH {name} must have shape {expected_shape}")

    if np.any(~np.isfinite(ebh.m_msun)) or np.any(ebh.m_msun <= 0.0):
        raise ValueError("Embedded-BH masses must be finite and positive")
    if np.any(~np.isfinite(ebh.t_form_s)) or np.any(ebh.t_form_s < 0.0):
        raise ValueError("Embedded-BH formation times must be finite and nonnegative")
    if np.any(~np.isfinite(ebh.R_cm)) or np.any(ebh.R_cm <= 0.0):
        raise ValueError("Embedded-BH initial radii must be finite and positive")


def _validate_trap_state(ebh: Embedded_BH, end_time_s: float) -> None:
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


def prepare_trap_arrival_state(
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    ebh: Embedded_BH,
) -> np.ndarray:
    """Resolve migration and mark reached trap objects as live singles."""

    _validate_embedded(ebh)

    end_time_s = agn.lifetime_myr * Myr

    if not np.isfinite(end_time_s) or end_time_s <= 0.0:
        raise ValueError("AGN lifetime must define a finite, positive end time")

    migration_results = resolve_migration(
        agn=agn,
        disk=disk,
        ebh=ebh,
    )

    migration_times_s = migration_results.migration_times_s

    # check
    if migration_times_s.shape != (ebh.num,):
        raise ValueError("Migration solver returned an invalid population shape")
    if np.any(np.isnan(migration_times_s)) or np.any(migration_times_s < 0.0):
        raise ValueError(
            "Migration times must be nonnegative finite values or infinity"
        )

    ebh.t_mig_s = migration_times_s
    ebh.t_reach_s = ebh.t_form_s + migration_times_s
    ebh.r_trap_target_cm = migration_results.target_radii_cm

    reached_mask = migration_results.stable_mask & (ebh.t_reach_s < end_time_s)

    trap_context = build_trap_context(disk, model_config)

    ebh.trap_width_cm = trap_context.width_cm

    ebh.trap_flag.fill(-1)
    ebh.trap_flag[reached_mask] = -2

    ebh.t_pair_s.fill(np.nan)  # now set to be nan

    return reached_mask


def _eligible_embedded_pair_candidates(
    ebh: Embedded_BH,
    waiting_idx: list[int],
    disk: DiskProfile,
    model_config: ModelConfig,
    smbh_mass_msun: float,
    trap_width_cm: float,
    current_time_s: float,
    end_time_s: float,
) -> PairCandidates | None:
    waiting = np.asarray(waiting_idx, dtype=np.intp)
    if waiting.size < 2:
        return None

    first_part_index, second_part_index = np.triu_indices(waiting.size, k=1)
    first_idx = waiting[first_part_index]
    second_idx = waiting[second_part_index]

    first_radius_cm = ebh.r_pair_cm[first_idx]
    second_radius_cm = ebh.r_pair_cm[second_idx]
    mean_radius_cm = 0.5 * (first_radius_cm + second_radius_cm)
    radial_separation_cm = np.abs(first_radius_cm - second_radius_cm)

    impact_parameters_cm = np.fromiter(
        (
            effective_capture_impact_parameter(
                model_config=model_config,
                total_mass_msun=float(ebh.m_msun[first] + ebh.m_msun[second]),
                trap_radius_cm=float(radius),
                smbh_mass_msun=smbh_mass_msun,
                trap_width_cm=trap_width_cm,
            )
            for first, second, radius in zip(first_idx, second_idx, mean_radius_cm)
        ),
        dtype=np.float64,
        count=first_idx.size,
    )
    relative_velocities_cms = np.fromiter(
        (
            shear_relative_velocity(smbh_mass_msun, float(first), float(second))
            for first, second in zip(first_radius_cm, second_radius_cm)
        ),
        dtype=np.float64,
        count=first_idx.size,
    )
    conjunction_rates_s_inv = np.fromiter(
        (
            conjunction_rate(smbh_mass_msun, float(first), float(second))
            for first, second in zip(first_radius_cm, second_radius_cm)
        ),
        dtype=np.float64,
        count=first_idx.size,
    )
    pair_delays_s = np.fromiter(
        (
            pair_up_timescale(
                disk,
                float(ebh.m_msun[first]),
                float(ebh.m_msun[second]),
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
                float(ebh.m_msun[first]),
                float(ebh.m_msun[second]),
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
    eligible_mask = (
        local_pairs_mask
        & strong_pairs_mask
        & (current_time_s + pair_delays_s <= end_time_s)
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


def _sample_next_embedded_encounter(
    ebh: Embedded_BH,
    waiting_idx: list[int],
    disk: DiskProfile,
    model_config: ModelConfig,
    smbh_mass_msun: float,
    trap_width_cm: float,
    current_time_s: float,
    time_limit_s: float,
    end_time_s: float,
    rng_pair_up: np.random.Generator,
) -> tuple[float, int, int, float] | None:
    candidates = _eligible_embedded_pair_candidates(
        ebh=ebh,
        waiting_idx=waiting_idx,
        disk=disk,
        model_config=model_config,
        smbh_mass_msun=smbh_mass_msun,
        trap_width_cm=trap_width_cm,
        current_time_s=current_time_s,
        end_time_s=end_time_s,
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


def construct_initial_pairing_timeline(
    *,
    ebh: Embedded_BH,
    reached_mask: npt.ArrayLike,
    model_config: ModelConfig,
    disk: DiskProfile,
    smbh_mass_msun: float,
    end_time_s: float,
    rng_pair_up: np.random.Generator,
) -> PairingTimeline:
    """Build the standalone embedded-BH pairing diagnostic timeline."""
    reached_mask = np.asarray(reached_mask, dtype=bool)
    if reached_mask.shape != (ebh.num,):
        raise ValueError(
            f"reached_mask must have shape ({ebh.num},), got {reached_mask.shape}"
        )
    if not np.isfinite(end_time_s) or end_time_s <= 0.0:
        raise ValueError("end_time_s must be finite and positive")
    if not np.isfinite(smbh_mass_msun) or smbh_mass_msun <= 0.0:
        raise ValueError("SMBH mass must be finite and positive")

    reached_times_s = ebh.t_reach_s[reached_mask]
    if np.any(~np.isfinite(reached_times_s)) or np.any(reached_times_s < 0.0):
        raise ValueError("Reached BH arrival times must be finite and nonnegative")
    reached_masses_msun = ebh.m_msun[reached_mask]
    if np.any(~np.isfinite(reached_masses_msun)) or np.any(
        reached_masses_msun <= 0.0
    ):
        raise ValueError("Reached BH masses must be finite and positive")

    active_mask = reached_mask & (ebh.t_reach_s < end_time_s)
    ebh.r_pair_cm[~active_mask] = np.nan
    trap_width_cm = float(ebh.trap_width_cm)
    if not np.isfinite(trap_width_cm) or trap_width_cm <= 0.0:
        trap_width_cm = build_trap_context(disk, model_config).width_cm
        ebh.trap_width_cm = trap_width_cm

    missing_radii_mask = active_mask & ~np.isfinite(ebh.r_pair_cm)
    if np.any(missing_radii_mask):
        radial_sigma_cm = (
            model_config.trap_radial_sigma_in_scale_height
            * disk.get_scale_height_cm(disk.r_trap_cm)
        )
        if radial_sigma_cm <= 0.0:
            raise ValueError("trap_radial_sigma_in_scale_height must be positive")

        centers_cm = np.where(
            np.isfinite(ebh.r_trap_target_cm[missing_radii_mask]),
            ebh.r_trap_target_cm[missing_radii_mask],
            disk.r_trap_cm,
        )
        offsets_cm = rng_pair_up.normal(0.0, radial_sigma_cm, missing_radii_mask.sum())
        offsets_cm = np.clip(
            offsets_cm,
            -0.5 * trap_width_cm,
            0.5 * trap_width_cm,
        )
        ebh.r_pair_cm[missing_radii_mask] = centers_cm + offsets_cm

    active_radii_cm = ebh.r_pair_cm[active_mask]
    if np.any(~np.isfinite(active_radii_cm)) or np.any(active_radii_cm <= 0.0):
        raise ValueError("Reached BH pairing radii must be finite and positive")

    ebh.trap_flag.fill(-1)
    ebh.trap_flag[active_mask] = -2
    ebh.t_pair_s.fill(np.nan)

    arrival_idx = np.flatnonzero(active_mask)
    arrival_idx = arrival_idx[np.argsort(ebh.t_reach_s[arrival_idx], kind="stable")]
    if arrival_idx.size == 0:
        return PairingTimeline.empty()

    waiting_idx: list[int] = []
    first_indices: list[int] = []
    second_indices: list[int] = []
    later_arrivals_s: list[float] = []
    encounter_times_s: list[float] = []
    pair_delays_s: list[float] = []
    formation_times_s: list[float] = []
    current_time_s = float(ebh.t_reach_s[arrival_idx[0]])

    def process_until(time_limit_s: float) -> None:
        nonlocal current_time_s
        while len(waiting_idx) >= 2 and current_time_s < time_limit_s:
            event_params = _sample_next_embedded_encounter(
                ebh=ebh,
                waiting_idx=waiting_idx,
                disk=disk,
                model_config=model_config,
                smbh_mass_msun=smbh_mass_msun,
                trap_width_cm=trap_width_cm,
                current_time_s=current_time_s,
                time_limit_s=time_limit_s,
                end_time_s=end_time_s,
                rng_pair_up=rng_pair_up,
            )
            if event_params is None:
                return

            encounter_abs_time_s, first_ind, second_ind, pair_delay_s = event_params
            formation_time_s = encounter_abs_time_s + pair_delay_s
            current_time_s = encounter_abs_time_s
            if formation_time_s > end_time_s:
                continue

            ebh.trap_flag[first_ind] = second_ind
            ebh.trap_flag[second_ind] = first_ind
            ebh.t_pair_s[first_ind] = formation_time_s
            ebh.t_pair_s[second_ind] = formation_time_s
            waiting_idx.remove(first_ind)
            waiting_idx.remove(second_ind)

            first_indices.append(first_ind)
            second_indices.append(second_ind)
            later_arrivals_s.append(
                max(float(ebh.t_reach_s[first_ind]), float(ebh.t_reach_s[second_ind]))
            )
            encounter_times_s.append(encounter_abs_time_s)
            pair_delays_s.append(pair_delay_s)
            formation_times_s.append(formation_time_s)

    waiting_idx.append(int(arrival_idx[0]))
    for index in arrival_idx[1:]:
        arrival_time_s = float(ebh.t_reach_s[index])
        process_until(arrival_time_s)
        current_time_s = arrival_time_s
        waiting_idx.append(int(index))
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


def construct_initial_trap_population(
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    ebh: Embedded_BH,
    rng_pair_up: np.random.Generator,
) -> tuple[BBH_properties, Single_BH_properties, PairingTimeline]:
    """Build first-generation BBH/single populations and their pairing timeline."""

    end_time_s = agn.lifetime_myr * Myr

    reached_mask = prepare_trap_arrival_state(
        model_config=model_config,
        agn=agn,
        disk=disk,
        ebh=ebh,
    )
    pairing_timeline = construct_initial_pairing_timeline(
        ebh=ebh,
        reached_mask=reached_mask,
        model_config=model_config,
        disk=disk,
        smbh_mass_msun=agn.M_SMBH_msun,
        end_time_s=end_time_s,
        rng_pair_up=rng_pair_up,
    )

    _validate_trap_state(ebh, end_time_s)

    # 3. Initialization population
    bbh = init_bbh(ebh)
    sbh = init_sbh(ebh)

    reached_count = int(np.count_nonzero(reached_mask))

    # number conservation check
    if 2 * bbh.num + sbh.num != reached_count:
        raise ValueError("Initial trap population does not conserve reached BHs")

    return bbh, sbh, pairing_timeline


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
