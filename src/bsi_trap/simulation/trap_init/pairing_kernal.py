from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bsi_trap.constant.constant import Myr
from bsi_trap.physics.binary.pairing import (
    conjunction_rate,
    effective_capture_impact_parameter,
    is_strong_gravity_encounter,
    pair_up_timescale,
    shear_relative_velocity,
)
from bsi_trap.physics.trap.context import build_trap_context
from bsi_trap.simulation.config.model_config import ModelConfig
from bsi_trap.simulation.trap_init.conversion import (
    init_bbh,
    init_sbh,
    validate_trap_state,
)
from bsi_trap.simulation.state.models import (
    Arrive_timeline,
    AGN_properties,
    BBH_properties,
    DiskProfile,
    Embedded_BH,
    Single_BH_properties,
)
from bsi_trap.simulation.timeline.models import PairingTimeline

IndexArray = npt.NDArray[np.intp]
FloatArray = npt.NDArray[np.float64]
PairCandidates = tuple[IndexArray, IndexArray, FloatArray, FloatArray]


def construct_next_pair_event(
    current_time_s: float,
    arrive_timeline: Arrive_timeline,
    agn: AGN_properties,
    disk: DiskProfile,
    *,
    sbh: Single_BH_properties | None = None,
    bbh: BBH_properties | None = None,
):
    """two case pair-up event method
    1: construct initial when sbh, bbh is None
    2: construct subsequent event when given sbh, bbh

    arrive_timeline contains initial embedded reaching timeline and future remnans
    """

    # check
    if current_time_s == 0 and sbh == None and bbh == None:
        print("initialize state")

    end_time_s = agn.lifetime_myr * Myr


# def construct_initial_pairing_timeline(
#     *,
#     ebh: Embedded_BH,
#     reached_mask: npt.ArrayLike,
#     model_config: ModelConfig,
#     disk: DiskProfile,
#     smbh_mass_msun: float,
#     end_time_s: float,
#     rng_pair_up: np.random.Generator,
# ) -> PairingTimeline:
#     """Build the standalone embedded-BH pairing diagnostic timeline."""

#     reached_mask = np.asarray(reached_mask, dtype=bool)

#     if reached_mask.shape != (ebh.num,):
#         raise ValueError(
#             f"reached_mask must have shape ({ebh.num},), got {reached_mask.shape}"
#         )
#     if not np.isfinite(end_time_s) or end_time_s <= 0.0:
#         raise ValueError("end_time_s must be finite and positive")
#     if not np.isfinite(smbh_mass_msun) or smbh_mass_msun <= 0.0:
#         raise ValueError("SMBH mass must be finite and positive")

#     reached_times_s = ebh.t_reach_s[reached_mask]

#     if np.any(~np.isfinite(reached_times_s)) or np.any(reached_times_s < 0.0):
#         raise ValueError("Reached BH arrival times must be finite and nonnegative")
#     reached_masses_msun = ebh.m_msun[reached_mask]
#     if np.any(~np.isfinite(reached_masses_msun)) or np.any(reached_masses_msun <= 0.0):
#         raise ValueError("Reached BH masses must be finite and positive")

#     active_mask = reached_mask & (ebh.t_reach_s < end_time_s)
#     ebh.r_pair_cm[~active_mask] = np.nan
#     trap_width_cm = float(ebh.trap_width_cm)

#     if not np.isfinite(trap_width_cm) or trap_width_cm <= 0.0:
#         trap_width_cm = build_trap_context(disk, model_config).width_cm
#         ebh.trap_width_cm = trap_width_cm

#     missing_radii_mask = active_mask & ~np.isfinite(ebh.r_pair_cm)

#     if np.any(missing_radii_mask):
#         radial_sigma_cm = (
#             model_config.trap_radial_sigma_in_scale_height
#             * disk.get_scale_height_cm(disk.r_trap_cm)
#         )
#         if radial_sigma_cm <= 0.0:
#             raise ValueError("trap_radial_sigma_in_scale_height must be positive")

#         centers_cm = np.where(
#             np.isfinite(ebh.r_trap_target_cm[missing_radii_mask]),
#             ebh.r_trap_target_cm[missing_radii_mask],
#             disk.r_trap_cm,
#         )
#         offsets_cm = rng_pair_up.normal(0.0, radial_sigma_cm, missing_radii_mask.sum())
#         offsets_cm = np.clip(
#             offsets_cm,
#             -0.5 * trap_width_cm,
#             0.5 * trap_width_cm,
#         )
#         ebh.r_pair_cm[missing_radii_mask] = centers_cm + offsets_cm

#     active_radii_cm = ebh.r_pair_cm[active_mask]
#     if np.any(~np.isfinite(active_radii_cm)) or np.any(active_radii_cm <= 0.0):
#         raise ValueError("Reached BH pairing radii must be finite and positive")

#     ebh.trap_flag.fill(-1)
#     ebh.trap_flag[active_mask] = -2
#     ebh.t_pair_s.fill(np.nan)

#     arrival_idx = np.flatnonzero(active_mask)
#     arrival_idx = arrival_idx[np.argsort(ebh.t_reach_s[arrival_idx], kind="stable")]
#     if arrival_idx.size == 0:
#         return PairingTimeline.empty()

#     waiting_idx: list[int] = []
#     first_indices: list[int] = []
#     second_indices: list[int] = []
#     later_arrivals_s: list[float] = []
#     encounter_times_s: list[float] = []
#     pair_delays_s: list[float] = []
#     formation_times_s: list[float] = []
#     current_time_s = float(ebh.t_reach_s[arrival_idx[0]])

#     def process_until(time_limit_s: float) -> None:
#         nonlocal current_time_s
#         while len(waiting_idx) >= 2 and current_time_s < time_limit_s:
#             event_params = _sample_next_embedded_encounter(
#                 ebh=ebh,
#                 waiting_idx=waiting_idx,
#                 disk=disk,
#                 model_config=model_config,
#                 smbh_mass_msun=smbh_mass_msun,
#                 trap_width_cm=trap_width_cm,
#                 current_time_s=current_time_s,
#                 time_limit_s=time_limit_s,
#                 end_time_s=end_time_s,
#                 rng_pair_up=rng_pair_up,
#             )
#             if event_params is None:
#                 return

#             encounter_abs_time_s, first_ind, second_ind, pair_delay_s = event_params
#             formation_time_s = encounter_abs_time_s + pair_delay_s
#             current_time_s = encounter_abs_time_s
#             if formation_time_s > end_time_s:
#                 continue

#             ebh.trap_flag[first_ind] = second_ind
#             ebh.trap_flag[second_ind] = first_ind
#             ebh.t_pair_s[first_ind] = formation_time_s
#             ebh.t_pair_s[second_ind] = formation_time_s
#             waiting_idx.remove(first_ind)
#             waiting_idx.remove(second_ind)

#             first_indices.append(first_ind)
#             second_indices.append(second_ind)
#             later_arrivals_s.append(
#                 max(float(ebh.t_reach_s[first_ind]), float(ebh.t_reach_s[second_ind]))
#             )
#             encounter_times_s.append(encounter_abs_time_s)
#             pair_delays_s.append(pair_delay_s)
#             formation_times_s.append(formation_time_s)

#     waiting_idx.append(int(arrival_idx[0]))
#     for index in arrival_idx[1:]:
#         arrival_time_s = float(ebh.t_reach_s[index])
#         process_until(arrival_time_s)
#         current_time_s = arrival_time_s
#         waiting_idx.append(int(index))
#     process_until(end_time_s)

#     if not formation_times_s:
#         return PairingTimeline.empty()

#     formation_array = np.asarray(formation_times_s, dtype=float)
#     order = np.argsort(formation_array, kind="stable")

#     return PairingTimeline(
#         first_idx=np.asarray(first_indices, dtype=np.intp)[order],
#         second_idx=np.asarray(second_indices, dtype=np.intp)[order],
#         later_arrival_s=np.asarray(later_arrivals_s, dtype=float)[order],
#         encounter_s=np.asarray(encounter_times_s, dtype=float)[order],
#         pair_delay_s=np.asarray(pair_delays_s, dtype=float)[order],
#         formation_s=formation_array[order],
#     )


# # def construct_initial_trap_population(
# #     model_config: ModelConfig,
# #     agn: AGN_properties,
# #     disk: DiskProfile,
# #     ebh: Embedded_BH,
# #     rng_pair_up: np.random.Generator,
# # ) -> tuple[BBH_properties, Single_BH_properties, PairingTimeline]:
# #     """Build first-generation BBH/single populations for diagnostics only."""

# #     end_time_s = agn.lifetime_myr * Myr
# #     reached_mask = embedded_trap_reach_cal(
# #         model_config=model_config,
# #         agn=agn,
# #         disk=disk,
# #         ebh=ebh,
# #     )
# #     pairing_timeline = construct_initial_pairing_timeline(
# #         ebh=ebh,
# #         reached_mask=reached_mask,
# #         model_config=model_config,
# #         disk=disk,
# #         smbh_mass_msun=agn.M_SMBH_msun,
# #         end_time_s=end_time_s,
# #         rng_pair_up=rng_pair_up,
# #     )

# #     validate_trap_state(ebh, end_time_s)
# #     bbh = init_bbh(ebh)
# #     sbh = init_sbh(ebh)

# #     reached_count = int(np.count_nonzero(reached_mask))
# #     if 2 * bbh.num + sbh.num != reached_count:
# #         raise ValueError("Initial trap population does not conserve reached BHs")

# #     return bbh, sbh, pairing_timeline


# def _eligible_embedded_pair_candidates(
#     ebh: Embedded_BH,
#     waiting_idx: list[int],
#     disk: DiskProfile,
#     model_config: ModelConfig,
#     smbh_mass_msun: float,
#     trap_width_cm: float,
#     current_time_s: float,
#     end_time_s: float,
# ) -> PairCandidates | None:
#     waiting = np.asarray(waiting_idx, dtype=np.intp)
#     if waiting.size < 2:
#         return None

#     first_part_index, second_part_index = np.triu_indices(waiting.size, k=1)
#     first_idx = waiting[first_part_index]
#     second_idx = waiting[second_part_index]

#     first_radius_cm = ebh.r_pair_cm[first_idx]
#     second_radius_cm = ebh.r_pair_cm[second_idx]
#     mean_radius_cm = 0.5 * (first_radius_cm + second_radius_cm)
#     radial_separation_cm = np.abs(first_radius_cm - second_radius_cm)

#     impact_parameters_cm = np.fromiter(
#         (
#             effective_capture_impact_parameter(
#                 model_config=model_config,
#                 total_mass_msun=float(ebh.m_msun[first] + ebh.m_msun[second]),
#                 trap_radius_cm=float(radius),
#                 smbh_mass_msun=smbh_mass_msun,
#                 trap_width_cm=trap_width_cm,
#             )
#             for first, second, radius in zip(first_idx, second_idx, mean_radius_cm)
#         ),
#         dtype=np.float64,
#         count=first_idx.size,
#     )
#     relative_velocities_cms = np.fromiter(
#         (
#             shear_relative_velocity(smbh_mass_msun, float(first), float(second))
#             for first, second in zip(first_radius_cm, second_radius_cm)
#         ),
#         dtype=np.float64,
#         count=first_idx.size,
#     )
#     conjunction_rates_s_inv = np.fromiter(
#         (
#             conjunction_rate(smbh_mass_msun, float(first), float(second))
#             for first, second in zip(first_radius_cm, second_radius_cm)
#         ),
#         dtype=np.float64,
#         count=first_idx.size,
#     )
#     pair_delays_s = np.fromiter(
#         (
#             pair_up_timescale(
#                 disk,
#                 float(ebh.m_msun[first]),
#                 float(ebh.m_msun[second]),
#                 trap_radius_cm=float(radius),
#             )
#             for first, second, radius in zip(first_idx, second_idx, mean_radius_cm)
#         ),
#         dtype=np.float64,
#         count=first_idx.size,
#     )

#     local_pairs_mask = radial_separation_cm < impact_parameters_cm
#     strong_pairs_mask = np.fromiter(
#         (
#             is_strong_gravity_encounter(
#                 float(ebh.m_msun[first]),
#                 float(ebh.m_msun[second]),
#                 float(relative_velocity),
#                 float(impact_parameter),
#             )
#             for first, second, relative_velocity, impact_parameter in zip(
#                 first_idx,
#                 second_idx,
#                 relative_velocities_cms,
#                 impact_parameters_cm,
#             )
#         ),
#         dtype=bool,
#         count=first_idx.size,
#     )
#     eligible_mask = (
#         local_pairs_mask
#         & strong_pairs_mask
#         & (current_time_s + pair_delays_s <= end_time_s)
#         & (conjunction_rates_s_inv > 0.0)
#     )
#     if not np.any(eligible_mask):
#         return None
#     return (
#         first_idx[eligible_mask],
#         second_idx[eligible_mask],
#         conjunction_rates_s_inv[eligible_mask],
#         pair_delays_s[eligible_mask],
#     )


# def _sample_next_embedded_encounter(
#     ebh: Embedded_BH,
#     waiting_idx: list[int],
#     disk: DiskProfile,
#     model_config: ModelConfig,
#     smbh_mass_msun: float,
#     trap_width_cm: float,
#     current_time_s: float,
#     time_limit_s: float,
#     end_time_s: float,
#     rng_pair_up: np.random.Generator,
# ) -> tuple[float, int, int, float] | None:
#     candidates = _eligible_embedded_pair_candidates(
#         ebh=ebh,
#         waiting_idx=waiting_idx,
#         disk=disk,
#         model_config=model_config,
#         smbh_mass_msun=smbh_mass_msun,
#         trap_width_cm=trap_width_cm,
#         current_time_s=current_time_s,
#         end_time_s=end_time_s,
#     )
#     if candidates is None:
#         return None

#     first_idx, second_idx, conjunction_rates_s_inv, pair_delays_s = candidates
#     encounter_rate_s_inv = conjunction_rates_s_inv.sum()
#     if not np.isfinite(encounter_rate_s_inv) or encounter_rate_s_inv <= 0.0:
#         raise ValueError("Pair encounter rate must be finite and positive")

#     encounter_abs_time_s = current_time_s + rng_pair_up.exponential(
#         1.0 / encounter_rate_s_inv
#     )
#     if encounter_abs_time_s >= time_limit_s:
#         return None

#     pair_position = int(
#         rng_pair_up.choice(
#             conjunction_rates_s_inv.size,
#             p=conjunction_rates_s_inv / conjunction_rates_s_inv.sum(),
#         )
#     )
#     return (
#         float(encounter_abs_time_s),
#         int(first_idx[pair_position]),
#         int(second_idx[pair_position]),
#         float(pair_delays_s[pair_position]),
#     )
