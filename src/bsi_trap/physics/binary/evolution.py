"""Shared gas/GW BBH orbit integration bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from bsi_trap.simulation.state.models import AGN_properties, BBHEvolverParams, BBH_properties
from bsi_trap.constant import constant as const
from bsi_trap.constant.stop_reason import StopReasons
from bsi_trap.physics.binary import bsi
from bsi_trap.physics.binary import numeric
from bsi_trap.physics.binary import quantities as pq
from bsi_trap.physics.binary import stop_conditions as stop_cond
from bsi_trap.physics.binary.tracks import DecimatedTrackRecorder, append_bbh_track

OdeFunction = Callable[..., tuple[float, float]]
MergerReachabilityGuard = Callable[["BinaryEvolutionState", float, float], bool]


@dataclass
class BinaryEvolutionState:
    m1_g: float
    m2_g: float
    sma_cm: float
    ecc: float
    delay_s: float = 0.0
    ecc10: float = 0.0
    ecc0p1: float = 0.0
    reached_10hz: bool = False
    reached_0p1hz: bool = False


@dataclass(frozen=True)
class EvolutionTimeLimit:
    hubble_s: float
    duty_cycle_s: float
    stop_s: float
    stop_reason: int


def build_evolution_time_limit(agn: AGN_properties) -> EvolutionTimeLimit:
    hubble_s = float(const.tHubble * const.Myr)
    duty_cycle_s = float(agn.lifetime_myr * const.Myr)
    if hubble_s <= duty_cycle_s:
        return EvolutionTimeLimit(
            hubble_s=hubble_s,
            duty_cycle_s=duty_cycle_s,
            stop_s=hubble_s,
            stop_reason=StopReasons.STOP_HUBBLE_LIMIT,
        )
    return EvolutionTimeLimit(
        hubble_s=hubble_s,
        duty_cycle_s=duty_cycle_s,
        stop_s=duty_cycle_s,
        stop_reason=StopReasons.STOP_AGN_LIFETIME,
    )


def _binary_state_from_bbh(bbh: BBH_properties, i: int) -> BinaryEvolutionState:
    return BinaryEvolutionState(
        m1_g=float(bbh.m1_msun[i] * const.msun),
        m2_g=float(bbh.m2_msun[i] * const.msun),
        sma_cm=float(bbh.sma_cm[i]),
        ecc=float(bbh.ecc[i]),
        delay_s=float(bbh.t_del_s[i]),
        ecc10=float(bbh.ecc10[i]),
        ecc0p1=float(bbh.ecc0p1[i]),
        reached_10hz=bool(bbh.ecc10[i] != 0.0),
        reached_0p1hz=bool(bbh.ecc0p1[i] != 0.0),
    )


def _mark_non_merger_stop(bbh: BBH_properties, i: int, stop_reason: int) -> None:
    flags = bbh.flags
    flags.merger_flag[i] = 0
    flags.continue_ng_flag[i] = 0
    flags.stop_reason_flag[i] = stop_reason


def _outside_merger_boundary(
    params: BBHEvolverParams,
    sma_cm: float,
    threshold_radius_cm: float,
) -> bool:
    return (
        (sma_cm - threshold_radius_cm) / threshold_radius_cm
        > params.merger_boundary_fraction
    )


def _update_frequency_probes(
    params: BBHEvolverParams,
    state: BinaryEvolutionState,
    *,
    previous_sma_cm: float | None = None,
    previous_ecc: float | None = None,
) -> None:
    if previous_sma_cm is None or previous_ecc is None:
        freq_gw = pq.frequency_from_sma(state.m1_g, state.m2_g, state.sma_cm)
        if not state.reached_0p1hz and freq_gw >= params.eccentricity_probe_low_hz:
            state.ecc0p1 = float(state.ecc)
            state.reached_0p1hz = True
        if not state.reached_10hz and freq_gw >= params.eccentricity_probe_high_hz:
            state.ecc10 = float(state.ecc)
            state.reached_10hz = True
        return

    if not state.reached_0p1hz:
        eccentricity_at_0p1 = pq.compute_eccentricity_at_frequency(
            state.m1_g,
            state.m2_g,
            state.sma_cm,
            state.ecc,
            target_freq=params.eccentricity_probe_low_hz,
            previous_a=previous_sma_cm,
            previous_e=previous_ecc,
        )
        if np.isfinite(eccentricity_at_0p1):
            state.ecc0p1 = float(eccentricity_at_0p1)
            state.reached_0p1hz = True

    if not state.reached_10hz:
        eccentricity_at_10 = pq.compute_eccentricity_at_frequency(
            state.m1_g,
            state.m2_g,
            state.sma_cm,
            state.ecc,
            target_freq=params.eccentricity_probe_high_hz,
            previous_a=previous_sma_cm,
            previous_e=previous_ecc,
        )
        if np.isfinite(eccentricity_at_10):
            state.ecc10 = float(eccentricity_at_10)
            state.reached_10hz = True


def _bsi_is_active(
    bbh: BBH_properties,
    i: int,
    *,
    bsi_enabled: bool,
    t_ini_s: float,
    delay_s: float,
    duty_cycle_s: float,
) -> bool:
    return (
        bsi_enabled
        and bbh.bsi_applied_flag[i] == 0
        and delay_s + t_ini_s <= duty_cycle_s
    )


def _record_bsi_jump_if_applied(
    params: BBHEvolverParams,
    bbh: BBH_properties,
    i: int,
    state: BinaryEvolutionState,
    recorder: DecimatedTrackRecorder,
    t_ini_s: float,
    was_applied: int | np.integer,
    previous_sma_cm: float,
    previous_ecc: float,
) -> None:
    if bbh.bsi_applied_flag[i] == was_applied:
        return

    recorder.record(
        t_ini_s + float(bbh.bsi_timescale_s[i]),
        state.sma_cm,
        state.ecc,
        force=True,
    )
    _update_frequency_probes(
        params,
        state,
        previous_sma_cm=previous_sma_cm,
        previous_ecc=previous_ecc,
    )


def _apply_bsi_step(
    params: BBHEvolverParams,
    bbh: BBH_properties,
    i: int,
    state: BinaryEvolutionState,
    *,
    elapsed_s: float,
    step_s: float,
    bsi_active: bool,
) -> int | None:
    (
        state.sma_cm,
        state.ecc,
        state.m1_g,
        state.m2_g,
        bsi_stop_reason,
    ) = bsi.apply_scheduled_bsi_step(
        params,
        bbh,
        bbh.flags,
        i,
        state.sma_cm,
        state.ecc,
        state.m1_g,
        state.m2_g,
        elapsed_s,
        step_s,
        bsi_active=bsi_active,
    )
    return bsi_stop_reason


def _limit_smooth_step_for_bsi(
    bbh: BBH_properties,
    i: int,
    state: BinaryEvolutionState,
    remaining_time_s: float,
    current_step_limit_s: float,
) -> tuple[float, bool]:
    bsi_delay_s = float(bbh.bsi_timescale_s[i])
    if not (np.isfinite(bsi_delay_s) and bsi_delay_s >= state.delay_s):
        return current_step_limit_s, False

    time_to_bsi_s = bsi_delay_s - state.delay_s
    if time_to_bsi_s <= 0.0:
        return current_step_limit_s, False
    return min(current_step_limit_s, time_to_bsi_s), time_to_bsi_s <= remaining_time_s


def _handle_due_bsi_at_step_start(
    params: BBHEvolverParams,
    bbh: BBH_properties,
    i: int,
    state: BinaryEvolutionState,
    recorder: DecimatedTrackRecorder,
    *,
    t_ini_s: float,
    bsi_active: bool,
) -> int | None:
    if not bsi_active:
        return None

    bsi_delay_s = float(bbh.bsi_timescale_s[i])
    if not (np.isfinite(bsi_delay_s) and bsi_delay_s >= state.delay_s):
        return None

    if bsi_delay_s - state.delay_s > 0.0:
        return None

    was_applied = bbh.bsi_applied_flag[i]
    previous_sma_cm = state.sma_cm
    previous_ecc = state.ecc
    bsi_stop_reason = _apply_bsi_step(
        params,
        bbh,
        i,
        state,
        elapsed_s=state.delay_s,
        step_s=0.0,
        bsi_active=True,
    )
    _record_bsi_jump_if_applied(
        params,
        bbh,
        i,
        state,
        recorder,
        t_ini_s,
        was_applied,
        previous_sma_cm,
        previous_ecc,
    )
    return bsi_stop_reason


def _record_final_state(
    bbh: BBH_properties,
    i: int,
    state: BinaryEvolutionState,
    recorder: DecimatedTrackRecorder,
    final_time_s: float,
) -> None:
    recorder.record_final(final_time_s, state.sma_cm, state.ecc)
    bbh.sma_cm[i] = state.sma_cm
    bbh.ecc[i] = state.ecc
    bbh.ecc10[i] = state.ecc10
    bbh.ecc0p1[i] = state.ecc0p1
    append_bbh_track(
        bbh,
        binary_idx=i,
        time_s=recorder.time_s,
        sma_cm=recorder.sma_cm,
        ecc=recorder.ecc,
    )


def evolve_binary_orbit(
    params: BBHEvolverParams,
    bbh: BBH_properties,
    agn: AGN_properties,
    i: int,
    *,
    t_ini_s: float,
    initial_step_fraction: float,
    ode_func: OdeFunction,
    ode_args: tuple[Any, ...] = (),
    bsi_enabled: bool = False,
    merger_reachability_guard: MergerReachabilityGuard | None = None,
    segment_stop_s: float | None = None,
) -> float:
    """Integrate one BBH orbit and own all mutable bookkeeping side effects."""
    flags = bbh.flags
    flags.trap_reached_flag[i] = 1
    state = _binary_state_from_bbh(bbh, i)
    time_limit = build_evolution_time_limit(agn)
    current_abs_s = t_ini_s + state.delay_s
    stop_s = time_limit.stop_s
    segment_mode = False
    if segment_stop_s is not None and np.isfinite(segment_stop_s):
        requested_stop_s = float(segment_stop_s)
        if requested_stop_s > current_abs_s:
            segment_mode = requested_stop_s < time_limit.stop_s
            stop_s = min(stop_s, requested_stop_s)

    t_est_s = pq.compute_merger_time_estimate(
        state.m1_g,
        state.m2_g,
        state.sma_cm,
        state.ecc,
    )
    h = float(initial_step_fraction * t_est_s)
    r_th = pq.threshold_radius(state.m1_g, state.m2_g)
    merger_radius_cm = params.merger_radius_factor * r_th

    recorder = DecimatedTrackRecorder(
        start_time_s=current_abs_s,
        start_sma_cm=state.sma_cm,
        start_ecc=state.ecc,
        stop_time_s=stop_s,
        max_smooth_points=params.track_max_smooth_points,
    )
    _update_frequency_probes(params, state)

    loop_num = 0
    max_loops = params.max_evolution_loops

    def record_smooth_state(elapsed_s, sma_cm, eccentricity):
        recorder.record(
            t_ini_s + state.delay_s + elapsed_s,
            sma_cm,
            eccentricity,
        )

    while (
        _outside_merger_boundary(params, state.sma_cm, r_th)
        and loop_num < max_loops
    ):
        if stop_cond.stop_for_walltime(flags, i):
            break

        remaining_time_s = stop_s - (t_ini_s + state.delay_s)
        if remaining_time_s <= 0.0:
            if segment_mode:
                flags.continue_ng_flag[i] = 1
            else:
                _mark_non_merger_stop(bbh, i, time_limit.stop_reason)
            break

        smooth_step_limit_s = remaining_time_s
        bsi_active = _bsi_is_active(
            bbh,
            i,
            bsi_enabled=bsi_enabled,
            t_ini_s=t_ini_s,
            delay_s=state.delay_s,
            duty_cycle_s=time_limit.duty_cycle_s,
        )
        bsi_stop_reason = _handle_due_bsi_at_step_start(
            params,
            bbh,
            i,
            state,
            recorder,
            t_ini_s=t_ini_s,
            bsi_active=bsi_active,
        )
        if bsi_stop_reason is not None:
            _mark_non_merger_stop(bbh, i, bsi_stop_reason)
            break

        bsi_event_before_limit = False
        if bsi_active:
            smooth_step_limit_s, bsi_event_before_limit = _limit_smooth_step_for_bsi(
                bbh,
                i,
                state,
                remaining_time_s,
                smooth_step_limit_s,
            )

        if (
            merger_reachability_guard is not None
            and not bsi_event_before_limit
            and not segment_mode
        ):
            if not merger_reachability_guard(state, merger_radius_cm, remaining_time_s):
                state.delay_s = max(stop_s - t_ini_s, 0.0)
                _mark_non_merger_stop(bbh, i, time_limit.stop_reason)
                recorder.record(
                    t_ini_s + state.delay_s,
                    state.sma_cm,
                    state.ecc,
                    force=True,
                )
                break

        _update_frequency_probes(params, state)
        loop_num += 1
        flags.continue_ng_flag[i] = 1

        previous_sma_cm = state.sma_cm
        previous_ecc = state.ecc
        a_new, e_new, h = numeric.para_evo(
            params,
            state.m1_g,
            state.m2_g,
            state.sma_cm,
            state.ecc,
            h,
            ode_func,
            *ode_args,
            max_step_s=smooth_step_limit_s,
            state_callback=record_smooth_state,
        )
        if not (
            numeric.is_physical_state(params, a_new, e_new)
            and np.isfinite(h)
            and h > 0.0
        ):
            stop_cond.stop_for_invalid_step(flags, i)
            break

        state.sma_cm = float(a_new)
        state.ecc = float(e_new)
        _update_frequency_probes(
            params,
            state,
            previous_sma_cm=previous_sma_cm,
            previous_ecc=previous_ecc,
        )

        was_applied = bbh.bsi_applied_flag[i]
        previous_sma_cm = state.sma_cm
        previous_ecc = state.ecc
        bsi_active = _bsi_is_active(
            bbh,
            i,
            bsi_enabled=bsi_enabled,
            t_ini_s=t_ini_s,
            delay_s=state.delay_s,
            duty_cycle_s=time_limit.duty_cycle_s,
        )
        bsi_stop_reason = _apply_bsi_step(
            params,
            bbh,
            i,
            state,
            elapsed_s=state.delay_s,
            step_s=h,
            bsi_active=bsi_active,
        )
        _record_bsi_jump_if_applied(
            params,
            bbh,
            i,
            state,
            recorder,
            t_ini_s,
            was_applied,
            previous_sma_cm,
            previous_ecc,
        )

        if bsi_stop_reason is not None:
            _mark_non_merger_stop(bbh, i, bsi_stop_reason)
            state.delay_s = float(bbh.bsi_timescale_s[i])
            break

        r_th = pq.threshold_radius(state.m1_g, state.m2_g)
        merger_radius_cm = params.merger_radius_factor * r_th
        state.delay_s += h

        if state.delay_s + t_ini_s > time_limit.hubble_s:
            _mark_non_merger_stop(bbh, i, StopReasons.STOP_HUBBLE_LIMIT)
            break

        if not _outside_merger_boundary(params, state.sma_cm, r_th):
            flags.merger_flag[i] = 1
            break

        if segment_mode and state.delay_s + t_ini_s >= stop_s:
            flags.continue_ng_flag[i] = 1
            break

        if state.delay_s + t_ini_s >= time_limit.duty_cycle_s:
            _mark_non_merger_stop(bbh, i, time_limit.stop_reason)
            break

        if loop_num >= max_loops:
            print(f"Warning: Max loops reached for binary {i}")
            _mark_non_merger_stop(bbh, i, StopReasons.STOP_INTEGRATION_LIMIT)
            break

    final_time_s = float(t_ini_s + state.delay_s)
    _record_final_state(bbh, i, state, recorder, final_time_s)
    return state.delay_s
