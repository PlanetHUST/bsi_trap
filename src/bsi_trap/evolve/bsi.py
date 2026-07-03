"""Reduced binary-single interaction updates and event-ledger construction.

The important model boundary is:

* ``apply_scheduled_bsi_step`` changes the mutable BBH state during evolution.
* ``construct_bsi_timeline`` reads the recorded flags/times and builds a ledger.

This keeps the physical update out of ``BSI_Timeline``.  Timelines answer
"when did the event happen?", while this module's step function answers "what
did that event do to the binary?".
"""

from __future__ import annotations

import numpy as np

from bsi_trap.classes.models import BBHEvolverParams, BBH_flags, BBH_properties
from bsi_trap.constant.constant import Myr
from bsi_trap.constant.stop_reason import StopReasons
from bsi_trap.evolve.timeline import (
    BSI_Timeline,
    FloatArray,
    IndexArray,
    StringArray,
)


def _event_crossed(delay_s: float, elapsed_s: float, step_s: float) -> bool:
    """Return True when an integration step crosses a scheduled BSI delay."""
    if not (
        np.isfinite(delay_s)
        and delay_s >= 0.0
        and np.isfinite(elapsed_s)
        and np.isfinite(step_s)
        and step_s >= 0.0
    ):
        return False
    return elapsed_s <= delay_s <= elapsed_s + step_s


def _apply_weak_perturbation(
    params: BBHEvolverParams,
    bbh: BBH_properties,
    i: int,
    a_cm: float,
    e: float,
) -> tuple[float, float]:
    """Apply the current reduced weak-BSI prescription to one binary.

    The weak channel is intentionally simple: it hardens the binary, pumps
    eccentricity, and perturbs the spin-tilt angles.  It is a placeholder for a
    future scattering/outcome kernel, but it already changes the physical BBH
    state rather than merely marking a timeline event.
    """
    delta_e = min(
        params.bsi_sigma_e0 * max(1.0 - e, params.ecc_floor),
        max(params.ecc_ceiling - e, 0.0),
    )
    delta_theta = min(delta_e, 0.5)

    e_after = float(np.clip(e + delta_e, 0.0, params.ecc_ceiling))
    a_after = float(max(a_cm * (1.0 - 0.5 * delta_e), np.finfo(float).tiny))

    bbh.bsi_weak_count[i] += 1
    bbh.bsi_delta_e_total[i] += delta_e
    bbh.bsi_delta_theta_total[i] += delta_theta
    bbh.th1[i] += delta_theta
    bbh.th2[i] -= delta_theta
    return a_after, e_after


def apply_scheduled_bsi_step(
    params: BBHEvolverParams,
    bbh: BBH_properties,
    flags: BBH_flags,
    i: int,
    a_cm: float,
    e: float,
    m1_g: float,
    m2_g: float,
    elapsed_s: float,
    step_s: float,
    *,
    bsi_active: bool,
) -> tuple[float, float, float, float, int | None]:
    """Apply the scheduled BSI effect when the current integration step crosses it.

    ``BSI_Timeline`` is only the ledger.  This function is the reduced local
    binary update: weak events perturb eccentricity/orientation and harden the
    binary, while ionization/ejection events stop the binary evolution.
    """
    if not bsi_active or bbh.bsi_applied_flag[i] != 0:
        return a_cm, e, m1_g, m2_g, None

    # ``bsi_timescale_s`` is a delay measured from BBH formation.  The gas/GW
    # integrators pass their current elapsed time and accepted step size, so an
    # event is applied exactly once when that step crosses the scheduled delay.
    delay_s = float(bbh.bsi_timescale_s[i])
    if not _event_crossed(delay_s, elapsed_s, step_s):
        return a_cm, e, m1_g, m2_g, None

    bbh.bsi_applied_flag[i] = 1
    flags.bsi_event_flag[i] = 1

    # Strong channels terminate this binary's evolution.  The timeline will
    # later classify the event from these flags; the stop reason is returned to
    # the integrator so it can end at the BSI time.
    if flags.bsi_ionization_flag[i] != 0:
        return a_cm, e, m1_g, m2_g, StopReasons.STOP_BSI_IONIZATION

    if flags.bsi_ejection_flag[i] != 0:
        return a_cm, e, m1_g, m2_g, StopReasons.STOP_BSI_EJECTION

    a_after, e_after = _apply_weak_perturbation(params, bbh, i, a_cm, e)
    return a_after, e_after, m1_g, m2_g, None


def _resolve_bsi_delay_s(
    bbh: BBH_properties,
    event_delay_s: np.ndarray | None,
) -> np.ndarray | None:
    """Return the BSI delay array used by the event ledger."""
    if event_delay_s is not None:
        delay_s = np.asarray(event_delay_s, dtype=float)
    else:
        delay_source = getattr(bbh, "bsi_timescale_s", None)
        if delay_source is not None:
            delay_s = np.asarray(delay_source, dtype=float)
        elif (delay_source := getattr(bbh, "t_bsi_s", None)) is not None:
            delay_s = np.asarray(delay_source, dtype=float)
        elif (delay_source := getattr(bbh, "t_bsi_myr", None)) is not None:
            delay_s = np.asarray(delay_source, dtype=float) * Myr
        else:
            return None

    if delay_s.shape != (bbh.num,):
        raise ValueError(f"BSI delay array must have shape ({bbh.num},)")
    return np.asarray(delay_s, dtype=np.float64)


def construct_bsi_timeline(
    bbh: BBH_properties,
    event_delay_s: np.ndarray | None = None,
) -> BSI_Timeline:
    """Build absolute-time BSI events from recorded flags and event delays.

    This function does not update orbital parameters.  It only translates the
    recorded BSI flags and delay fields into absolute event times,
    ``event_s = t_ini_s + delay_s``.  If no delay field is available it returns
    an empty ledger instead of assigning artificial event times.
    """
    if bbh.num == 0:
        return BSI_Timeline.empty()

    delay_s = _resolve_bsi_delay_s(bbh, event_delay_s)
    if delay_s is None:
        return BSI_Timeline.empty()

    valid_time = np.isfinite(delay_s) & (delay_s >= 0.0)
    flags = bbh.flags
    strong_mask = (
        (flags.bsi_exchange_flag != 0)
        | (flags.bsi_ionization_flag != 0)
        | (flags.bsi_ejection_flag != 0)
    )

    event_indices: list[IndexArray] = []
    event_types: list[StringArray] = []

    weak_mask = valid_time & (flags.bsi_event_flag != 0) & ~strong_mask
    if np.any(weak_mask):
        weak_idx = np.flatnonzero(weak_mask)
        event_indices.append(weak_idx.astype(np.intp, copy=False))
        event_types.append(np.full(weak_idx.size, "weak", dtype="<U16"))

    for flag_name, event_kind in (
        ("bsi_exchange_flag", "exchange"),
        ("bsi_ionization_flag", "ionization"),
        ("bsi_ejection_flag", "ejection"),
    ):
        mask = valid_time & (getattr(flags, flag_name) != 0)
        if not np.any(mask):
            continue
        indices = np.flatnonzero(mask)
        event_indices.append(indices.astype(np.intp, copy=False))
        event_types.append(np.full(indices.size, event_kind, dtype="<U16"))

    if not event_indices:
        return BSI_Timeline.empty()

    binary_idx: IndexArray = np.concatenate(event_indices).astype(
        np.intp,
        copy=False,
    )
    event_type: StringArray = np.concatenate(event_types).astype(
        "<U16",
        copy=False,
    )
    event_s: FloatArray = np.asarray(
        bbh.t_ini_s[binary_idx] + delay_s[binary_idx],
        dtype=np.float64,
    )
    order: IndexArray = np.asarray(
        np.lexsort((binary_idx, event_s)),
        dtype=np.intp,
    )

    return BSI_Timeline(
        binary_idx=binary_idx[order],
        event_s=event_s[order],
        event_type=event_type[order],
    )
