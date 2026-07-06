"""BSI event-ledger construction.

The physical BSI module schedules and applies state changes.  This module only
translates the recorded applied-event flags and delays into absolute event
times for storage and analysis.
"""

from __future__ import annotations

import numpy as np

from bsi_trap.classes.models import BBH_properties
from bsi_trap.constant.constant import Myr

from .models import BSI_Timeline, FloatArray, IndexArray


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
    """Build absolute-time BSI events from recorded flags and event delays."""

    if bbh.num == 0:
        return BSI_Timeline.empty()

    delay_s = _resolve_bsi_delay_s(bbh, event_delay_s)
    if delay_s is None:
        return BSI_Timeline.empty()

    valid_time = np.isfinite(delay_s) & (delay_s >= 0.0)
    flags = bbh.flags
    weak_mask = valid_time & (flags.bsi_event_flag != 0)
    if not np.any(weak_mask):
        return BSI_Timeline.empty()

    binary_idx: IndexArray = np.flatnonzero(weak_mask).astype(
        np.intp,
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
    )
