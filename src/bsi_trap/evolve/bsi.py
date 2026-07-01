from __future__ import annotations

import numpy as np

from bsi_trap.classes.models import BBH_properties
from bsi_trap.constant.constant import Myr
from bsi_trap.evolve.timeline import BsiTimeline


def _resolve_bsi_delay_s(
    bbh: BBH_properties,
    event_delay_s: np.ndarray | None,
) -> np.ndarray | None:
    if event_delay_s is not None:
        delay_s = np.asarray(event_delay_s, dtype=float)
    elif hasattr(bbh, "bsi_timescale_s"):
        delay_s = np.asarray(bbh.bsi_timescale_s, dtype=float)
    elif hasattr(bbh, "t_bsi_s"):
        delay_s = np.asarray(bbh.t_bsi_s, dtype=float)
    elif hasattr(bbh, "t_bsi_myr"):
        delay_s = np.asarray(bbh.t_bsi_myr, dtype=float) * Myr
    else:
        return None

    if delay_s.shape != (bbh.num,):
        raise ValueError(f"BSI delay array must have shape ({bbh.num},)")
    return delay_s


def construct_bsi_timeline(
    bbh: BBH_properties,
    event_delay_s: np.ndarray | None = None,
) -> BsiTimeline:
    """Build absolute-time BSI events from recorded flags and event delays.

    The current evolution code does not yet resolve BSI encounter times.  When
    no BSI delay/timescale field is supplied, this function returns an empty
    timeline instead of assigning artificial event times.
    """
    if bbh.num == 0:
        return BsiTimeline.empty()

    delay_s = _resolve_bsi_delay_s(bbh, event_delay_s)
    if delay_s is None:
        return BsiTimeline.empty()

    valid_time = np.isfinite(delay_s) & (delay_s >= 0.0)
    flags = bbh.flags
    strong_mask = (
        (flags.bsi_exchange_flag != 0)
        | (flags.bsi_ionization_flag != 0)
        | (flags.bsi_ejection_flag != 0)
    )

    event_indices: list[np.ndarray] = []
    event_types: list[np.ndarray] = []

    weak_mask = valid_time & (flags.bsi_event_flag != 0) & ~strong_mask
    if np.any(weak_mask):
        weak_idx = np.flatnonzero(weak_mask)
        event_indices.append(weak_idx)
        event_types.append(np.full(weak_idx.size, "weak", dtype="<U16"))

    for flag_name, event_type in (
        ("bsi_exchange_flag", "exchange"),
        ("bsi_ionization_flag", "ionization"),
        ("bsi_ejection_flag", "ejection"),
    ):
        mask = valid_time & (getattr(flags, flag_name) != 0)
        if not np.any(mask):
            continue
        indices = np.flatnonzero(mask)
        event_indices.append(indices)
        event_types.append(np.full(indices.size, event_type, dtype="<U16"))

    if not event_indices:
        return BsiTimeline.empty()

    binary_idx = np.concatenate(event_indices).astype(np.intp, copy=False)
    event_type = np.concatenate(event_types)
    event_s = bbh.t_ini_s[binary_idx] + delay_s[binary_idx]
    order = np.lexsort((binary_idx, event_s))

    return BsiTimeline(
        binary_idx=binary_idx[order],
        event_s=event_s[order],
        event_type=event_type[order],
    )
