from __future__ import annotations

import numpy as np

from bsi_trap.classes.models import BBH_properties


def append_bbh_track(
    bbh: BBH_properties,
    *,
    binary_idx: int,
    time_s: list[float] | np.ndarray,
    sma_cm: list[float] | np.ndarray,
    ecc: list[float] | np.ndarray,
) -> None:
    """Append one accepted evolution path to the generation-level BBH ledger."""
    time = np.asarray(time_s, dtype=np.float64)
    sma = np.asarray(sma_cm, dtype=np.float64)
    eccentricity = np.asarray(ecc, dtype=np.float64)
    if not (time.size == sma.size == eccentricity.size):
        raise ValueError("BBH track arrays must have matching lengths")
    if time.size == 0:
        return

    mask = np.isfinite(time) & np.isfinite(sma) & np.isfinite(eccentricity)
    mask &= sma > 0.0
    if not np.any(mask):
        return

    time = time[mask]
    sma = sma[mask]
    eccentricity = eccentricity[mask]
    step_idx = np.arange(time.size, dtype=np.int64)
    binary_idx_arr = np.full(time.size, int(binary_idx), dtype=np.int64)

    bbh.track_binary_idx = np.concatenate((bbh.track_binary_idx, binary_idx_arr))
    bbh.track_step_idx = np.concatenate((bbh.track_step_idx, step_idx))
    bbh.track_time_s = np.concatenate((bbh.track_time_s, time))
    bbh.track_sma_cm = np.concatenate((bbh.track_sma_cm, sma))
    bbh.track_ecc = np.concatenate((bbh.track_ecc, eccentricity))
