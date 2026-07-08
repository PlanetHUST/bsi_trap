from __future__ import annotations

import numpy as np

from bsi_trap.simulation.state.models import BBH_properties


def _empty_int_array() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


def _empty_float_array() -> np.ndarray:
    return np.empty(0, dtype=np.float64)


class DecimatedTrackRecorder:
    """Collect a sparse BBH evolution path with forced physical endpoints."""

    def __init__(
        self,
        *,
        start_time_s: float,
        start_sma_cm: float,
        start_ecc: float,
        stop_time_s: float,
        max_smooth_points: int,
    ) -> None:
        self.time_s = [float(start_time_s)]
        self.sma_cm = [float(start_sma_cm)]
        self.ecc = [float(start_ecc)]
        track_span_s = max(float(stop_time_s) - float(start_time_s), 0.0)
        self._min_interval_s = (
            track_span_s / max_smooth_points
            if track_span_s > 0.0 and max_smooth_points > 0
            else 0.0
        )
        self._start_time_s = float(start_time_s)
        self._next_time_s = float(start_time_s + self._min_interval_s)
        self._smooth_record_count = 0

    def record(
        self,
        time_s: float,
        sma_cm: float,
        eccentricity: float,
        *,
        force: bool = False,
    ) -> None:
        """Record smooth states on cadence, but always keep forced states."""
        if not (
            np.isfinite(time_s)
            and np.isfinite(sma_cm)
            and np.isfinite(eccentricity)
        ):
            return

        first_smooth_state = self._smooth_record_count == 0 and not force
        cadence_reached = (
            self._min_interval_s <= 0.0 or time_s + 1.0e-9 >= self._next_time_s
        )
        if not (force or first_smooth_state or cadence_reached):
            return

        self.time_s.append(float(time_s))
        self.sma_cm.append(float(sma_cm))
        self.ecc.append(float(eccentricity))
        if not force:
            self._smooth_record_count += 1
        if self._min_interval_s > 0.0:
            intervals_passed = max(
                1.0,
                np.floor((time_s - self._start_time_s) / self._min_interval_s) + 1.0,
            )
            self._next_time_s = float(
                self._start_time_s + intervals_passed * self._min_interval_s
            )

    def record_final(
        self,
        time_s: float,
        sma_cm: float,
        eccentricity: float,
    ) -> None:
        if (
            self.time_s[-1] != float(time_s)
            or self.sma_cm[-1] != float(sma_cm)
            or self.ecc[-1] != float(eccentricity)
        ):
            self.record(time_s, sma_cm, eccentricity, force=True)


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

    bbh._track_binary_idx_chunks.append(binary_idx_arr)
    bbh._track_step_idx_chunks.append(step_idx)
    bbh._track_time_s_chunks.append(time)
    bbh._track_sma_cm_chunks.append(sma)
    bbh._track_ecc_chunks.append(eccentricity)
    if bbh.num <= 1:
        finalize_bbh_tracks(bbh)


def finalize_bbh_tracks(bbh: BBH_properties) -> None:
    """Materialize staged BBH track chunks into contiguous arrays once."""
    if len(bbh._track_time_s_chunks) == 0:
        if bbh.track_time_s.size == 0:
            bbh.track_binary_idx = _empty_int_array()
            bbh.track_step_idx = _empty_int_array()
            bbh.track_time_s = _empty_float_array()
            bbh.track_sma_cm = _empty_float_array()
            bbh.track_ecc = _empty_float_array()
        return

    bbh.track_binary_idx = np.concatenate(bbh._track_binary_idx_chunks)
    bbh.track_step_idx = np.concatenate(bbh._track_step_idx_chunks)
    bbh.track_time_s = np.concatenate(bbh._track_time_s_chunks)
    bbh.track_sma_cm = np.concatenate(bbh._track_sma_cm_chunks)
    bbh.track_ecc = np.concatenate(bbh._track_ecc_chunks)
