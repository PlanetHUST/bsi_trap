"""Event timeline data contracts used by evolution stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.intp]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]
StringArray = npt.NDArray[np.str_]


@dataclass(frozen=True)
class PairingTimeline:
    """Resolved initial sBH-sBH pairing events.

    ``formation_s`` is the BBH creation timeline used by downstream binary
    evolution.  ``encounter_s`` and ``pair_delay_s`` preserve the two physical
    pieces that produced each formation time.
    """

    first_idx: IndexArray
    second_idx: IndexArray
    later_arrival_s: FloatArray
    encounter_s: FloatArray
    pair_delay_s: FloatArray
    formation_s: FloatArray

    def __post_init__(self) -> None:
        expected_shape = self.formation_s.shape
        for name in (
            "first_idx",
            "second_idx",
            "later_arrival_s",
            "encounter_s",
            "pair_delay_s",
        ):
            if getattr(self, name).shape != expected_shape:
                raise ValueError("PairingTimeline arrays must have matching shapes")
        if len(expected_shape) != 1:
            raise ValueError("PairingTimeline arrays must be one-dimensional")
        if np.any(self.pair_delay_s < 0.0):
            raise ValueError("PairingTimeline delays must be nonnegative")
        if np.any(self.encounter_s < self.later_arrival_s):
            raise ValueError("Pairing encounters cannot precede component arrival")
        if not np.allclose(self.formation_s, self.encounter_s + self.pair_delay_s):
            raise ValueError("Pairing formation times must equal encounter plus delay")

    @classmethod
    def empty(cls) -> "PairingTimeline":
        empty_idx = np.empty(0, dtype=np.intp)
        empty_float = np.empty(0, dtype=float)
        return cls(
            first_idx=empty_idx,
            second_idx=empty_idx,
            later_arrival_s=empty_float,
            encounter_s=empty_float,
            pair_delay_s=empty_float,
            formation_s=empty_float,
        )

    @property
    def bbh_pairing_up_s(self) -> FloatArray:
        """Compatibility view for the BBH formation timeline."""
        return self.formation_s

    @property
    def size(self) -> int:
        return int(self.formation_s.size)


@dataclass(frozen=True)
class BsiTimeline:
    """Resolved binary-single interaction events with absolute event times."""

    binary_idx: IndexArray
    event_s: FloatArray
    event_type: StringArray

    def __post_init__(self) -> None:
        expected_shape = self.event_s.shape
        for name in ("binary_idx", "event_type"):
            if getattr(self, name).shape != expected_shape:
                raise ValueError("BsiTimeline arrays must have matching shapes")
        if len(expected_shape) != 1:
            raise ValueError("BsiTimeline arrays must be one-dimensional")
        if np.any(~np.isfinite(self.event_s)) or np.any(self.event_s < 0.0):
            raise ValueError("BsiTimeline event times must be finite and nonnegative")

    @classmethod
    def empty(cls) -> "BsiTimeline":
        return cls(
            binary_idx=np.empty(0, dtype=np.intp),
            event_s=np.empty(0, dtype=float),
            event_type=np.empty(0, dtype="<U16"),
        )

    @property
    def size(self) -> int:
        return int(self.event_s.size)

    @property
    def weak_perturbation_s(self) -> FloatArray:
        return self.event_s[self.event_type == "weak"]

    @property
    def strong_perturbation_s(self) -> FloatArray:
        return self.event_s[self.event_type != "weak"]


@dataclass(frozen=True)
class BbhEvolutionTimeline:
    """Absolute-time outcome timeline for one evolved BBH generation."""

    binary_idx: IndexArray
    start_s: FloatArray
    delay_s: FloatArray
    end_s: FloatArray
    merger_s: FloatArray
    merged: BoolArray
    stop_reason: IntArray
    bsi_timeline: BsiTimeline

    def __post_init__(self) -> None:
        expected_shape = self.start_s.shape
        for name in (
            "binary_idx",
            "delay_s",
            "end_s",
            "merger_s",
            "merged",
            "stop_reason",
        ):
            if getattr(self, name).shape != expected_shape:
                raise ValueError(
                    "BbhEvolutionTimeline arrays must have matching shapes"
                )
        if len(expected_shape) != 1:
            raise ValueError("BbhEvolutionTimeline arrays must be one-dimensional")
        if np.any(~np.isfinite(self.start_s)):
            raise ValueError("BBH evolution start times must be finite")
        if np.any(~np.isfinite(self.delay_s)) or np.any(self.delay_s < 0.0):
            raise ValueError("BBH evolution delays must be finite and nonnegative")
        if not np.allclose(self.end_s, self.start_s + self.delay_s):
            raise ValueError("BBH evolution end times must equal start plus delay")
        merger_times = self.merger_s[self.merged]
        if np.any(~np.isfinite(merger_times)):
            raise ValueError("Merged BBHs must have finite merger times")
        if np.any(np.isfinite(self.merger_s[~self.merged])):
            raise ValueError("Unmerged BBHs must not have merger times")

    @classmethod
    def empty(cls, bsi_timeline: BsiTimeline | None = None) -> "BbhEvolutionTimeline":
        empty_idx = np.empty(0, dtype=np.intp)
        empty_float = np.empty(0, dtype=float)
        return cls(
            binary_idx=empty_idx,
            start_s=empty_float,
            delay_s=empty_float,
            end_s=empty_float,
            merger_s=empty_float,
            merged=np.empty(0, dtype=bool),
            stop_reason=np.empty(0, dtype=np.int64),
            bsi_timeline=BsiTimeline.empty() if bsi_timeline is None else bsi_timeline,
        )

    @property
    def size(self) -> int:
        return int(self.start_s.size)

    @property
    def bbh_merge_s(self) -> FloatArray:
        return self.merger_s[self.merged]
