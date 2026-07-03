from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from bsi_trap.constant.constant import Myr
from bsi_trap.storage.contract import HDF5Contract


@dataclass(frozen=True)
class TimelineCatalog:
    trap_reach_s: np.ndarray
    trap_reach_sample: np.ndarray
    pair_formation_s: np.ndarray
    pair_sample: np.ndarray
    merger_s: np.ndarray
    merger_sample: np.ndarray
    start_s: np.ndarray
    end_s: np.ndarray
    sma_initial_cm: np.ndarray
    sma_final_cm: np.ndarray
    ecc_initial: np.ndarray
    ecc_final: np.ndarray
    remnant_mass_msun: np.ndarray
    kick_velocity_cgs: np.ndarray
    generation: np.ndarray
    sample_index: np.ndarray
    duty_cycle_s: np.ndarray
    merged: np.ndarray
    track_binary_idx: np.ndarray
    track_time_s: np.ndarray
    track_sma_cm: np.ndarray
    track_ecc: np.ndarray

    @property
    def binary_count(self) -> int:
        return int(self.start_s.size)

    @property
    def max_duty_cycle_s(self) -> float:
        values = self.duty_cycle_s[np.isfinite(self.duty_cycle_s)]
        if values.size == 0:
            return np.nan
        return float(np.max(values))

    @classmethod
    def empty(cls) -> "TimelineCatalog":
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int64)
        return cls(
            trap_reach_s=empty_float,
            trap_reach_sample=empty_int,
            pair_formation_s=empty_float,
            pair_sample=empty_int,
            merger_s=empty_float,
            merger_sample=empty_int,
            start_s=empty_float,
            end_s=empty_float,
            sma_initial_cm=empty_float,
            sma_final_cm=empty_float,
            ecc_initial=empty_float,
            ecc_final=empty_float,
            remnant_mass_msun=empty_float,
            kick_velocity_cgs=empty_float,
            generation=empty_int,
            sample_index=empty_int,
            duty_cycle_s=empty_float,
            merged=np.empty(0, dtype=bool),
            track_binary_idx=empty_int,
            track_time_s=empty_float,
            track_sma_cm=empty_float,
            track_ecc=empty_float,
        )

    @classmethod
    def concat(cls, catalogs: Sequence["TimelineCatalog"]) -> "TimelineCatalog":
        catalogs = [catalog for catalog in catalogs if not catalog.is_empty]
        if not catalogs:
            return cls.empty()

        return cls(
            trap_reach_s=_concat_float(catalog.trap_reach_s for catalog in catalogs),
            trap_reach_sample=_concat_int(
                catalog.trap_reach_sample for catalog in catalogs
            ),
            pair_formation_s=_concat_float(
                catalog.pair_formation_s for catalog in catalogs
            ),
            pair_sample=_concat_int(catalog.pair_sample for catalog in catalogs),
            merger_s=_concat_float(catalog.merger_s for catalog in catalogs),
            merger_sample=_concat_int(catalog.merger_sample for catalog in catalogs),
            start_s=_concat_float(catalog.start_s for catalog in catalogs),
            end_s=_concat_float(catalog.end_s for catalog in catalogs),
            sma_initial_cm=_concat_float(
                catalog.sma_initial_cm for catalog in catalogs
            ),
            sma_final_cm=_concat_float(catalog.sma_final_cm for catalog in catalogs),
            ecc_initial=_concat_float(catalog.ecc_initial for catalog in catalogs),
            ecc_final=_concat_float(catalog.ecc_final for catalog in catalogs),
            remnant_mass_msun=_concat_float(
                catalog.remnant_mass_msun for catalog in catalogs
            ),
            kick_velocity_cgs=_concat_float(
                catalog.kick_velocity_cgs for catalog in catalogs
            ),
            generation=_concat_int(catalog.generation for catalog in catalogs),
            sample_index=_concat_int(catalog.sample_index for catalog in catalogs),
            duty_cycle_s=_concat_float(catalog.duty_cycle_s for catalog in catalogs),
            merged=_concat_bool(catalog.merged for catalog in catalogs),
            track_binary_idx=_concat_int(
                _offset_track_indices(catalog, offset)
                for catalog, offset in _catalog_track_offsets(catalogs)
            ),
            track_time_s=_concat_float(catalog.track_time_s for catalog in catalogs),
            track_sma_cm=_concat_float(catalog.track_sma_cm for catalog in catalogs),
            track_ecc=_concat_float(catalog.track_ecc for catalog in catalogs),
        )

    @property
    def is_empty(self) -> bool:
        return (
            self.trap_reach_s.size == 0
            and self.pair_formation_s.size == 0
            and self.merger_s.size == 0
            and self.binary_count == 0
        )


def read_timeline_catalog(
    path: str | Path,
    contract: HDF5Contract = HDF5Contract(),
) -> TimelineCatalog:
    with h5py.File(Path(path), "r") as h5f:
        sample_index = _int_attr(h5f, contract.attr_sample_index, -1)
        generation = _int_attr(h5f, contract.attr_generation, -1)

        trap_group = contract.group_trap_population
        bbh_group = contract.group_bbh
        flags_group = contract.group_flags
        times_group = contract.group_timescales
        tracks_group = contract.group_bbh_tracks
        agn_group = contract.group_agn
        lifetime_myr = _group_float_attr(
            h5f,
            agn_group,
            "lifetime_myr",
            np.nan,
        )
        duty_cycle_s = lifetime_myr * Myr

        trap_reach_s = _finite_array(h5f, f"{trap_group}/embedded_bh/t_reach_s")
        trap_reach_s = _within_duty_cycle(trap_reach_s, duty_cycle_s)
        pair_formation_s = _finite_array(
            h5f,
            f"{trap_group}/pairing_timeline/formation_s",
        )
        pair_formation_s = _within_duty_cycle(pair_formation_s, duty_cycle_s)

        merger_flag = _array(h5f, f"{flags_group}/merger_flag", dtype=int)
        tini_myr = _array(h5f, f"{times_group}/tini_myr")
        tdel_myr = _array(h5f, f"{times_group}/tdel_myr")
        tmerg_myr = _array(h5f, f"{times_group}/tmerg_myr")
        binary_count = max(tini_myr.size, tdel_myr.size, merger_flag.size)
        merged = _resize(merger_flag, binary_count, dtype=int) == 1

        start_s = _resize(tini_myr, binary_count) * Myr
        end_s = (_resize(tini_myr, binary_count) + _resize(tdel_myr, binary_count)) * Myr
        merger_s = _resize(tmerg_myr, binary_count) * Myr
        if merged.size > 0:
            merger_s = merger_s[merged]
        merger_s = merger_s[np.isfinite(merger_s)]
        merger_s = _within_duty_cycle(merger_s, duty_cycle_s)
        track_time_s = _within_duty_cycle(
            _array(h5f, f"{tracks_group}/time_s"),
            duty_cycle_s,
        )
        raw_track_time_s = _array(h5f, f"{tracks_group}/time_s")
        track_mask = np.isfinite(raw_track_time_s)
        if np.isfinite(duty_cycle_s):
            track_mask &= raw_track_time_s <= duty_cycle_s

        return TimelineCatalog(
            trap_reach_s=trap_reach_s,
            trap_reach_sample=_full_sample(trap_reach_s.size, sample_index),
            pair_formation_s=pair_formation_s,
            pair_sample=_full_sample(pair_formation_s.size, sample_index),
            merger_s=merger_s,
            merger_sample=_full_sample(merger_s.size, sample_index),
            start_s=start_s,
            end_s=end_s,
            sma_initial_cm=_resize(
                _array(h5f, f"{bbh_group}/sma_initial"),
                binary_count,
            ),
            sma_final_cm=_resize(
                _array(h5f, f"{bbh_group}/sma_final"),
                binary_count,
            ),
            ecc_initial=_resize(
                _array(h5f, f"{bbh_group}/ecc_initial"),
                binary_count,
            ),
            ecc_final=_resize(_array(h5f, f"{bbh_group}/ecc_final"), binary_count),
            remnant_mass_msun=_resize(_array(h5f, f"{bbh_group}/mrem"), binary_count),
            kick_velocity_cgs=_resize(_array(h5f, f"{bbh_group}/vk_cgs"), binary_count),
            generation=np.full(binary_count, generation, dtype=np.int64),
            sample_index=np.full(binary_count, sample_index, dtype=np.int64),
            duty_cycle_s=np.full(binary_count, duty_cycle_s, dtype=np.float64),
            merged=merged,
            track_binary_idx=_array(h5f, f"{tracks_group}/binary_idx", dtype=int)[
                track_mask
            ],
            track_time_s=track_time_s,
            track_sma_cm=_array(h5f, f"{tracks_group}/sma_cm")[track_mask],
            track_ecc=_array(h5f, f"{tracks_group}/ecc")[track_mask],
        )


def read_timeline_catalogs(
    paths: Sequence[str | Path],
    contract: HDF5Contract = HDF5Contract(),
) -> TimelineCatalog:
    return TimelineCatalog.concat(
        [read_timeline_catalog(path, contract=contract) for path in paths]
    )


def _array(
    h5f: h5py.File,
    path: str,
    dtype: type = float,
) -> np.ndarray:
    node = h5f.get(path)
    if not isinstance(node, h5py.Dataset):
        return np.empty(0, dtype=dtype)
    return np.asarray(node[()], dtype=dtype)


def _finite_array(h5f: h5py.File, path: str) -> np.ndarray:
    values = _array(h5f, path)
    return values[np.isfinite(values)]


def _within_duty_cycle(values: np.ndarray, duty_cycle_s: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if np.isfinite(duty_cycle_s):
        values = values[values <= duty_cycle_s]
    return values


def _resize(
    values: np.ndarray,
    size: int,
    dtype: type = float,
) -> np.ndarray:
    fill_value = 0 if dtype is int else np.nan
    output = np.full(size, fill_value, dtype=dtype)
    values = np.asarray(values, dtype=dtype)
    copied = min(size, values.size)
    if copied > 0:
        output[:copied] = values[:copied]
    return output


def _full_sample(size: int, sample_index: int) -> np.ndarray:
    return np.full(size, sample_index, dtype=np.int64)


def _int_attr(h5f: h5py.File, name: str, default: int) -> int:
    return int(np.asarray(h5f.attrs.get(name, default)).item())


def _group_float_attr(
    h5f: h5py.File,
    group_name: str,
    attr_name: str,
    default: float,
) -> float:
    group = h5f.get(group_name)
    if not isinstance(group, h5py.Group):
        return default
    return float(np.asarray(group.attrs.get(attr_name, default)).item())


def _concat_float(arrays) -> np.ndarray:
    items = [np.asarray(array, dtype=np.float64) for array in arrays if array.size > 0]
    if not items:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(items)


def _concat_int(arrays) -> np.ndarray:
    items = [np.asarray(array, dtype=np.int64) for array in arrays if array.size > 0]
    if not items:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(items)


def _concat_bool(arrays) -> np.ndarray:
    items = [np.asarray(array, dtype=bool) for array in arrays if array.size > 0]
    if not items:
        return np.empty(0, dtype=bool)
    return np.concatenate(items)


def _catalog_track_offsets(
    catalogs: Sequence[TimelineCatalog],
) -> list[tuple[TimelineCatalog, int]]:
    offset = 0
    output: list[tuple[TimelineCatalog, int]] = []
    for catalog in catalogs:
        output.append((catalog, offset))
        offset += catalog.binary_count
    return output


def _offset_track_indices(catalog: TimelineCatalog, offset: int) -> np.ndarray:
    if catalog.track_binary_idx.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.asarray(catalog.track_binary_idx, dtype=np.int64) + int(offset)
