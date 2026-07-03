from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .contract import HDF5Contract


@dataclass(frozen=True)
class GenerationSummary:
    sample_index: int
    generation: int
    smbh_mass_msun: float
    bbh_count: int
    merger_count: int
    embedded_count: int
    single_count: int
    pairing_event_count: int
    m1_msun: np.ndarray
    m2_msun: np.ndarray
    sma_initial_cm: np.ndarray
    sma_cm: np.ndarray
    ecc_initial: np.ndarray
    ecc: np.ndarray
    tdel_myr: np.ndarray
    tini_myr: np.ndarray
    tmerg_myr: np.ndarray
    merged: np.ndarray


def _array(h5f: h5py.File, path: str, dtype: type = float) -> np.ndarray:
    node = h5f.get(path)
    if not isinstance(node, h5py.Dataset):
        return np.asarray([], dtype=dtype)
    return np.asarray(node[()], dtype=dtype)


def _count(h5f: h5py.File, path: str) -> int:
    node = h5f.get(path)
    if not isinstance(node, h5py.Dataset):
        return 0
    return int(np.asarray(node[()]).size)


def _int_attr(h5f: h5py.File, name: str, default: int) -> int:
    value = h5f.attrs.get(name, default)
    return int(np.asarray(value).item())


def _float_attr(h5f: h5py.File, name: str, default: float) -> float:
    value = h5f.attrs.get(name, default)
    return float(np.asarray(value).item())


def read_generation_summary(
    path: str | Path,
    contract: HDF5Contract = HDF5Contract(),
) -> GenerationSummary:
    file_path = Path(path)
    with h5py.File(file_path, "r") as h5f:
        bbh_group = contract.group_bbh
        flags_group = contract.group_flags
        times_group = contract.group_timescales
        trap_group = contract.group_trap_population

        m1 = _array(h5f, f"{bbh_group}/m1")
        m2 = _array(h5f, f"{bbh_group}/m2")
        merged = _array(h5f, f"{flags_group}/merger_flag", dtype=int).astype(bool)

        return GenerationSummary(
            sample_index=_int_attr(h5f, contract.attr_sample_index, -1),
            generation=_int_attr(h5f, contract.attr_generation, -1),
            smbh_mass_msun=_float_attr(
                h5f,
                contract.attr_smbh_mass_msun,
                np.nan,
            ),
            bbh_count=int(m1.size),
            merger_count=int(np.count_nonzero(merged)),
            embedded_count=_count(h5f, f"{trap_group}/embedded_bh/m_msun"),
            single_count=_count(h5f, f"{trap_group}/singles/m_msun"),
            pairing_event_count=_count(
                h5f,
                f"{trap_group}/pairing_timeline/formation_s",
            ),
            m1_msun=m1,
            m2_msun=m2,
            sma_initial_cm=_array(h5f, f"{bbh_group}/sma_initial"),
            sma_cm=_array(h5f, f"{bbh_group}/sma"),
            ecc_initial=_array(h5f, f"{bbh_group}/ecc_initial"),
            ecc=_array(h5f, f"{bbh_group}/ecc"),
            tdel_myr=_array(h5f, f"{times_group}/tdel_myr"),
            tini_myr=_array(h5f, f"{times_group}/tini_myr"),
            tmerg_myr=_array(h5f, f"{times_group}/tmerg_myr"),
            merged=merged,
        )
