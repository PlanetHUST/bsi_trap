from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from bsi_trap.storage.contract import HDF5Contract

from .generation_reader import read_generation_payload

MYR_TO_YR = 1.0e6


@dataclass(frozen=True)
class MergerCatalog:
    primary_mass_msun: np.ndarray
    secondary_mass_msun: np.ndarray
    primary_spin: np.ndarray
    secondary_spin: np.ndarray
    ecc10: np.ndarray
    delay_yr: np.ndarray
    generation: np.ndarray
    sample_index: np.ndarray

    @property
    def size(self) -> int:
        return int(self.primary_mass_msun.size)

    @classmethod
    def empty(cls) -> "MergerCatalog":
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int64)
        return cls(
            primary_mass_msun=empty_float,
            secondary_mass_msun=empty_float,
            primary_spin=empty_float,
            secondary_spin=empty_float,
            ecc10=empty_float,
            delay_yr=empty_float,
            generation=empty_int,
            sample_index=empty_int,
        )

    @classmethod
    def concat(cls, catalogs: list["MergerCatalog"]) -> "MergerCatalog":
        catalogs = [catalog for catalog in catalogs if catalog.size > 0]
        if not catalogs:
            return cls.empty()

        return cls(
            primary_mass_msun=np.concatenate(
                [catalog.primary_mass_msun for catalog in catalogs]
            ),
            secondary_mass_msun=np.concatenate(
                [catalog.secondary_mass_msun for catalog in catalogs]
            ),
            primary_spin=np.concatenate([catalog.primary_spin for catalog in catalogs]),
            secondary_spin=np.concatenate(
                [catalog.secondary_spin for catalog in catalogs]
            ),
            ecc10=np.concatenate([catalog.ecc10 for catalog in catalogs]),
            delay_yr=np.concatenate([catalog.delay_yr for catalog in catalogs]),
            generation=np.concatenate([catalog.generation for catalog in catalogs]),
            sample_index=np.concatenate([catalog.sample_index for catalog in catalogs]),
        )

    def select(self, mask: np.ndarray) -> "MergerCatalog":
        mask = np.asarray(mask, dtype=bool)
        return MergerCatalog(
            primary_mass_msun=self.primary_mass_msun[mask],
            secondary_mass_msun=self.secondary_mass_msun[mask],
            primary_spin=self.primary_spin[mask],
            secondary_spin=self.secondary_spin[mask],
            ecc10=self.ecc10[mask],
            delay_yr=self.delay_yr[mask],
            generation=self.generation[mask],
            sample_index=self.sample_index[mask],
        )


def read_merger_catalog(
    path: str | Path,
    contract: HDF5Contract = HDF5Contract(),
) -> MergerCatalog:
    file_path = Path(path)
    with h5py.File(file_path, "r") as h5f:
        payload = read_generation_payload(h5f, contract)
        generation = int(np.asarray(h5f.attrs.get(contract.attr_generation, 1)).item())
        sample_index = int(
            np.asarray(h5f.attrs.get(contract.attr_sample_index, -1)).item()
        )

    return _catalog_from_payload(
        payload=payload,
        generation=generation,
        sample_index=sample_index,
    )


def read_merger_catalogs(
    paths: Sequence[str | Path],
    contract: HDF5Contract = HDF5Contract(),
) -> MergerCatalog:
    return MergerCatalog.concat(
        [read_merger_catalog(path, contract=contract) for path in paths]
    )


def _catalog_from_payload(
    *,
    payload: dict[str, object],
    generation: int,
    sample_index: int,
) -> MergerCatalog:
    merger_flag = _array(payload, "merger_flag", dtype=int)
    if merger_flag.size == 0:
        return MergerCatalog.empty()

    mask = merger_flag == 1
    if not np.any(mask):
        return MergerCatalog.empty()

    row_count = int(np.count_nonzero(mask))
    return MergerCatalog(
        primary_mass_msun=_array(payload, "m1")[mask],
        secondary_mass_msun=_array(payload, "m2")[mask],
        primary_spin=_array(payload, "chi1")[mask],
        secondary_spin=_array(payload, "chi2")[mask],
        ecc10=_array(payload, "ecc10")[mask],
        delay_yr=_array(payload, "tdel_myr")[mask] * MYR_TO_YR,
        generation=np.full(row_count, generation, dtype=np.int64),
        sample_index=np.full(row_count, sample_index, dtype=np.int64),
    )


def _array(
    payload: dict[str, object],
    name: str,
    dtype: type = float,
) -> np.ndarray:
    value = payload.get(name)
    if value is None:
        return np.empty(0, dtype=dtype)
    return np.asarray(value, dtype=dtype)
