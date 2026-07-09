# bsi_trap/storage/catalog.py

from __future__ import annotations

import h5py
import numpy as np

from .contract import HDF5Contract


def _int_attr(group: h5py.Group, name: str, default: int) -> int:
    value = group.attrs.get(name, default)
    return int(np.asarray(value).item())


def _float_attr(group: h5py.Group, name: str, default: float) -> float:
    value = group.attrs.get(name, default)
    return float(np.asarray(value).item())


def write_sample_catalog(
    *,
    f_out: h5py.File,
    samples_group: h5py.Group,
    contract: HDF5Contract,
) -> None:
    catalog = f_out.create_group(contract.group_catalog)

    sample_names = sorted(samples_group.keys())

    catalog.attrs["num_samples"] = len(sample_names)
    catalog.attrs["sample_names"] = np.asarray(sample_names, dtype="S")

    sample_indices = []
    process_ids = []
    smbh_masses = []

    for sample_name in sample_names:
        node = samples_group[sample_name]
        if not isinstance(node, h5py.Group):
            continue
        sample_group = node

        sample_indices.append(_int_attr(sample_group, contract.attr_sample_index, -1))
        process_ids.append(_int_attr(sample_group, contract.attr_process_id, -1))
        smbh_masses.append(
            _float_attr(sample_group, contract.attr_smbh_mass_msun, np.nan)
        )

    catalog.create_dataset("sample_index", data=np.asarray(sample_indices, dtype=int))
    catalog.create_dataset("process_id", data=np.asarray(process_ids, dtype=int))
    catalog.create_dataset("smbh_mass_msun", data=np.asarray(smbh_masses, dtype=float))
