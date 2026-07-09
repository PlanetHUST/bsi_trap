"""Read sample catalog from the combined HDF5 file."""

from __future__ import annotations

from typing import Any

import h5py

from bsi_trap.storage.contract import HDF5Contract

from .array_utils import dataset_to_array


def read_catalog(
    h5_file: h5py.File,
    contract: HDF5Contract,
) -> dict[str, Any]:
    if contract.group_catalog not in h5_file:
        return {}

    catalog_group = h5_file[contract.group_catalog]

    if not isinstance(catalog_group, h5py.Group):
        return {}

    output: dict[str, Any] = {}

    for key, value in catalog_group.attrs.items():
        output[key] = value.decode("utf-8") if isinstance(value, bytes) else value

    for key in catalog_group.keys():
        node = catalog_group[key]

        if isinstance(node, h5py.Dataset):
            output[key] = dataset_to_array(node)

    return output