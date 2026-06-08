# bsi_trap/storage/hdf5_utils.py

from __future__ import annotations

from typing import Any, Mapping
import json

import h5py
import numpy as np

from .contract import HDF5Contract


def write_scalar_or_array(group: h5py.Group, key: str, value: Any) -> None:
    if value is None:
        return

    if isinstance(value, (str, bytes)) or np.isscalar(value):
        group.attrs[key] = value
        return

    arr = np.asarray(value)

    if arr.dtype == object:
        raise TypeError(
            f"Cannot safely write object array for key '{key}' to HDF5."
        )

    group.create_dataset(key, data=arr)


def write_mapping_group(
    parent: h5py.Group,
    name: str,
    payload: Mapping[str, Any],
) -> None:
    group = parent.create_group(name)

    for key, value in payload.items():
        if isinstance(value, Mapping):
            write_mapping_group(group, key, value)
        else:
            write_scalar_or_array(group, key, value)


def write_model_config_attrs(
    h5obj: h5py.Group | h5py.File,
    model_config: Any,
    contract: HDF5Contract,
) -> None:
    if model_config is None:
        return

    if hasattr(model_config, "to_dict"):
        model_dict = dict(model_config.to_dict())
    elif isinstance(model_config, Mapping):
        model_dict = dict(model_config)
    else:
        model_dict = dict(model_config)

    h5obj.attrs[contract.attr_model_config_json] = json.dumps(
        model_dict,
        sort_keys=True,
    )

    for key, value in model_dict.items():
        if isinstance(value, (str, int, float, bool, np.bool_)):
            h5obj.attrs[f"model_{key}"] = value