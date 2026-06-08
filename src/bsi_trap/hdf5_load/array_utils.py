"""Small HDF5 / NumPy conversion helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import h5py
import numpy as np


def decode_value(value: Any) -> Any:
    """Decode HDF5 byte strings into normal Python strings."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def dataset_to_array(dataset: h5py.Dataset) -> np.ndarray:
    """Read one HDF5 dataset as a NumPy array."""
    values = dataset[()]
    values = np.asarray(values)

    if values.dtype.kind == "S":
        return values.astype(str)

    return values


def read_attributes(group: h5py.Group) -> dict[str, Any]:
    """Read HDF5 attributes into a plain dict."""
    return {key: decode_value(group.attrs[key]) for key in group.attrs.keys()}


def read_group_recursively(group: h5py.Group) -> dict[str, Any]:
    """Read a whole HDF5 group into nested dictionaries and NumPy arrays."""
    payload: dict[str, Any] = read_attributes(group)

    for key in group.keys():
        node = group[key]
        if isinstance(node, h5py.Group):
            payload[key] = read_group_recursively(node)
        elif isinstance(node, h5py.Dataset):
            payload[key] = dataset_to_array(node)

    return payload


def first_array_length(payload: Mapping[str, Any]) -> int:
    """Return the first non-scalar array length found in a mapping."""
    for value in payload.values():
        if isinstance(value, Mapping):
            nested_length = first_array_length(value)
            if nested_length > 0:
                return nested_length
            continue

        array = np.asarray(value)
        if array.ndim > 0 and array.size > 0:
            return int(array.shape[0])

    return 0


def finalize_arrays(value: Any) -> Any:
    """Convert accumulated lists in nested dicts into NumPy arrays."""
    if isinstance(value, dict):
        return {key: finalize_arrays(child) for key, child in value.items()}

    array = np.asarray(value)
    if array.dtype.kind in {"S", "U"}:
        return array.astype(str)

    return array
