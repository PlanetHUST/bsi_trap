"""Read one generation group using the storage-side HDF5 schema."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from bsi_trap.storage.contract import HDF5Contract
from bsi_trap.storage.schemas import generation_level_schema

from .array_utils import dataset_to_array
from .derived_fields import add_derived_fields


def read_generation_payload(
    generation_group: h5py.Group,
    contract: HDF5Contract,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    for group_spec in generation_level_schema(contract):
        if group_spec.group_name not in generation_group:
            continue

        group = generation_group[group_spec.group_name]

        if not isinstance(group, h5py.Group):
            continue

        if group_spec.recursive:
            payload[group_spec.group_name] = _read_recursive_group(group)
            continue

        payload.update(_read_declared_fields(group, group_spec))

    add_derived_fields(payload)
    return payload


def _read_declared_fields(
    group: h5py.Group,
    group_spec: Any,
) -> dict[str, Any]:
    output: dict[str, Any] = {}

    for field in group_spec.fields:
        source_name = field.source_name
        output_name = field.output_name

        if source_name in group:
            node = group[source_name]

            if isinstance(node, h5py.Dataset):
                output[output_name] = dataset_to_array(node)

        elif source_name in group.attrs:
            output[output_name] = np.asarray(group.attrs[source_name])

    return output


def _read_recursive_group(group: h5py.Group) -> dict[str, Any]:
    output: dict[str, Any] = {}

    for key, value in group.attrs.items():
        output[key] = value.decode("utf-8") if isinstance(value, bytes) else value

    for key in group.keys():
        node = group[key]

        if isinstance(node, h5py.Dataset):
            output[key] = dataset_to_array(node)
        elif isinstance(node, h5py.Group):
            output[key] = _read_recursive_group(node)

    return output