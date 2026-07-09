"""Read sample-level metadata from combined HDF5 files."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import json

import h5py
import numpy as np

from bsi_trap.storage.contract import HDF5Contract

from .array_utils import read_attributes


def read_sample_metadata(
    sample_group: h5py.Group,
    contract: HDF5Contract,
) -> dict[str, Any]:
    metadata = read_attributes(sample_group)

    metadata.setdefault(
        contract.attr_sample_index,
        int(metadata.get(contract.attr_sample_index, -1)),
    )

    _merge_model_config_json(metadata, contract)
    _add_common_physics_metadata(metadata)

    return metadata


def _merge_model_config_json(
    metadata: dict[str, Any],
    contract: HDF5Contract,
) -> None:
    raw_config = metadata.get(contract.attr_model_config_json)

    if not isinstance(raw_config, str) or not raw_config:
        return

    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError:
        return

    if isinstance(config, Mapping):
        for key, value in config.items():
            metadata.setdefault(key, value)


def _add_common_physics_metadata(metadata: dict[str, Any]) -> None:
    if "rtrap_rg" in metadata:
        return

    if "rtrap" not in metadata or "rg" not in metadata:
        return

    try:
        metadata["rtrap_rg"] = float(metadata["rtrap"]) / float(metadata["rg"])
    except Exception:
        metadata["rtrap_rg"] = np.nan