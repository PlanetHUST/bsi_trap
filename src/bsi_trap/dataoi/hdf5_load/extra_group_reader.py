"""Read recursive extra generation groups such as trap_population."""

from __future__ import annotations

from typing import Any

import h5py

from bsi_trap.storage.contract import HDF5Contract

from .array_utils import read_group_recursively
from .result_store import append_event_rows, append_scalar_row, ensure_path


def route_extra_groups(
    result: dict[str, Any],
    generation_group: h5py.Group,
    sample_metadata: dict[str, Any],
    generation_number: int,
    contract: HDF5Contract,
) -> None:
    if contract.group_trap_population not in generation_group:
        return

    trap_group = generation_group[contract.group_trap_population]

    if not isinstance(trap_group, h5py.Group):
        return

    for section_name in trap_group.keys():
        section = trap_group[section_name]

        if not isinstance(section, h5py.Group):
            continue

        payload = read_group_recursively(section)
        target = ensure_path(result, f"{contract.group_trap_population}/{section_name}")

        if _looks_like_event_table(payload):
            append_event_rows(
                target=target,
                payload=payload,
                metadata=sample_metadata,
                generation_number=generation_number,
            )
        else:
            append_scalar_row(
                target=target,
                payload=payload,
                metadata=sample_metadata,
                generation_number=generation_number,
            )


def _looks_like_event_table(payload: dict[str, Any]) -> bool:
    for value in payload.values():
        if hasattr(value, "ndim") and value.ndim > 0 and value.size > 1:
            return True

    return False