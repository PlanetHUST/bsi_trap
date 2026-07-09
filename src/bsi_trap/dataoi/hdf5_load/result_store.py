"""Generic nested result writing.

This module knows nothing about HDF5 names.  It only appends rows into paths
requested by the layout routes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .array_utils import decode_value, first_array_length


def empty_result(simulation_type: str, route_paths: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "metadata": {
            "simulation_type": simulation_type,
            "sample_table": {},
        },
    }

    for route_path in route_paths:
        ensure_path(result, route_path)

    return result


def ensure_path(root: dict[str, Any], path: str) -> dict[str, Any]:
    current = root
    for part in path.strip("/").split("/"):
        if not part:
            continue
        current = current.setdefault(part, {})
    return current


def format_output_path(template: str, generation_number: int) -> str:
    return template.format(generation=generation_number)


def append_event_rows(
    target: dict[str, list],
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    generation_number: int,
    row_mask: np.ndarray | None = None,
) -> None:
    event_count = first_array_length(payload)
    if event_count <= 0:
        return

    if row_mask is None:
        selected_indices = np.arange(event_count)
    else:
        selected_indices = np.where(row_mask[:event_count])[0]

    if selected_indices.size == 0:
        return

    for key, value in payload.items():
        if isinstance(value, Mapping):
            continue

        values = np.asarray(value)
        if values.ndim == 0:
            target.setdefault(key, []).extend([decode_value(values.item())] * len(selected_indices))
            continue

        if values.size == 0:
            continue

        if values.shape[0] == event_count:
            target.setdefault(key, []).extend(values[selected_indices].tolist())

    append_metadata_columns(
        target=target,
        metadata=metadata,
        generation_number=generation_number,
        row_count=int(selected_indices.size),
    )


def append_scalar_row(
    target: dict[str, list],
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    generation_number: int,
) -> None:
    for key, value in payload.items():
        if isinstance(value, Mapping) or key == "generation":
            continue

        array = np.asarray(value)
        if array.ndim == 0:
            target.setdefault(key, []).append(decode_value(array.item()))
        elif array.size == 1:
            target.setdefault(key, []).append(decode_value(array.reshape(-1)[0]))

    append_metadata_columns(
        target=target,
        metadata=metadata,
        generation_number=generation_number,
        row_count=1,
    )


def append_metadata_columns(
    target: dict[str, list],
    metadata: Mapping[str, Any],
    generation_number: int,
    row_count: int,
) -> None:
    target.setdefault("generation", []).extend([generation_number] * row_count)

    for key, value in metadata.items():
        value = decode_value(value)

        if isinstance(value, str):
            normalized_value: Any = value
        else:
            try:
                normalized_value = float(value)
            except Exception:
                normalized_value = np.nan

        target.setdefault(key, []).extend([normalized_value] * row_count)
