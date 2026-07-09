"""Route generation payloads into analysis output sections."""

from __future__ import annotations

from typing import Any

import numpy as np

from .result_store import append_event_rows, ensure_path


def route_generation_payload(
    result: dict[str, Any],
    payload: dict[str, Any],
    sample_metadata: dict[str, Any],
    generation_number: int,
) -> None:
    if generation_number == 1:
        main_path = "first_gen"
        merger_path = "merge_data/first_gen"
    else:
        main_path = f"n_gen/gen_{generation_number:03d}"
        merger_path = f"merge_data/n_gen/gen_{generation_number:03d}"

    append_event_rows(
        target=ensure_path(result, main_path),
        payload=payload,
        metadata=sample_metadata,
        generation_number=generation_number,
    )

    if "merger_flag" not in payload:
        return

    merger_mask = np.asarray(payload["merger_flag"]) == 1

    append_event_rows(
        target=ensure_path(result, merger_path),
        payload=payload,
        metadata=sample_metadata,
        generation_number=generation_number,
        row_mask=merger_mask,
    )