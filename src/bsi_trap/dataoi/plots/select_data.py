from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def scenario_keys(
    data: Mapping[str, object], scenarios: Sequence[str] | None = None
) -> list[str]:
    """Return selected scenario keys in stable order."""
    if scenarios is not None:
        return [key for key in scenarios if key in data]
    return list(data.keys())


def to_1d_float(values: Any) -> np.ndarray:
    """Convert arbitrary array-like values to a flat float array."""
    if values is None:
        return np.asarray([], dtype=float)
    try:
        return np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)


def finite(values: Any) -> np.ndarray:
    """Return finite entries from arbitrary array-like values."""
    arr = to_1d_float(values)
    return arr[np.isfinite(arr)]


def positive_finite_log10(values: Any) -> np.ndarray:
    """Return log10 of positive finite values."""
    arr = finite(values)
    arr = arr[arr > 0]
    return np.log10(arr) if arr.size else arr


def paired_positive_finite_log10(
    x_values: Any, y_values: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired log10 arrays, dropping invalid rows."""
    x_arr = to_1d_float(x_values)
    y_arr = to_1d_float(y_values)
    n = min(x_arr.size, y_arr.size)
    if n == 0:
        return np.asarray([]), np.asarray([])
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0) & (y_arr > 0)
    return np.log10(x_arr[mask]), np.log10(y_arr[mask])
