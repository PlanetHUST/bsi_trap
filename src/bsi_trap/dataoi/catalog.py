"""Catalog loading for the compact demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_CATALOG = Path(__file__).resolve().parents[3] / "input" / "input_data.txt"


def load_remnant_masses(
    catalog_path: Path = DEFAULT_CATALOG,
    *,
    max_rows: int | None = 50_000,
) -> np.ndarray:
    """Load remnant masses from the local stellar-remnant catalog."""
    if not catalog_path.exists():
        raise FileNotFoundError(f"Remnant catalog not found: {catalog_path}")

    masses = np.loadtxt(
        catalog_path,
        usecols=(2,),
        skiprows=3,
        max_rows=max_rows,
        dtype=float,
        ndmin=1,
    ).reshape(-1)
    keep = np.isfinite(masses) & (masses > 0.0) & (masses <= 80.0)
    masses = masses[keep]
    if masses.size == 0:
        raise ValueError(f"No usable remnant masses found in {catalog_path}")
    return masses
