from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bsi_trap.hdf5_load.merger_catalog import MergerCatalog


def plot_merger_parameter_histograms(
    catalog: MergerCatalog,
    output_dir: Path,
) -> Path | None:
    """Plot merged-BBH parameter histograms split by hierarchical outcome."""
    if catalog.size == 0:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    groups = _split_catalog(catalog)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))

    _panel(
        axes[0, 0],
        groups,
        field="primary_mass_msun",
        bins=_log_bins(catalog.primary_mass_msun, fallback=(3.0, 1.0e4)),
        xlabel=r"Primary Mass [$M_\odot$]",
        ylabel="pdf",
        density=True,
        xscale="log",
    )
    _panel(
        axes[0, 1],
        groups,
        field="secondary_mass_msun",
        bins=_log_bins(catalog.secondary_mass_msun, fallback=(3.0, 1.0e3)),
        xlabel=r"Secondary Mass [$M_\odot$]",
        ylabel="pdf",
        density=True,
        xscale="log",
    )
    _panel(
        axes[1, 0],
        groups,
        field="primary_spin",
        bins=np.linspace(0.0, 1.0, 22),
        xlabel="Primary Spin",
        ylabel="pdf",
        density=True,
        xscale="linear",
    )
    _panel(
        axes[1, 1],
        groups,
        field="secondary_spin",
        bins=np.linspace(0.0, 1.0, 22),
        xlabel="Secondary Spin",
        ylabel="pdf",
        density=True,
        xscale="linear",
    )
    _panel(
        axes[2, 0],
        groups,
        field="ecc10",
        bins=_log_bins(catalog.ecc10, fallback=(1.0e-6, 1.0)),
        xlabel=r"Eccentricity at $f_\mathrm{GW}=10$ Hz",
        ylabel="events",
        density=False,
        xscale="log",
    )
    _panel(
        axes[2, 1],
        groups,
        field="delay_yr",
        bins=_log_bins(catalog.delay_yr, fallback=(1.0e-3, 1.0e10)),
        xlabel="Delay Timescale[yr]",
        ylabel="events",
        density=False,
        xscale="log",
    )

    for axis in axes.flat:
        axis.set_yscale("log")
        axis.grid(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=True, fontsize=10)

    fig.tight_layout()
    output_path = output_dir / "merger_parameter_histograms.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _split_catalog(catalog: MergerCatalog) -> dict[str, tuple[MergerCatalog, str, bool]]:
    hierarchical_samples = set(
        catalog.sample_index[catalog.generation > 1].astype(int).tolist()
    )
    sample_is_hierarchical = np.isin(
        catalog.sample_index,
        np.asarray(sorted(hierarchical_samples), dtype=np.int64),
    )
    first_generation = catalog.generation == 1

    return {
        "GH": (catalog.select(sample_is_hierarchical), "#7ec8ed", False),
        "no-GH": (catalog.select(~sample_is_hierarchical), "#d5b083", False),
        "GH 1g": (
            catalog.select(sample_is_hierarchical & first_generation),
            "#aee7f7",
            True,
        ),
        "no-GH 1g": (
            catalog.select((~sample_is_hierarchical) & first_generation),
            "#ead8bf",
            True,
        ),
    }


def _panel(
    axis,
    groups: dict[str, tuple[MergerCatalog, str, bool]],
    *,
    field: str,
    bins: np.ndarray,
    xlabel: str,
    ylabel: str,
    density: bool,
    xscale: str,
) -> None:
    for label, (catalog, color, filled) in groups.items():
        values = _valid_values(getattr(catalog, field), positive=xscale == "log")
        if values.size == 0:
            continue

        if filled:
            axis.hist(
                values,
                bins=bins,
                density=density,
                histtype="stepfilled",
                alpha=0.55,
                color=color,
                label=label,
            )
        else:
            axis.hist(
                values,
                bins=bins,
                density=density,
                histtype="step",
                linewidth=2.0,
                color=color,
                label=label,
            )

    axis.set_xscale(xscale)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)


def _valid_values(values: np.ndarray, *, positive: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if positive:
        values = values[values > 0.0]
    return values


def _log_bins(values: np.ndarray, *, fallback: tuple[float, float]) -> np.ndarray:
    values = _valid_values(values, positive=True)
    if values.size == 0:
        lo, hi = fallback
    else:
        lo = min(float(values.min()), fallback[0])
        hi = max(float(values.max()), fallback[1])
    lo = max(lo, np.finfo(float).tiny)
    if hi <= lo:
        hi = lo * 10.0
    return np.logspace(np.log10(lo), np.log10(hi), 28)
