"""Plotting helpers for the SMBH-aware AGN merger pipeline.

This module sits at the final stage of the workflow:

    combined HDF5 -> hdf5_load.py -> plot_data_utils.py -> fig_plot.py

The plotting functions here do not read files directly. They expect already
prepared dictionaries whose structure is defined by ``hdf5_load.py`` and
reshaped by ``plot_data_utils.py``. Most plotting issues are therefore easier
to debug by checking the upstream payload first rather than the Matplotlib code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import constants as const
from migration_torque import (
    MigrationParameters,
    compute_torque_profile,
    find_stable_traps,
)
from pagn_disk_adapter import build_pagn_disk_profile

from plot_data_utils import (
    build_timescale_data,
    ensure_dir,
    flatten_generation_data,
    positive_finite,
    prepare_plot_values,
    to_array,
)

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "mathtext.rm": "STIXGeneral",
        "mathtext.it": "STIXGeneral:italic",
        "mathtext.bf": "STIXGeneral:bold",
        "font.size": 28,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "legend.fontsize": 22,
        "figure.titlesize": 24,
    }
)
if sns is not None:
    sns.set_theme(style="whitegrid", rc=plt.rcParams)

# Default scenario metadata.  ``fig_main.py`` may pass explicit labels/colors
# for arbitrary model grids; these defaults preserve the historical comparison.
SCENARIO_DISPLAY = {
    "damp": "Damped Simulation",
    "lin": "Linear Sampling",
    "log": "Log Sampling",
    "with_gas_damp": "Damped Simulation",
    "with_gas_damp_no_copy": "Damped No Copy",
    "with_gas_lin": "Linear Sampling",
    "with_gas_log": "Log Sampling",
    "without_gas_damp": "No Gas Damped",
    "without_gas_lin": "No Gas Linear",
    "without_gas_log": "No Gas Log",
    "BSI": "BSI",
}

# Keep a single color choice per scenario so the same simulation mode is easy
# to track across all figures.
SCENARIO_COLORS = {
    "damp": {"initial": "#1f77b4", "merger": "#1f77b4"},
    "lin": {"initial": "#BE9A62", "merger": "#BE9A62"},
    "log": {"initial": "#6B4E25", "merger": "#6B4E25"},
    "with_gas_damp": {"initial": "#1f77b4", "merger": "#1f77b4"},
    "with_gas_damp_no_copy": {"initial": "#FF7F0E", "merger": "#FF7F0E"},
    "with_gas_lin": {"initial": "#BE9A62", "merger": "#BE9A62"},
    "with_gas_log": {"initial": "#6B4E25", "merger": "#6B4E25"},
    "without_gas_damp": {"initial": "#2CA02C", "merger": "#2CA02C"},
    "without_gas_lin": {"initial": "#9467BD", "merger": "#9467BD"},
    "without_gas_log": {"initial": "#D62728", "merger": "#D62728"},
    "BSI": {"initial": "#17BECF", "merger": "#17BECF"},
}

FALLBACK_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

GENERATION_STYLE = {
    "first_gen": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
    "n_gen": {"color": "#6382AA", "linestyle": "--", "marker": "s"},
}

RADIAL_AXIS_LABELSIZE = 34
RADIAL_TITLE_FONTSIZE = 30
MERGER_AXIS_LABELSIZE = 34
TIME_HIST_TICK_LABELSIZE = 30
TIME_HIST_LEGEND_FONTSIZE = 28


_LABELS = {
    "m1": r"$M_{1}\,(\mathrm{M}_\odot)$",
    "m2": r"$M_{2}\,(\mathrm{M}_\odot)$",
    "m1+m2": r"$M_{\mathrm{tot}}\,(\mathrm{M}_\odot)$",
    "m2/m1": r"$q$",
    "chi1": r"$\chi_{1}$",
    "chi2": r"$\chi_{2}$",
    "chieff": r"$\chi_{\mathrm{eff}}$",
    "chip": r"$\chi_{\mathrm{p}}$",
    "ecc10": r"$e_{10\,\mathrm{Hz}}$",
    "Aini": r"$a_{\mathrm{out}}$",
    "tmerg": r"$t_{\mathrm{merg}}\,(\mathrm{Myr})$",
    "trel": r"$t_{\mathrm{rel}}\,(\mathrm{Myr})$",
    "tdamp": r"$t_{\mathrm{damp}}\,(\mathrm{Myr})$",
    "tmig": r"$t_{\mathrm{mig}}\,(\mathrm{Myr})$",
    "tdel": r"$t_{\mathrm{del}}\,(\mathrm{Myr})$",
    "t_bsi": r"$t_{\mathrm{BSI}}\,(\mathrm{Myr})$",
    "M_SMBH": r"$M_{\mathrm{\bullet}}\,(\mathrm{M}_\odot)$",
}


def _positive_finite_log10(values) -> np.ndarray:
    """Return log10 of positive finite values, dropping invalid entries."""
    arr = positive_finite(values)
    return np.log10(arr) if arr.size else arr


def _paired_positive_finite_log10(x_values, y_values) -> Tuple[np.ndarray, np.ndarray]:
    """Return paired log10 arrays, dropping rows where either value is invalid."""
    x_arr = np.asarray(x_values, dtype=float).ravel()
    y_arr = np.asarray(y_values, dtype=float).ravel()
    n = min(x_arr.size, y_arr.size)
    if n == 0:
        return np.asarray([]), np.asarray([])
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0) & (y_arr > 0)
    return np.log10(x_arr[mask]), np.log10(y_arr[mask])


def _lower_edge_without_isolated_tail(values: np.ndarray) -> Tuple[float, int]:
    """Drop an isolated low-end plotting tail when it would create empty axis space."""
    sorted_values = np.sort(np.asarray(values, dtype=float))
    sorted_values = sorted_values[np.isfinite(sorted_values)]
    if sorted_values.size < 4:
        return float(sorted_values[0]), 0

    low_window = max(3, int(np.ceil(sorted_values.size * 0.01)))
    low_window = min(low_window, sorted_values.size - 1)
    lower_values = sorted_values[: low_window + 1]
    gaps = np.diff(lower_values)
    if gaps.size == 0:
        return float(sorted_values[0]), 0

    gap_index = int(np.argmax(gaps))
    if gaps[gap_index] > 0.05:
        lower_edge = float(lower_values[gap_index + 1])
        clipped_count = int(np.searchsorted(sorted_values, lower_edge, side="left"))
        return lower_edge, clipped_count
    return float(sorted_values[0]), 0


def _positive_finite_log10_by_flag(data: Mapping, value_key: str, flag_key: str):
    values = to_array(data.get(value_key, []))
    flags = to_array(data.get(flag_key, []))
    n = min(values.size, flags.size)
    if n == 0:
        return np.asarray([])
    mask = flags[:n].astype(bool)
    return _positive_finite_log10(values[:n][mask])


def _weighted_gaussian_density(
    values: np.ndarray,
    x_grid: np.ndarray,
    total_weight: float,
    bandwidth: float | None = None,
) -> np.ndarray:
    """Smooth a 1D event sample into an absolute weighted density."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    x_grid = np.asarray(x_grid, dtype=float)
    if values.size == 0 or x_grid.size == 0:
        return np.zeros_like(x_grid, dtype=float)

    if values.size == 1:
        if bandwidth is None:
            bandwidth = 0.12
        bandwidth = max(float(bandwidth), 1e-3)
    elif bandwidth is None:
        std = float(np.std(values, ddof=1))
        iqr = float(np.subtract(*np.percentile(values, [75, 25])))
        robust_sigma = min(std, iqr / 1.349) if iqr > 0 else std
        robust_sigma = robust_sigma if robust_sigma > 0 else std
        bandwidth = 0.9 * robust_sigma * values.size ** (-1.0 / 5.0)
        bandwidth = max(float(bandwidth), 0.08)
    else:
        bandwidth = max(float(bandwidth), 1e-3)

    z = (x_grid[:, None] - values[None, :]) / bandwidth
    kernel = np.exp(-0.5 * z**2) / (np.sqrt(2.0 * np.pi) * bandwidth)
    return total_weight * kernel.mean(axis=1)


def _plot_kde_or_density(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    color: str,
    linewidth: float,
    label: str,
    weights: np.ndarray | None = None,
) -> None:
    """Plot a smooth density curve, with a Matplotlib fallback if seaborn is absent."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return

    if sns is not None:
        sns.kdeplot(
            x=values,
            weights=weights,
            color=color,
            linewidth=linewidth,
            label=label,
            ax=ax,
        )
        return

    counts, edges = np.histogram(values, bins=40, weights=weights, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    positive = counts > 0
    if np.any(positive):
        ax.plot(
            centers[positive],
            counts[positive],
            color=color,
            linewidth=linewidth,
            label=label,
        )


def _scenario_keys(
    payload: Mapping[str, Any], scenarios: Sequence[str] | None = None
) -> list[str]:
    """Return ordered model keys from a plot payload, excluding metadata keys."""
    if scenarios is not None:
        return [key for key in scenarios if key in payload]
    return [key for key in payload.keys() if key != "outfile"]


def _scenario_title(key: str, model_labels: Mapping[str, str] | None = None) -> str:
    if model_labels and key in model_labels:
        return str(model_labels[key])
    if key in SCENARIO_DISPLAY:
        return SCENARIO_DISPLAY[key]
    return key.replace("_", " ").title()


def _fallback_color(key: str) -> str:
    index = sum(ord(char) for char in str(key)) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[index]


def _scenario_color(
    key: str,
    role: str = "initial",
    model_colors: Mapping[str, Any] | None = None,
) -> str:
    if model_colors and key in model_colors:
        color_spec = model_colors[key]
        if isinstance(color_spec, Mapping):
            return str(
                color_spec.get(role, color_spec.get("initial", _fallback_color(key)))
            )
        return str(color_spec)
    if key in SCENARIO_COLORS:
        return SCENARIO_COLORS[key].get(role, SCENARIO_COLORS[key]["initial"])
    return _fallback_color(key)


def _reshape_axes(axes: Any, nrows: int, ncols: int) -> np.ndarray:
    return np.asarray(axes, dtype=object).reshape(nrows, ncols)


def _format_count_label(count: int | float) -> str:
    """Format legend counts consistently as readable integers."""
    return f"{int(round(float(count))):,}"


def _save(fig: plt.Figure, outdir: str | Path, stem: str, dpi: int = 300) -> None:
    """Save each figure in both PNG and PDF form for quick view and publication."""
    outdir = ensure_dir(outdir)
    fig.savefig(outdir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")


def _stem_with_suffix(stem: str, filename_suffix: str | None = None) -> str:
    """Append a caller-provided suffix to a figure stem."""
    if not filename_suffix:
        return stem
    suffix = str(filename_suffix)
    if not suffix.startswith("_"):
        suffix = f"_{suffix}"
    return f"{stem}{suffix}"


def _as_1d_float(values: Any) -> np.ndarray:
    """Normalize numeric-like inputs to a flat float array.

    Plot payloads may come in as scalars, lists, NumPy arrays, or HDF5-derived
    arrays. This helper makes downstream plotting code assume a single shape.
    """
    if values is None:
        return np.array([], dtype=float)

    if isinstance(values, (str, bytes)):
        raise TypeError("string metadata is not numeric")

    try:
        arr = np.asarray(values, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("non-numeric field")

    if arr.ndim == 0:
        return np.array([float(arr)], dtype=float)
    return arr.reshape(-1).astype(float, copy=False)


def _finite(values: Any) -> np.ndarray:
    arr = _as_1d_float(values)
    return arr[np.isfinite(arr)]


def _legend_dedup(ax: plt.Axes, **kwargs) -> None:
    """Remove duplicate legend entries created by repeated overlays."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    unique_handles = []
    unique_labels = []
    seen = set()
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    legend = ax.legend(unique_handles, unique_labels, **kwargs)
    if legend is not None:
        legend.set_zorder(10)


def _collect_trap_band_limits(*scenario_payloads):
    """Build one shared trap-location band from per-scenario summary stats.

    The shaded band used in the radial-distribution figures is intentionally
    based on sample-level trap statistics, not event-level arrays. That keeps
    the visual guide tied to the underlying AGN sample population rather than
    to whichever scenario happened to produce more events.
    """
    lows = []
    highs = []

    for payload in scenario_payloads:
        if not isinstance(payload, dict):
            continue

        stats = payload.get("trap_stats", {})
        if not isinstance(stats, dict):
            continue

        lo = stats.get("p16_log10")
        hi = stats.get("p84_log10")

        if lo is None or hi is None:
            continue
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue

        lows.append(float(lo))
        highs.append(float(hi))

    if not lows or not highs:
        return None

    return min(lows), max(highs)


class HDF5FigureGenerator:
    """Generate the distribution figures that compare 1g and ng merger products."""

    def __init__(
        self,
        output_dir: str = "figures",
        model_configs: Mapping[str, Mapping] | None = None,
        model_labels: Mapping[str, str] | None = None,
        model_colors: Mapping[str, Any] | None = None,
        # style: str = "default",
    ):
        self.output_dir = ensure_dir(output_dir)
        self.model_configs = dict(model_configs or {})
        self.model_labels = dict(model_labels or {})
        self.model_colors = dict(model_colors or {})
        self.colors = {
            "gas": {
                "first_gen": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
                "n_gen": {"color": "#6382AA", "linestyle": "--", "marker": "s"},
            },
            "nogas": {
                "first_gen": {"color": "#BE9A62", "linestyle": "-", "marker": "o"},
                "n_gen": {"color": "#6B4E25", "linestyle": "--", "marker": "s"},
            },
        }
        # if style in plt.style.available:
        #     plt.style.use(style)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "mathtext.rm": "STIXGeneral",
            "mathtext.it": "STIXGeneral:italic",
            "mathtext.bf": "STIXGeneral:bold",
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 18,
            "figure.titlesize": 18,
        }
    )

    def create_distribution_plots(
        self,
        plot_distribution_data: Mapping[str, Mapping],
        scenarios: Sequence[str] | None = None,
        model_labels: Mapping[str, str] | None = None,
        filename_suffix: str | None = None,
    ) -> None:
        """Create the main BBH distribution figures.

        Parameters
        ----------
        plot_distribution_data
            Expected shape:
            ``{scenario: {"first_gen": {...}, "n_gen": {...}}}``
            where each leaf contains event-aligned arrays. In practice this is
            the output of ``prepare_distribution_data``.
        """
        scenario_names = _scenario_keys(plot_distribution_data, scenarios)
        if not scenario_names:
            return
        plot_labels = self.model_labels if model_labels is None else model_labels

        ncols = len(scenario_names)
        fig1, axes1 = plt.subplots(
            4, ncols, figsize=(max(8, 7.3 * ncols), 18), sharey="row"
        )
        fig2, axes2 = plt.subplots(
            2, ncols, figsize=(max(8, 6.0 * ncols), 10), sharey="row"
        )
        axes1 = _reshape_axes(axes1, 4, ncols)
        axes2 = _reshape_axes(axes2, 2, ncols)

        parameters_fig1 = [
            ("m1", r"$M_{1}\,(\mathrm{M}_\odot)$", (10**0.5, 1e4)),
            ("chi1", r"$\chi_{1}$", (0, 1)),
            ("ecc10", r"$e_{10\,\mathrm{Hz}}$", (1e-9, 1)),
            (
                "tmerg",
                r"$t_{\mathrm{merg}}\;(\mathrm{Myr})$",
                (1e0, 1e4),
            ),
        ]

        parameters_fig2 = [
            ("chieff", r"$\chi_{\mathrm{eff}}$", (-1, 1)),
            ("chip", r"$\chi_{\mathrm{p}}$", (0, 1)),
        ]

        for column_idx, scenario in enumerate(scenario_names):
            scenario_data = plot_distribution_data.get(scenario)
            if not scenario_data:
                continue
            # Merge any per-generation dictionaries into the 2-layer layout
            # used by this plot: first generation vs all higher generations.
            flat_data = flatten_generation_data(scenario_data)
            self._plot_parameter_set(
                axes=axes1,
                parameters=parameters_fig1,
                data=flat_data,
                column_idx=column_idx,
                title=_scenario_title(scenario, plot_labels),
                use_log_y=True,
            )
            self._plot_parameter_set(
                axes=axes2,
                parameters=parameters_fig2,
                data=flat_data,
                column_idx=column_idx,
                title=_scenario_title(scenario, plot_labels),
                use_log_y=True,
            )

        fig1.tight_layout(rect=[0, 0, 1, 0.98])
        _save(
            fig1,
            self.output_dir,
            _stem_with_suffix("bbh_basic", filename_suffix),
            dpi=150,
        )
        plt.close(fig1)

        fig2.tight_layout(rect=[0, 0, 1, 0.98])
        _save(
            fig2,
            self.output_dir,
            _stem_with_suffix("bbh_spin", filename_suffix),
            dpi=150,
        )
        plt.close(fig2)

    def _plot_parameter_set(
        self,
        axes: np.ndarray,
        parameters: Sequence[tuple[str, str, tuple[float, float]]],
        data: Mapping[str, Mapping[str, np.ndarray]],
        column_idx: int,
        title: str,
        use_log_y: bool,
    ) -> None:
        """Populate one grid of histograms for a fixed set of parameters."""
        for row_idx, (param, label, xlim) in enumerate(parameters):
            ax = axes[row_idx, column_idx]
            # Masses, semimajor axes, and timescales span many decades, so we
            # plot them in log10-space before histogramming. Spin-like fields
            # are already bounded and stay in linear space.
            use_log_x = "chi" not in param
            density_plot = param not in {"ecc10", "tmerg"}

            first_vals = prepare_plot_values(
                data.get("first_gen", {}).get(param, []), use_log_x
            )
            n_vals = prepare_plot_values(
                data.get("n_gen", {}).get(param, []), use_log_x
            )

            all_vals = (
                np.concatenate([arr for arr in (first_vals, n_vals) if arr.size > 0])
                if (first_vals.size or n_vals.size)
                else np.array([])
            )

            # Use a shared binning per panel so 1g and ng curves are directly
            # comparable inside the same subplot.
            if all_vals.size > 1:
                bins = np.linspace(all_vals.min(), all_vals.max(), 20)
            elif all_vals.size == 1:
                bins = np.linspace(all_vals[0] - 0.5, all_vals[0] + 0.5, 10)
            else:
                bins = 30

            if first_vals.size:
                ax.hist(
                    first_vals,
                    bins=bins,
                    density=density_plot,
                    label=f"1g (N={_format_count_label(len(first_vals))})",
                    edgecolor=self.colors["gas"]["first_gen"]["color"],
                    linewidth=3,
                    histtype="step",
                    zorder=3,
                )

            if n_vals.size:
                ax.hist(
                    n_vals,
                    bins=bins,
                    density=density_plot,
                    alpha=0.3,
                    color=self.colors["gas"]["n_gen"]["color"],
                    label=f"ng (N={_format_count_label(len(n_vals))})",
                    edgecolor="black",
                    linewidth=1.2,
                    histtype="stepfilled",
                    linestyle="-",
                    zorder=2,
                )

            log_labels = {
                "m1": r"$\log_{10}\!\left(M_{1}/M_\odot\right)$",
                "m2": r"$\log_{10}\!\left(M_{2}/M_\odot\right)$",
                "m1+m2": r"$\log_{10}\!\left(M_{\mathrm{tot}}/M_\odot\right)$",
                "ecc10": r"$\log_{10}\!\left(e_{10\,\mathrm{Hz}}\right)$",
                "Aini": r"$\log_{10}\!\left(a_{\mathrm{out}}\right)$",
                "tmerg": r"$\log_{10}\!\left(t_{\mathrm{merg}}/\mathrm{Myr}\right)$",
                "trel": r"$\log_{10}\!\left(t_{\mathrm{rel}}/\mathrm{Myr}\right)$",
                "tdamp": r"$\log_{10}\!\left(t_{\mathrm{damp}}/\mathrm{Myr}\right)$",
                "tmig": r"$\log_{10}\!\left(t_{\mathrm{mig}}/\mathrm{Myr}\right)$",
                "M_SMBH": r"$\log_{10}\!\left(M_{\mathrm{\bullet}}/M_\odot\right)$",
            }

            shown_label = log_labels.get(param, label) if use_log_x else label
            # shown_label = label
            ax.set_xlabel(shown_label, fontsize=MERGER_AXIS_LABELSIZE)
            if column_idx == 0:
                ax.set_ylabel(
                    "PDF" if density_plot else "Events",
                    fontsize=MERGER_AXIS_LABELSIZE,
                )

            if xlim[0] is not None and xlim[1] is not None:
                if use_log_x:
                    ax.set_xlim(np.log10(xlim[0]), np.log10(xlim[1]))
                else:
                    ax.set_xlim(xlim)

            if use_log_y:
                ax.set_yscale("log")

            ax.grid(True, alpha=0.3, linestyle="--", which="both")

            if row_idx == 0:
                current_title = ax.get_title()
                title_text = title if not current_title else f"{title}\n{current_title}"
                ax.set_title(title_text, fontweight="bold", fontsize=28, pad=20)
                _legend_dedup(ax, loc="best")

    def _format_axis_label(self, key: str) -> str:
        return _LABELS.get(key, key)


def plot_sma_distribution(
    plot_sma_data: Dict,
    trap_band: Tuple[float, float] | None,
    scenarios: Sequence[str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    model_colors: Mapping[str, Any] | None = None,
) -> None:
    """Plot the initial outer-orbit radius distribution for first-generation BBHs.

    ``plot_sma_data`` is produced by ``prepare_sma_data``. The legend count is
    the raw number of loaded systems, while the plotted curve uses only positive
    finite radii that can safely be transformed to ``log10``.
    """
    output_dir = plot_sma_data["outfile"]
    fig, axes = plt.subplots(2, 1, figsize=(18, 14), sharex=True)
    ax_counts, ax_density = axes

    for scenario in _scenario_keys(plot_sma_data, scenarios):
        all_count = int(plot_sma_data[scenario]["all"].get("count", 0))

        all_a = _positive_finite_log10(plot_sma_data[scenario]["all"].get("Aini", []))

        if all_a.size == 0:
            continue

        bins = np.linspace(all_a.min(), all_a.max(), 40) if all_a.size > 1 else 15
        color = _scenario_color(scenario, "initial", model_colors)
        label = _scenario_title(scenario, model_labels)

        ax_counts.hist(
            all_a,
            bins=bins,
            alpha=0.8,
            color=color,
            edgecolor=color,
            linewidth=3.5,
            label=f"{label}: All systems (N={_format_count_label(all_count)})",
            histtype="step",
        )

        if len(all_a) > 1:
            # KDE is used only for the normalized panel. The top panel keeps the
            # raw counts so scenario yield differences remain visible.
            _plot_kde_or_density(
                ax_density,
                all_a,
                color=color,
                linewidth=2.5,
                label=label,
            )

    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    ax_counts.set_ylabel("Count", fontsize=RADIAL_AXIS_LABELSIZE)
    ax_counts.set_title(
        r"$N(\log(a_{\mathrm{out}}))=\mathrm{d}N_{\mathrm{tot}}/\mathrm{d}(\log(a_{\mathrm{out}}))$",
        fontweight="bold",
        fontsize=RADIAL_TITLE_FONTSIZE,
    )
    ax_counts.set_yscale("log")

    if trap_band is not None:
        trap_lo, trap_hi = trap_band
        print(f"Shading trap band between log10(a_out)={trap_lo:.2f} and {trap_hi:.2f}")
        for ax in axes:
            ax.axvspan(trap_lo, trap_hi, alpha=0.3, label=r"SMBH trap-radius range")

    _legend_dedup(ax_counts, framealpha=0.8, loc="best")

    # ax_density.set_xlabel(r"Initial $\log(a_{\mathrm{out}})\ [\log(r_g)]$")
    ax_density.set_xlabel(
        r"$\log_{10}(a_{\rm out}/r_g)$", fontsize=RADIAL_AXIS_LABELSIZE
    )
    ax_density.set_ylabel("PDF", fontsize=RADIAL_AXIS_LABELSIZE)
    ax_density.set_title(
        r"$f(\log(a_{\mathrm{out}}))=N(\log(a_{\mathrm{out}}))/N_{\mathrm{tot}}$",
        fontweight="bold",
        fontsize=RADIAL_TITLE_FONTSIZE,
    )
    ax_density.set_xlim([1, 6])
    ax_density.set_ylim([1e-3, None])
    ax_density.set_yscale("log")

    _legend_dedup(ax_density, framealpha=0.8, loc="best")

    plt.tight_layout()
    _save(fig, output_dir, "A_gen1")
    plt.close(fig)


def plot_sma_merger_distribution(
    plot_sma_data: Dict,
    trap_band: Tuple[float, float] | None,
    scenarios: Sequence[str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    model_colors: Mapping[str, Any] | None = None,
    filename_suffix: str | None = None,
) -> None:
    """Plot the initial outer-orbit radius distribution for merging 1g BBHs."""
    output_dir = plot_sma_data["outfile"]
    fig, axes = plt.subplots(2, 1, figsize=(18, 14), sharex=True)
    ax_counts, ax_density = axes

    merger_samples = {}
    scenario_names = _scenario_keys(plot_sma_data, scenarios)
    for scenario in scenario_names:
        merger_a = _positive_finite_log10(
            plot_sma_data[scenario]["mergers"].get("Aini", [])
        )
        if merger_a.size:
            merger_samples[scenario] = merger_a

    if not merger_samples:
        plt.close(fig)
        return

    all_merger_a = np.concatenate(list(merger_samples.values()))
    bins = (
        np.linspace(all_merger_a.min(), all_merger_a.max(), 40)
        if all_merger_a.size > 1
        else 15
    )

    for scenario in scenario_names:
        merger_a = merger_samples.get(scenario)
        if merger_a is None:
            continue

        merger_count = int(plot_sma_data[scenario]["mergers"].get("count", 0))
        color = _scenario_color(scenario, "merger", model_colors)
        label = _scenario_title(scenario, model_labels)

        ax_counts.hist(
            merger_a,
            bins=bins,
            alpha=0.85,
            color=color,
            edgecolor=color,
            linewidth=3.5,
            label=f"{label} (N={_format_count_label(merger_count)})",
            histtype="step",
        )

        if merger_a.size > 1:
            _plot_kde_or_density(
                ax_density,
                merger_a,
                color=color,
                linewidth=2.5,
                label=label,
            )

    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    ax_counts.set_ylabel("Count", fontsize=MERGER_AXIS_LABELSIZE)
    ax_counts.set_title(
        r"$N_{\mathrm{merg}}(\log(a_{\mathrm{out}}))=\mathrm{d}N_{\mathrm{merg}}/\mathrm{d}(\log(a_{\mathrm{out}}))$",
        fontweight="bold",
        fontsize=RADIAL_TITLE_FONTSIZE,
    )
    ax_counts.set_yscale("log")

    if trap_band is not None:
        trap_lo, trap_hi = trap_band
        for ax in axes:
            ax.axvspan(trap_lo, trap_hi, alpha=0.3, label=r"SMBH trap-radius range")

    _legend_dedup(ax_counts, framealpha=0.8, loc="best")

    ax_density.set_xlabel(
        r"$\log_{10}(a_{\rm out}/r_g)$", fontsize=MERGER_AXIS_LABELSIZE
    )
    ax_density.set_ylabel("PDF", fontsize=MERGER_AXIS_LABELSIZE)
    ax_density.set_title(
        r"$f_{\mathrm{merg}}(\log(a_{\mathrm{out}}))=N_{\mathrm{merg}}(\log(a_{\mathrm{out}}))/N_{\mathrm{merg,tot}}$",
        fontweight="bold",
        fontsize=RADIAL_TITLE_FONTSIZE,
    )
    ax_density.set_xlim([1, 6])
    ax_density.set_ylim([1e-3, None])
    ax_density.set_yscale("log")

    _legend_dedup(ax_density, framealpha=0.8, loc="best")

    plt.tight_layout()
    _save(fig, output_dir, _stem_with_suffix("A_gen1_mergers", filename_suffix))
    plt.close(fig)


def plot_mass_radius_distribution(
    plot_mass_radius_data: Dict,
    trap_band: Tuple[float, float] | None = None,
    scenarios: Sequence[str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    model_colors: Mapping[str, Any] | None = None,
) -> None:
    """Plot the primary-mass-weighted radial distribution of first-generation BHs."""
    output_dir = plot_mass_radius_data["outfile"]
    fig, axes = plt.subplots(figsize=(18, 7))
    ax_density = axes

    scenario_names = _scenario_keys(plot_mass_radius_data, scenarios)

    # Global bins so all scenarios are directly comparable
    all_radius_list = []
    for scenario in scenario_names:
        if scenario not in plot_mass_radius_data:
            continue
        radius = _positive_finite_log10(
            plot_mass_radius_data[scenario]["all"].get("radius", [])
        )
        if radius.size:
            all_radius_list.append(radius)

    if not all_radius_list:
        plt.close(fig)
        return

    all_radius = np.concatenate(all_radius_list)
    bins = (
        np.linspace(all_radius.min(), all_radius.max(), 40)
        if all_radius.size > 1
        else 15
    )

    for scenario in scenario_names:
        if scenario not in plot_mass_radius_data:
            continue

        radius = _positive_finite_log10(
            plot_mass_radius_data[scenario]["all"].get("radius", [])
        )
        mass = positive_finite(plot_mass_radius_data[scenario]["all"].get("mass", []))

        merger_radius = _positive_finite_log10(
            plot_mass_radius_data[scenario]["mergers"].get("radius", [])
        )
        merger_mass = positive_finite(
            plot_mass_radius_data[scenario]["mergers"].get("mass", [])
        )

        n_all = min(radius.size, mass.size)
        radius = radius[:n_all]
        mass = mass[:n_all]

        n_merger = min(merger_radius.size, merger_mass.size)
        merger_radius = merger_radius[:n_merger]
        merger_mass = merger_mass[:n_merger]

        if radius.size == 0 or mass.size == 0:
            continue

        color = _scenario_color(scenario, "initial", model_colors)
        label = _scenario_title(scenario, model_labels)

        # # Top panel: weighted histogram, same style as SMA
        # ax_counts.hist(
        #     radius,
        #     bins=bins,
        #     weights=mass,
        #     alpha=0.8,
        #     color=color,
        #     edgecolor=color,
        #     linewidth=3.5,
        #     label=f"{scenario.title()}: Weighted all",
        #     histtype="step",
        # )

        # # Optional merger overlay
        # if merger_radius.size and merger_mass.size:
        #     ax_counts.hist(
        #         merger_radius,
        #         bins=bins,
        #         weights=merger_mass,
        #         alpha=0.25,
        #         color=color,
        #         edgecolor=color,
        #         linewidth=2.5,
        #         label=f"{scenario.title()}: Weighted mergers",
        #         histtype="stepfilled",
        #     )

        # Bottom panel: weighted KDE
        if radius.size > 1 and np.sum(mass) > 0:
            # Weight by BH mass so the curve reflects where system mass is
            # distributed radially, not just where the raw event count peaks.
            kde_weights = mass / np.sum(mass)
            _plot_kde_or_density(
                ax_density,
                radius,
                weights=kde_weights,
                color=color,
                linewidth=2.5,
                label=label,
            )

    # for ax in axes:
    #     ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax_density.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    if trap_band is not None:
        trap_lo, trap_hi = trap_band
        print(f"Shading trap band between log10(a_out)={trap_lo:.2f} and {trap_hi:.2f}")
        ax_density.axvspan(trap_lo, trap_hi, alpha=0.3, label=r"SMBH trap-radius range")

    # ax_counts.set_ylabel(r"Count")
    # ax_counts.set_title(
    #     r"Mass-weighted distribution vs. $\log(a_{\mathrm{out}})$",
    #     fontweight="bold",
    # )
    # ax_counts.set_title(
    #     r"$N(\log(a_{\mathrm{out}}))=\mathrm{d}N_{\mathrm{tot}}/\mathrm{d}(\log(a_{\mathrm{out}}))$",
    #     fontweight="bold",
    # )
    # ax_counts.set_yscale("log")
    # _legend_dedup(ax_counts, framealpha=0.8, loc="best")

    ax_density.set_xlabel(
        r"Initial $\log(a_{\mathrm{out}})\ [\log(r_g)]$",
        fontsize=RADIAL_AXIS_LABELSIZE,
    )
    ax_density.set_ylabel("Primary-Mass PDF", fontsize=RADIAL_AXIS_LABELSIZE)
    # ax_density.set_title(
    #     r"Mass-weighted radial distribution",
    #     fontweight="bold",
    # )
    ax_density.set_title(
        r"$f_M(\log(a_{\mathrm{out}}))=\mathrm{d}M_1/\mathrm{d}\log(a_{\mathrm{out}})\,/\,M_{1,\mathrm{tot}}$",
        fontweight="bold",
        fontsize=RADIAL_TITLE_FONTSIZE,
    )
    ax_density.set_xlim([1, 6])
    ax_density.set_yscale("log")
    ax_density.set_ylim([1e-3, None])
    _legend_dedup(ax_density, framealpha=0.8, loc="best")

    plt.tight_layout()
    _save(fig, output_dir, "mass_radius_distribution")
    plt.close(fig)


def plot_mass_radius_heatmap(
    plot_mass_radius_data: Dict,
    trap_band: Tuple[float, float] | None = None,
    scenarios: Sequence[str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    model_colors: Mapping[str, Any] | None = None,
    filename_suffix: str | None = None,
) -> None:
    """Plot a 2D mass-radius heatmap for first-generation embedded BHs.

    Each panel shows the full first-generation embedded population as a 2D
    density map in ``log(a_out / r_g)`` and ``log(M_1 / M_sun)``. The subset
    that eventually merges is overplotted so it can be visually separated from
    the full initial population.
    """
    output_dir = plot_mass_radius_data["outfile"]
    scenario_names = _scenario_keys(plot_mass_radius_data, scenarios)
    if not scenario_names:
        return
    fig, axes = plt.subplots(
        1,
        len(scenario_names),
        figsize=(max(8, 8.5 * len(scenario_names)), 7),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#c62828",
            markeredgecolor="none",
            markersize=7,
            alpha=0.6,
            label="To be merger",
        )
    ]
    if trap_band is not None:
        legend_handles.append(
            Patch(
                facecolor="grey",
                alpha=0.15,
                edgecolor="none",
                label=r"SMBH trap-radius range",
            )
        )

    all_radius_list = []
    all_mass_list = []
    for scenario in scenario_names:
        if scenario not in plot_mass_radius_data:
            continue
        radius, mass = _paired_positive_finite_log10(
            plot_mass_radius_data[scenario]["all"].get("radius", []),
            plot_mass_radius_data[scenario]["all"].get("mass", []),
        )
        if radius.size > 0:
            all_radius_list.append(radius)
            all_mass_list.append(mass)

    if not all_radius_list or not all_mass_list:
        plt.close(fig)
        return

    global_radius = np.concatenate(all_radius_list)
    global_mass = np.concatenate(all_mass_list)

    x_min = max(1.0, float(np.nanmin(global_radius)))
    x_max = min(6.0, float(np.nanmax(global_radius)))
    y_min, clipped_low_mass = _lower_edge_without_isolated_tail(global_mass)
    y_max = float(np.nanmax(global_mass))
    print(f"Global radius range: {x_min:.2f} to {x_max:.2f}")
    print(f"Global mass range: {y_min:.2f} to {y_max:.2f}")
    if clipped_low_mass:
        print(
            f"Ignoring {clipped_low_mass} isolated low-mass point(s) "
            "when setting the heatmap y-axis."
        )

    # if np.isclose(x_min, x_max):
    #     x_min -= 0.5
    #     x_max += 0.5
    # if np.isclose(y_min, y_max):
    #     y_min -= 0.5
    #     y_max += 0.5

    x_edges = np.linspace(x_min, x_max, 45)
    y_edges = np.linspace(y_min, y_max, 36)

    mappable = None
    for ax, scenario in zip(axes, scenario_names):
        if scenario not in plot_mass_radius_data:
            ax.set_visible(False)
            continue

        radius, mass = _paired_positive_finite_log10(
            plot_mass_radius_data[scenario]["all"].get("radius", []),
            plot_mass_radius_data[scenario]["all"].get("mass", []),
        )

        agn_merger_radius, agn_merger_mass = _paired_positive_finite_log10(
            plot_mass_radius_data[scenario]["agn_mergers"].get("radius", []),
            plot_mass_radius_data[scenario]["agn_mergers"].get("mass", []),
        )
        isolated_merger_radius, isolated_merger_mass = _paired_positive_finite_log10(
            plot_mass_radius_data[scenario]["post_agn_isolated_mergers"].get(
                "radius", []
            ),
            plot_mass_radius_data[scenario]["post_agn_isolated_mergers"].get(
                "mass", []
            ),
        )

        if radius.size == 0 or mass.size == 0:
            continue

        counts, _, _ = np.histogram2d(radius, mass, bins=[x_edges, y_edges])
        with np.errstate(divide="ignore"):
            density = np.log10(counts.T)
        density[counts.T <= 0] = np.nan

        mappable = ax.pcolormesh(
            x_edges,
            y_edges,
            density,
            cmap="YlGnBu",
            shading="auto",
        )

        # if agn_merger_radius.size > 0 and agn_merger_mass.size > 0:
        #     ax.scatter(
        #         agn_merger_radius,
        #         agn_merger_mass,
        #         s=18,
        #         color="#c62828",
        #         alpha=0.45,
        #         edgecolors="none",
        #         label="To be mergers",
        #         zorder=3,
        #     )

        # if isolated_merger_radius.size > 0 and isolated_merger_mass.size > 0:
        #     ax.scatter(
        #         isolated_merger_radius,
        #         isolated_merger_mass,
        #         s=26,
        #         marker="x",
        #         color="#1b1b1b",
        #         alpha=0.6,
        #         linewidths=1.2,
        #         label="post-AGN isolated",
        #         zorder=3,
        #     )

        if trap_band is not None:
            trap_lo, trap_hi = trap_band
            ax.axvspan(
                trap_lo,
                trap_hi,
                alpha=0.15,
                color="grey",
                label=r"trap range",
                zorder=1,
            )

        ax.set_title(
            _scenario_title(scenario, model_labels),
            fontweight="bold",
            fontsize=RADIAL_TITLE_FONTSIZE,
        )
        ax.set_xlabel(
            r"Initial $\log(a_{\mathrm{out}})\ [\log(r_g)]$",
            fontsize=RADIAL_AXIS_LABELSIZE,
        )
        ax.set_xlim([1, 6])
        ax.grid(True, alpha=0.2, linestyle="--", which="both")

    axes[0].set_ylabel(r"$\log(M_1/M_\odot)$", fontsize=RADIAL_AXIS_LABELSIZE)
    legend_ax = axes[min(1, len(axes) - 1)]
    _legend_dedup(legend_ax, loc="upper left", framealpha=0.8)
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, shrink=0.92)
        cbar.set_label(r"$\log_{10}(N_{\mathrm{1g}})$ per bin")

    # fig.legend(
    #     handles=legend_handles,
    #     loc="upper right",
    #     bbox_to_anchor=(0.5, 1),
    #     ncol=len(legend_handles),
    #     framealpha=0.9,
    # )

    # plt.tight_layout(rect=[0, 0, 1, 0.9])
    _save(fig, output_dir, _stem_with_suffix("mass_radius_heatmap", filename_suffix))
    plt.close(fig)


def plot_trel_tdamp_heatmap(
    first_gen: Mapping[str, np.ndarray], output_dir: str = "fig_damp"
) -> None:
    """Visualize where the first-generation population sits in t_rel/t_damp space."""
    trel = np.asarray(first_gen.get("trel", []), dtype=float)
    tdamp = np.asarray(first_gen.get("tdamp", []), dtype=float)

    if trel.size == 0 or tdamp.size == 0:
        return

    n = min(trel.size, tdamp.size)
    trel = trel[:n]
    tdamp = tdamp[:n]

    valid = np.isfinite(trel) & np.isfinite(tdamp) & (trel > 0) & (tdamp > 0)
    if not np.any(valid):
        return

    log_trel = np.log10(trel[valid])
    log_tdamp = np.log10(tdamp[valid])
    log_ratio = log_trel - log_tdamp

    bins = 30

    x_min = float(np.nanmin(log_trel))
    x_max = float(np.nanmax(log_trel))
    y_min = float(np.nanmin(log_tdamp))
    # y_max = float(np.nanmax(log_tdamp))
    y_max = 1

    # Guard against pathological cases where all values are identical, which
    # would otherwise produce zero-width bins and a blank or misleading image.
    if np.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5
    if np.isclose(y_min, y_max):
        y_min -= 0.5
        y_max += 0.5

    x_edges = np.linspace(x_min, x_max, bins + 1)
    y_edges = np.linspace(y_min, y_max, bins + 1)

    # Use true 2D binning rather than a sparse pivot table so every pixel stays
    # aligned with the correct physical bin even when many bins are empty.
    count_grid, _, _ = np.histogram2d(log_trel, log_tdamp, bins=[x_edges, y_edges])
    sum_grid, _, _ = np.histogram2d(
        log_trel, log_tdamp, bins=[x_edges, y_edges], weights=log_ratio
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_grid = np.divide(
            sum_grid,
            count_grid,
            out=np.full_like(sum_grid, np.nan, dtype=float),
            where=count_grid > 0,
        )

    # Suppress visually noisy bins with too few systems.
    mean_grid[count_grid <= 5] = np.nan

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        mean_grid.T,
        cmap="RdBu_r",
        vmin=-5,
        vmax=5,
        shading="auto",
    )
    plt.colorbar(im, label=r"$\log_{10}(t_{\mathrm{rel}}/t_{\mathrm{damp}})$")

    ax.set_xlabel(r"$\log(t_{\rm rel}/{\rm Myr})$", fontsize=MERGER_AXIS_LABELSIZE)
    ax.set_ylabel(r"$\log(t_{\rm damp}/{\rm Myr})$", fontsize=MERGER_AXIS_LABELSIZE)
    ax.set_xlim([x_edges[0], x_edges[-1]])
    ax.set_ylim([y_edges[0], y_edges[-1]])

    # diag_lo = max(x_edges[0], y_edges[0])
    # diag_hi = min(x_edges[-1], y_edges[-1])
    # if diag_hi > diag_lo:
    #     ax.plot(
    #         [diag_lo, diag_hi],
    #         [diag_lo, diag_hi],
    #         color="black",
    #         linestyle="--",
    #         linewidth=1.5,
    #         alpha=0.8,
    #         label=r"$t_{\mathrm{rel}}=t_{\mathrm{damp}}$",
    #     )
    ax.grid(True, alpha=0.25, linestyle="--", which="both")
    _legend_dedup(ax, loc="best", framealpha=0.8)

    plt.tight_layout()
    _save(fig, output_dir, "trel_tdamp_heatmap")
    plt.close(fig)


def plot_timescales_histogram(
    first_gen: Mapping[str, np.ndarray], output_dir: str = "fig_damp"
) -> None:
    """Compare the main first-generation duration timescales on a common log axis."""
    times = build_timescale_data(first_gen)
    fig, ax = plt.subplots(figsize=(14, 10))

    plotted = False
    all_vals = []
    for key in ("trel", "tdamp", "tmig", "tdel"):
        vals = _positive_finite_log10(times.get(key, []))
        if vals.size:
            all_vals.append(vals)

    bins = np.linspace(
        min(v.min() for v in all_vals),
        max(v.max() for v in all_vals),
        40,
    )
    for key, linestyle, histtype, color in (
        ("trel", "-", "stepfilled", "grey"),
        ("tdamp", "-", "stepfilled", "skyblue"),
        ("tmig", "--", "step", "#BE9A62"),
        ("tdel", "--", "step", "green"),
    ):
        vals = _positive_finite_log10(times.get(key, []))
        if vals.size == 0:
            continue
        # bins = np.linspace(vals.min(), vals.max(), 35) if vals.size > 1 else 15

        ax.hist(
            vals,
            bins=bins,
            color=color,
            edgecolor=color,
            alpha=0.5 if histtype == "stepfilled" else 1,
            histtype=histtype,
            linewidth=3.5 if histtype == "step" else 2.5,
            linestyle=linestyle,
            label=_LABELS.get(key, key),
        )
        plotted = True

    # dutycycle = positive_finite(first_gen.get("AGNlifetime_myr", []))
    # if dutycycle.size:
    #     ax.axvline(
    #         np.log10(np.nanmedian(dutycycle)),
    #         color="black",
    #         linestyle=":",
    #         linewidth=2.5,
    #         label="AGN duty cycle",
    #     )

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel(r"$\log(t/{\rm Myr})$", fontsize=MERGER_AXIS_LABELSIZE)
    ax.set_ylabel("Events", fontsize=MERGER_AXIS_LABELSIZE)
    ax.set_yscale("log")
    ax.set_xlim([None, np.log10(const.tHubble)])
    ax.grid(True, alpha=0.3, linestyle="--", which="both")
    _legend_dedup(ax, framealpha=0.8, loc="best")

    plt.tight_layout()
    _save(fig, output_dir, "timescales_histogram")
    plt.close(fig)


def _expanded_linear_edges(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([])
    v_min = float(np.nanmin(finite))
    v_max = float(np.nanmax(finite))
    if np.isclose(v_min, v_max):
        v_min -= 0.5
        v_max += 0.5
    return np.linspace(v_min, v_max, bins + 1)


def _paired_log_aini_timescale(
    first_gen: Mapping[str, np.ndarray], time_key: str
) -> Tuple[np.ndarray, np.ndarray]:
    aini = np.asarray(first_gen.get("Aini", []), dtype=float).ravel()
    timescale = np.asarray(first_gen.get(time_key, []), dtype=float).ravel()
    n = min(aini.size, timescale.size)
    if n == 0:
        return np.array([]), np.array([])
    aini = aini[:n]
    timescale = timescale[:n]
    valid = (
        np.isfinite(aini) & np.isfinite(timescale) & (aini > 0.0) & (timescale > 0.0)
    )
    if not np.any(valid):
        return np.array([]), np.array([])
    return np.log10(aini[valid]), np.log10(timescale[valid])


def plot_aini_timescale_heatmaps(
    simulation_data: Mapping[str, Mapping],
    output_dir: str = "fig_damp",
    scenarios: Sequence[str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    filename_suffix: str | None = None,
    bins: int = 50,
) -> None:
    """Show first-generation radial dependence of migration and damping clocks."""
    scenario_names = _scenario_keys(simulation_data, scenarios)
    time_keys = ("tmig", "tdamp")
    time_titles = {
        "tmig": r"$\log_{10}(t_{\mathrm{mig}}/\mathrm{Myr})$",
        "tdamp": r"$\log_{10}(t_{\mathrm{damp}}/\mathrm{Myr})$",
    }

    paired: dict[str, dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    all_x = []
    all_y_by_key = {key: [] for key in time_keys}
    for scenario in scenario_names:
        first_gen = simulation_data.get(scenario, {}).get("first_gen", {})
        paired[scenario] = {}
        for key in time_keys:
            x_vals, y_vals = _paired_log_aini_timescale(first_gen, key)
            paired[scenario][key] = (x_vals, y_vals)
            if x_vals.size:
                all_x.append(x_vals)
                all_y_by_key[key].append(y_vals)

    if not all_x:
        return

    x_edges = _expanded_linear_edges(np.concatenate(all_x), bins)
    y_edges_by_key = {
        key: _expanded_linear_edges(np.concatenate(values), bins)
        for key, values in all_y_by_key.items()
        if values
    }
    if x_edges.size == 0 or any(key not in y_edges_by_key for key in time_keys):
        return

    histograms: dict[tuple[str, str], np.ndarray] = {}
    color_max = 0.0
    for scenario in scenario_names:
        for key in time_keys:
            x_vals, y_vals = paired.get(scenario, {}).get(
                key, (np.array([]), np.array([]))
            )
            if x_vals.size == 0:
                continue
            counts, _, _ = np.histogram2d(
                x_vals, y_vals, bins=[x_edges, y_edges_by_key[key]]
            )
            with np.errstate(divide="ignore"):
                log_counts = np.log10(counts.T)
            log_counts[counts.T <= 0] = np.nan
            if np.any(np.isfinite(log_counts)):
                color_max = max(color_max, float(np.nanmax(log_counts)))
            histograms[(scenario, key)] = log_counts

    if not histograms:
        return

    nrows = len(scenario_names)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(16, max(5.0, 4.2 * nrows)),
        squeeze=False,
        sharex=True,
    )

    mappable = None
    for row, scenario in enumerate(scenario_names):
        for col, key in enumerate(time_keys):
            ax = axes[row, col]
            log_counts = histograms.get((scenario, key))
            if log_counts is None:
                ax.set_visible(False)
                continue

            mappable = ax.pcolormesh(
                x_edges,
                y_edges_by_key[key],
                log_counts,
                cmap="viridis",
                shading="auto",
                vmin=0.0,
                vmax=color_max if color_max > 0.0 else None,
            )
            if row == 0:
                ax.set_title(time_titles[key])
            if row == nrows - 1:
                ax.set_xlabel(r"$\log_{10}(A_{\mathrm{ini}}/r_g)$")
            ax.set_ylabel(time_titles[key])
            if col == 0:
                ax.text(
                    0.02,
                    0.96,
                    _scenario_title(scenario, model_labels),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=18,
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.8,
                    },
                )
            ax.grid(True, alpha=0.2, linestyle="--", which="both")

    if mappable is not None:
        fig.colorbar(
            mappable,
            ax=axes.ravel().tolist(),
            label=r"$\log_{10}(N_{\mathrm{bin}})$",
            shrink=0.92,
        )

    _save(
        fig,
        output_dir,
        _stem_with_suffix("aini_timescale_heatmaps", filename_suffix),
    )
    plt.close(fig)


def _plot_time_distribution_histograms(
    event_rate_data: Mapping[str, Mapping],
    time_key: str,
    xlabel: str,
    output_stem: str,
    output_dir: str = "fig_damp",
    scenarios: Sequence[str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    model_colors: Mapping[str, Any] | None = None,
    filename_suffix: str | None = None,
) -> None:
    scenario_names = _scenario_keys(event_rate_data, scenarios)
    series_by_scenario = {}
    all_series = []

    for scenario in scenario_names:
        scenario_data = event_rate_data.get(scenario)
        if not scenario_data:
            continue
        vals = np.concatenate(
            [
                _positive_finite_log10(
                    scenario_data.get("first_gen", {}).get(time_key, [])
                ),
                _positive_finite_log10(
                    scenario_data.get("n_gen", {}).get(time_key, [])
                ),
            ]
        )
        if vals.size == 0:
            continue
        series_by_scenario[scenario] = vals
        all_series.append(vals)

    if not all_series:
        return

    global_vals = np.concatenate(all_series)
    if global_vals.size > 1:
        bins = np.linspace(global_vals.min(), global_vals.max(), 40)
    else:
        bins = np.linspace(global_vals[0] - 0.5, global_vals[0] + 0.5, 15)

    fig, ax = plt.subplots(figsize=(14, 9))
    for scenario in scenario_names:
        vals = series_by_scenario.get(scenario)
        if vals is None:
            continue
        color = _scenario_color(scenario, "merger", model_colors)
        label = _scenario_title(scenario, model_labels)
        ax.hist(
            vals,
            bins=bins,
            color=color,
            edgecolor=color,
            histtype="step",
            linewidth=3.0,
            label=f"{label} (N={_format_count_label(vals.size)})",
        )

    ax.set_xlabel(
        xlabel,
        fontsize=MERGER_AXIS_LABELSIZE,
    )
    ax.set_ylabel("Events", fontsize=MERGER_AXIS_LABELSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TIME_HIST_TICK_LABELSIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TIME_HIST_TICK_LABELSIZE)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--", which="both")
    _legend_dedup(ax, framealpha=0.8, loc="best", fontsize=TIME_HIST_LEGEND_FONTSIZE)

    plt.tight_layout()
    _save(
        fig,
        output_dir,
        _stem_with_suffix(output_stem, filename_suffix),
    )
    plt.close(fig)


def plot_tmerg_distribution_histograms(
    event_rate_data: Mapping[str, Mapping],
    output_dir: str = "fig_damp",
    scenarios: Sequence[str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    model_colors: Mapping[str, Any] | None = None,
    filename_suffix: str | None = None,
) -> None:
    """Compare absolute merger-time distributions across scenarios."""
    _plot_time_distribution_histograms(
        event_rate_data,
        "tmerg",
        r"$\log_{10}(t_{\mathrm{merg}}/\mathrm{Myr})$",
        "tmerg_distribution_histograms",
        output_dir=output_dir,
        scenarios=scenarios,
        model_labels=model_labels,
        model_colors=model_colors,
        filename_suffix=filename_suffix,
    )


def plot_tini_distribution_histograms(
    event_rate_data: Mapping[str, Mapping],
    output_dir: str = "fig_damp",
    scenarios: Sequence[str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    model_colors: Mapping[str, Any] | None = None,
    filename_suffix: str | None = None,
) -> None:
    """Compare merger initial-time distributions across scenarios."""
    _plot_time_distribution_histograms(
        event_rate_data,
        "tini",
        r"$\log_{10}(t_{\mathrm{ini}}/\mathrm{Myr})$",
        "tini_distribution_histograms",
        output_dir=output_dir,
        scenarios=scenarios,
        model_labels=model_labels,
        model_colors=model_colors,
        filename_suffix=filename_suffix,
    )


def _bsi_timescale_myr(first_gen: Mapping[str, np.ndarray]) -> np.ndarray:
    if "t_bsi" in first_gen:
        return to_array(first_gen.get("t_bsi", [])).astype(float, copy=False)
    if "t_bsi_myr" in first_gen:
        return to_array(first_gen.get("t_bsi_myr", [])).astype(float, copy=False)
    if "bsi_timescale_s" in first_gen:
        return (
            to_array(first_gen.get("bsi_timescale_s", [])).astype(float, copy=False)
            / const.Myr
        )
    return np.array([], dtype=float)


def plot_bsi_delay_timescale(
    first_gen: Mapping[str, np.ndarray],
    output_dir: str = "fig_damp",
    filename_suffix: str | None = None,
) -> None:
    """Compare first-order weak-BSI timescales with binary delay times."""
    t_bsi = _bsi_timescale_myr(first_gen)
    tdel = to_array(first_gen.get("tdel", [])).astype(float, copy=False)
    bsi_count = to_array(first_gen.get("bsi_weak_count", [])).astype(float, copy=False)

    n = min(t_bsi.size, tdel.size)
    if n == 0:
        return

    t_bsi = t_bsi[:n]
    tdel = tdel[:n]
    if bsi_count.size >= n:
        bsi_count = bsi_count[:n]
    else:
        bsi_count = np.zeros(n, dtype=float)

    mask = np.isfinite(t_bsi) & np.isfinite(tdel) & (t_bsi > 0.0) & (tdel > 0.0)
    if not np.any(mask):
        return

    t_bsi = t_bsi[mask]
    tdel = tdel[mask]
    bsi_count = bsi_count[mask]
    log_t_bsi = np.log10(t_bsi)
    log_tdel = np.log10(tdel)

    fig, ax_scatter = plt.subplots(figsize=(10, 8))

    scatter = ax_scatter.scatter(
        log_t_bsi,
        log_tdel,
        c=np.log10(bsi_count + 1.0),
        cmap="viridis",
        s=36,
        alpha=0.75,
        edgecolors="none",
    )
    colorbar = fig.colorbar(scatter, ax=ax_scatter)
    colorbar.set_label(r"$\log_{10}(N_{\mathrm{BSI}}+1)$")

    diag_lo = min(float(log_t_bsi.min()), float(log_tdel.min()))
    diag_hi = max(float(log_t_bsi.max()), float(log_tdel.max()))
    if diag_hi > diag_lo:
        ax_scatter.plot(
            [diag_lo, diag_hi],
            [diag_lo, diag_hi],
            color="black",
            linestyle="--",
            linewidth=2.0,
            alpha=0.8,
            label=r"$t_{\mathrm{del}}=t_{\mathrm{BSI}}$",
        )

    ax_scatter.set_xlabel(
        r"$\log_{10}(t_{\mathrm{BSI}}/{\rm Myr})$",
        fontsize=MERGER_AXIS_LABELSIZE,
    )
    ax_scatter.set_ylabel(
        r"$\log_{10}(t_{\mathrm{del}}/{\rm Myr})$",
        fontsize=MERGER_AXIS_LABELSIZE,
    )
    ax_scatter.set_ylim([None, 1])
    # ax_scatter.set_xlim([None, 10])
    ax_scatter.grid(True, alpha=0.3, linestyle="--")
    _legend_dedup(ax_scatter, framealpha=0.8, loc="best")

    plt.tight_layout()
    _save(fig, output_dir, _stem_with_suffix("bsi_delay_timescale", filename_suffix))
    plt.close(fig)


def plot_stop_condition_statistics(
    stop_condition_data: Mapping[str, Mapping],
    output_dir: str = "fig_damp",
    scenarios: Sequence[str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    model_colors: Mapping[str, Any] | None = None,
) -> None:
    """Compare stored stop-condition counts and fractions across scenarios.

    This plot uses full event populations, not just merger subsets, because stop
    reasons are most informative for systems removed from the hierarchy.
    """
    reason_codes = [
        code for code in sorted(const.STOP_REASON_LABELS) if code != const.STOP_NONE
    ]
    scenario_keys = _scenario_keys(stop_condition_data, scenarios)
    if not scenario_keys:
        return

    count_matrix = np.zeros((len(reason_codes), len(scenario_keys)), dtype=float)
    for j, scenario in enumerate(scenario_keys):
        scenario_payload = stop_condition_data.get(scenario, {})
        merged = flatten_generation_data(
            {
                "first_gen": scenario_payload.get("first_gen", {}),
                "n_gen": scenario_payload.get("n_gen", {}),
            }
        )
        reasons = to_array(merged.get("first_gen", {}).get("stop_reason_flag", []))
        reasons_ng = to_array(merged.get("n_gen", {}).get("stop_reason_flag", []))
        all_reasons = np.concatenate([reasons, reasons_ng])
        all_reasons = all_reasons[np.isfinite(all_reasons)].astype(int, copy=False)
        all_reasons = all_reasons[all_reasons != const.STOP_NONE]
        for i, code in enumerate(reason_codes):
            count_matrix[i, j] = np.sum(all_reasons == code)

    if not np.any(count_matrix > 0):
        return

    frac_matrix = np.zeros_like(count_matrix)
    col_sums = count_matrix.sum(axis=0)
    nonzero = col_sums > 0
    frac_matrix[:, nonzero] = count_matrix[:, nonzero] / col_sums[nonzero]

    reason_labels = [const.STOP_REASON_LABELS[code] for code in reason_codes]
    scenario_labels = [_scenario_title(key, model_labels) for key in scenario_keys]
    scenario_colors = [
        _scenario_color(key, "initial", model_colors) for key in scenario_keys
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(18, max(7, 0.55 * len(reason_codes) + 3)), sharey=True
    )
    y = np.arange(len(reason_codes))

    for j, (label, color) in enumerate(zip(scenario_labels, scenario_colors)):
        axes[0].barh(
            y + (j - (len(scenario_keys) - 1) / 2) * 0.22,
            count_matrix[:, j],
            height=0.2,
            color=color,
            alpha=0.85,
            label=label,
        )
        axes[1].barh(
            y + (j - (len(scenario_keys) - 1) / 2) * 0.22,
            frac_matrix[:, j],
            height=0.2,
            color=color,
            alpha=0.85,
            label=label,
        )

    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("Stop condition")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(reason_labels)
    axes[0].grid(True, axis="x", alpha=0.3, linestyle="--")

    axes[1].set_xlabel("Fraction")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(reason_labels)
    axes[1].set_xlim(0.0, 1.0)
    axes[1].grid(True, axis="x", alpha=0.3, linestyle="--")

    _legend_dedup(axes[0], loc="best", framealpha=0.85)
    plt.tight_layout()
    _save(fig, output_dir, "stop_condition_statistics")
    plt.close(fig)


def _create_combined_trap_comparison(
    all_trapped_counts: list[np.ndarray],
    time_bin_centers: np.ndarray,
    scenario_labels: list[str],
    scenario_colors: list[str],
    output_dir: str | Path,
    filename_suffix: str | None = None,
) -> None:
    """Create a scenario-only comparison after the generation-resolved trap plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for trapped_counts, label, color in zip(
        all_trapped_counts, scenario_labels, scenario_colors
    ):
        ax.plot(
            time_bin_centers,
            trapped_counts,
            label=label,
            color=color,
            linewidth=2.5,
            alpha=0.8,
        )

    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel("Average Trapped BHs per Sample")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    ax.set_yscale("log")

    plt.tight_layout()
    _save(
        fig,
        output_dir,
        _stem_with_suffix("trapped_comparison", filename_suffix),
        dpi=150,
    )
    plt.close(fig)


def plot_bh_trap_evolution(
    scenarios_data: Mapping[str, Mapping] | None = None,
    *,
    scenario_damp: Mapping[str, Mapping] | None = None,
    scenario_lin: Mapping[str, Mapping] | None = None,
    scenario_log: Mapping[str, Mapping] | None = None,
    time_bins: int = 50,
    max_time: float | None = None,
    dutycycle: float = 1e1,
    output_dir: str = "fig_damp",
    scenarios: Sequence[str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    model_colors: Mapping[str, Any] | None = None,
    filename_suffix: str | None = None,
    save_generation_detail: bool = True,
) -> None:
    """Estimate how many BHs are in the migration trap as a function of time.

    This version uses the full event population rather than merger-only events.
    Systems that reach the trap and do not merge are counted as trapped until
    the end of the AGN duty cycle.
    """
    if scenarios_data is None:
        legacy_payload = {
            "damp": scenario_damp or {},
            "lin": scenario_lin or {},
            "log": scenario_log or {},
        }
        scenarios_data = {key: value for key, value in legacy_payload.items() if value}

    scenario_names = _scenario_keys(scenarios_data, scenarios)
    if not scenario_names:
        return

    fig, ax = plt.subplots(figsize=(14, 12))

    def _flatten_full_scenario(scenario_data: Mapping[str, Mapping]) -> Dict[str, Dict]:
        return flatten_generation_data(
            {
                "first_gen": scenario_data.get("first_gen", {}),
                "n_gen": scenario_data.get("n_gen", {}),
            }
        )

    flat_by_scenario = {
        scenario: _flatten_full_scenario(scenarios_data.get(scenario, {}))
        for scenario in scenario_names
    }

    def _safe_flat(data: Mapping, gen: str, key: str) -> np.ndarray:
        return _finite(data.get(gen, {}).get(key, []))

    def _infer_num_samples(
        scenario_data: Mapping[str, Mapping], flat_data: Mapping
    ) -> int:
        sample_table = scenario_data.get("metadata", {}).get("sample_table", {})
        sample_index = sample_table.get("sample_index", [])
        sample_index = np.asarray(sample_index).ravel()
        if sample_index.size:
            return int(np.unique(sample_index).size)

        all_indices = np.concatenate(
            [
                to_array(flat_data.get("first_gen", {}).get("sample_index", [])),
                to_array(flat_data.get("n_gen", {}).get("sample_index", [])),
            ]
        )
        if all_indices.size:
            return int(np.unique(all_indices).size)
        return 1

    if max_time is None:
        tmerg_series = []
        for flat_data in flat_by_scenario.values():
            for gen_type in ("first_gen", "n_gen"):
                vals = _safe_flat(flat_data, gen_type, "tmerg")
                if vals.size:
                    tmerg_series.append(vals)
        all_tmerg = np.concatenate(tmerg_series) if tmerg_series else np.array([])
        max_time = float(np.nanmax(all_tmerg)) if all_tmerg.size else dutycycle

    time_bin_edges = np.linspace(0, min(max_time, dutycycle), time_bins + 1)
    time_bin_centers = 0.5 * (time_bin_edges[:-1] + time_bin_edges[1:])

    def calculate_trapped_bh(
        tini_array: np.ndarray,
        tmerg_array: np.ndarray,
        trap_flag_array: np.ndarray,
        merger_flag_array: np.ndarray,
        num_samples: int,
    ) -> np.ndarray:
        # Systems that reach the trap but do not merge are counted until the
        # AGN duty cycle ends. Merger products leave the trapped population at
        # their merger time.
        end_times = np.where(merger_flag_array == 1, tmerg_array, dutycycle)
        counts = np.zeros(len(time_bin_edges) - 1)
        for i in range(len(time_bin_edges) - 1):
            t = time_bin_centers[i]
            mask = (
                (trap_flag_array == 1)
                & (tini_array <= t)
                & (end_times > t)
                & (t < dutycycle)
            )
            counts[i] = np.sum(mask)
        return counts / max(num_samples, 1)

    plot_scenarios = [
        (
            flat_by_scenario[scenario],
            _infer_num_samples(
                scenarios_data.get(scenario, {}), flat_by_scenario[scenario]
            ),
            _scenario_color(scenario, "initial", model_colors),
            _scenario_title(scenario, model_labels),
        )
        for scenario in scenario_names
    ]

    all_trapped_counts = []
    scenario_labels = []
    scenario_colors = []

    for data, num_samples, color, label in plot_scenarios:
        trapped_counts_total = None

        for gen_type, gen_label in [("first_gen", "1g"), ("n_gen", "ng")]:
            tini = _finite(data.get(gen_type, {}).get("tini", []))
            tmerg = _finite(data.get(gen_type, {}).get("tmerg", []))
            trap_reached = _finite(data.get(gen_type, {}).get("trap_reached_flag", []))
            merger_flag = _finite(data.get(gen_type, {}).get("merger_flag", []))
            if tini.size == 0:
                continue

            n = min(len(tini), len(tmerg), len(trap_reached), len(merger_flag))
            if n == 0:
                continue
            trapped_counts = calculate_trapped_bh(
                tini[:n],
                tmerg[:n],
                trap_reached[:n],
                merger_flag[:n],
                num_samples,
            )

            if trapped_counts_total is None:
                trapped_counts_total = trapped_counts.copy()
            else:
                trapped_counts_total += trapped_counts

            ax.plot(
                time_bin_centers,
                trapped_counts,
                label=f"{label} - {gen_label}",
                color=color,
                linewidth=3,
                alpha=0.8,
                linestyle="--" if gen_type == "n_gen" else "-",
            )

        if trapped_counts_total is not None:
            all_trapped_counts.append(trapped_counts_total)
            scenario_labels.append(label)
            scenario_colors.append(color)

    if not all_trapped_counts:
        plt.close(fig)
        return

    bar_width = (
        time_bin_centers[1] - time_bin_centers[0] if len(time_bin_centers) > 1 else 0.2
    )
    bottom = np.zeros(len(time_bin_centers))

    for i, (counts, label, color) in enumerate(
        zip(all_trapped_counts, scenario_labels, scenario_colors)
    ):
        ax.bar(
            time_bin_centers,
            counts,
            width=bar_width,
            alpha=0.3,
            color=color,
            label=label + " All gen",
            bottom=bottom if i > 0 else None,
        )
        bottom += counts

    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel("Mean Trapped BHs per Sample")
    ax.grid(True, alpha=0.3, linestyle="--")
    _legend_dedup(ax, framealpha=0.8, loc="best")

    plt.tight_layout()
    if save_generation_detail:
        _save(
            fig,
            output_dir,
            _stem_with_suffix("bh_trap_evolution", filename_suffix),
            dpi=150,
        )
    plt.close(fig)

    _create_combined_trap_comparison(
        all_trapped_counts,
        time_bin_centers,
        scenario_labels,
        scenario_colors,
        output_dir,
        filename_suffix=filename_suffix,
    )


def plot_event_rate_comparison(
    event_rate_data: Mapping[str, Mapping],
    output_dir: str = "fig_damp",
    scenarios: Sequence[str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    model_colors: Mapping[str, Any] | None = None,
    filename_suffix: str | None = None,
) -> None:
    """Compare binned merger-yield rates across scenarios.

    This intentionally uses histogram rates rather than a smooth KDE/fit. The
    y-axis is the number of merger events per AGN sample per unit
    ``log10(t_merg / Myr)``. Here ``tmerg`` is the absolute merger time
    measured from AGN start, not the post-trap delay time.
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    scenario_names = _scenario_keys(event_rate_data, scenarios)
    all_series = []
    for scenario in scenario_names:
        scenario_data = event_rate_data.get(scenario)
        if not scenario_data:
            continue
        all_vals = np.concatenate(
            [
                _positive_finite_log10(
                    scenario_data.get("first_gen", {}).get("tmerg", [])
                ),
                _positive_finite_log10(scenario_data.get("n_gen", {}).get("tmerg", [])),
            ]
        )
        if all_vals.size:
            all_series.append(all_vals)

    if not all_series:
        plt.close(fig)
        return

    global_vals = np.concatenate(all_series)
    x_min = float(np.floor(global_vals.min() / 0.25) * 0.25)
    x_max = float(np.ceil(global_vals.max() / 0.25) * 0.25)
    if x_max <= x_min:
        x_min -= 0.5
        x_max += 0.5

    # Keep bins coarse enough that sparse samples do not masquerade as a
    # precision fit, but fine enough to show broad timing shifts.
    bin_width = 0.25
    bins = np.arange(x_min, x_max + bin_width, bin_width)
    if bins.size < 5:
        bins = np.linspace(x_min, x_max, 5)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    actual_bin_widths = np.diff(bins)

    positive_rates = []

    for scenario in scenario_names:
        scenario_data = event_rate_data.get(scenario)
        if not scenario_data:
            continue
        flat = scenario_data
        color = _scenario_color(scenario, "initial", model_colors)
        label = _scenario_title(scenario, model_labels)
        num_samples = max(int(flat.get("num_samples", 0)), 1)

        channel_specs = (
            ("AGN/trap", "agn_channel_merger_flag", "-", 3.0),
            ("post-AGN isolated", "post_agn_isolated_merger_flag", ":", 2.5),
        )
        for channel_label, flag_key, linestyle, linewidth in channel_specs:
            vals = np.concatenate(
                [
                    _positive_finite_log10_by_flag(
                        flat.get("first_gen", {}), "tmerg", flag_key
                    ),
                    _positive_finite_log10_by_flag(
                        flat.get("n_gen", {}), "tmerg", flag_key
                    ),
                ]
            )
            if vals.size == 0:
                continue

            counts, _ = np.histogram(vals, bins=bins)
            rate = counts / (num_samples * actual_bin_widths)
            rate_err = np.sqrt(counts) / (num_samples * actual_bin_widths)
            valid = counts > 0
            if not np.any(valid):
                continue

            positive_rates.extend(rate[valid].tolist())
            ax.plot(
                bin_centers[valid],
                rate[valid],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                marker="o" if linestyle == "-" else "s",
                markersize=5.5,
                label=(
                    f"{label} {channel_label} " f"(N={_format_count_label(vals.size)})"
                ),
            )
            # lower = np.maximum(rate - rate_err, 1e-12)
            # upper = rate + rate_err
            # ax.fill_between(
            #     bin_centers[valid],
            #     lower[valid],
            #     upper[valid],
            #     color=color,
            #     alpha=0.14 if linestyle == "-" else 0.09,
            #     linewidth=0,
            # )

    ax.set_xlabel(
        r"$\log(t_{\mathrm{merg}}/{\rm Myr})$",
        fontsize=MERGER_AXIS_LABELSIZE,
    )
    ax.set_ylabel(
        r"$\mathrm{d}N_{\mathrm{merg}}/"
        r"(N_{\mathrm{sample}}\,\mathrm{d}\log t_{\mathrm{merg},\mathrm{abs}})$",
        fontsize=MERGER_AXIS_LABELSIZE,
    )
    # ax.set_title("Binned merger-rate comparison")
    ax.grid(True, alpha=0.3, linestyle="--", which="both")
    ax.set_yscale("log")
    if positive_rates:
        positive_rates = np.asarray(positive_rates, dtype=float)
        ax.set_ylim(
            bottom=max(np.nanmin(positive_rates) / 3.0, 1e-4),
            top=np.nanmax(positive_rates) * 3.0,
        )
    ax.set_xlim(bins[0], bins[-1])
    _legend_dedup(ax, framealpha=0.8, loc="best")

    plt.tight_layout()
    _save(fig, output_dir, _stem_with_suffix("event_rate_comparison", filename_suffix))
    plt.close(fig)


from scipy.stats import gaussian_kde
from scipy.interpolate import griddata


def plot_chieff_mass_ratio_contours(
    distribution_data: Mapping[str, Mapping],
    output_dir: str = "fig_damp",
    scenarios: Sequence[str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    model_colors: Mapping[str, Any] | None = None,
    filename_suffix: str | None = None,
) -> None:
    """Compare merger outcomes in effective spin versus mass ratio heatmaps."""
    scenario_names = _scenario_keys(distribution_data, scenarios)
    if not scenario_names:
        return

    scenario_points: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for scenario in scenario_names:
        scenario_data = distribution_data.get(scenario)
        if not scenario_data:
            continue
        flat = flatten_generation_data(scenario_data)
        q_values = np.concatenate(
            [
                _finite(flat.get("first_gen", {}).get("m2/m1", [])),
                _finite(flat.get("n_gen", {}).get("m2/m1", [])),
            ]
        )
        chieff_values = np.concatenate(
            [
                _finite(flat.get("first_gen", {}).get("chieff", [])),
                _finite(flat.get("n_gen", {}).get("chieff", [])),
            ]
        )
        n = min(q_values.size, chieff_values.size)
        if n == 0:
            continue
        q_values = q_values[:n]
        chieff_values = chieff_values[:n]
        valid = (
            np.isfinite(q_values)
            & np.isfinite(chieff_values)
            & (q_values > 0.0)
            & (q_values <= 1.0)
            & (chieff_values >= -1.0)
            & (chieff_values <= 1.0)
        )
        if np.any(valid):
            scenario_points[scenario] = (q_values[valid], chieff_values[valid])

    if not scenario_points:
        return

    # Create finer grid for smooth interpolation
    n_grid = 100
    x_edges = np.linspace(0.0, 1.0, n_grid)
    y_edges = np.linspace(-1.0, 1.0, n_grid)

    density_maps = {}
    vmax = 0.0
    for scenario, (q_values, chieff_values) in scenario_points.items():
        # First create coarse histogram
        coarse_bins = 30
        counts, x_edges_coarse, y_edges_coarse = np.histogram2d(
            q_values, chieff_values, bins=coarse_bins
        )

        # Get bin centers
        x_centers = (x_edges_coarse[:-1] + x_edges_coarse[1:]) / 2
        y_centers = (y_edges_coarse[:-1] + y_edges_coarse[1:]) / 2
        X_coarse, Y_coarse = np.meshgrid(x_centers, y_centers)

        # Create fine grid
        X_fine, Y_fine = np.meshgrid(x_edges, y_edges)

        # Interpolate to finer grid
        points_coords = np.column_stack([X_coarse.ravel(), Y_coarse.ravel()])
        density_fine = griddata(
            points_coords,
            counts.T.ravel(),
            (X_fine, Y_fine),
            method="cubic",
            fill_value=0,
        )

        with np.errstate(divide="ignore"):
            log_density = np.log10(density_fine)
        log_density[density_fine <= 0] = np.nan

        density_maps[scenario] = log_density
        if np.any(np.isfinite(log_density)):
            vmax = max(vmax, float(np.nanmax(log_density)))

    if not density_maps:
        return

    ncols = len(density_maps)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(max(9, 8.0 * ncols), 8),
        sharex=True,
        sharey=True,
    )
    if ncols == 1:
        axes = [axes]
    else:
        axes = np.asarray(axes, dtype=object).reshape(-1)

    mappable = None
    for ax, scenario in zip(axes, density_maps.keys()):
        points = scenario_points.get(scenario)
        if points is None:
            ax.set_visible(False)
            continue

        q_values, chieff_values = points
        label = _scenario_title(scenario, model_labels)
        log_density = density_maps[scenario]

        mappable = ax.pcolormesh(
            x_edges,
            y_edges,
            log_density,
            cmap="viridis",
            shading="auto",
            vmin=0.0,
            vmax=max(vmax, 0.0),
        )
        ax.set_title(
            f"{label}\nN={_format_count_label(q_values.size)}", fontweight="bold"
        )
        ax.set_xlabel(r"$q=m_2/m_1$", fontsize=MERGER_AXIS_LABELSIZE)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(True, alpha=0.2, linestyle="--", which="both")

    axes[0].set_ylabel(r"$\chi_{\mathrm{eff}}$", fontsize=MERGER_AXIS_LABELSIZE)
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, location="right", pad=0.02)
        cbar.set_label(r"$\log_{10}(\mathrm{density})$")

    _save(
        fig,
        output_dir,
        _stem_with_suffix("chieff_mass_ratio_heatmap", filename_suffix),
    )
    plt.close(fig)


def plot_chieff_mass_ratio_contours(
    distribution_data: Mapping[str, Mapping],
    output_dir: str = "fig_damp",
    scenarios: Sequence[str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    model_colors: Mapping[str, Any] | None = None,
    filename_suffix: str | None = None,
) -> None:
    """Compare merger outcomes in effective spin versus mass ratio heatmaps."""
    scenario_names = _scenario_keys(distribution_data, scenarios)
    if not scenario_names:
        return

    scenario_points: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for scenario in scenario_names:
        scenario_data = distribution_data.get(scenario)
        if not scenario_data:
            continue
        flat = flatten_generation_data(scenario_data)
        q_values = np.concatenate(
            [
                _finite(flat.get("first_gen", {}).get("m2/m1", [])),
                _finite(flat.get("n_gen", {}).get("m2/m1", [])),
            ]
        )
        chieff_values = np.concatenate(
            [
                _finite(flat.get("first_gen", {}).get("chieff", [])),
                _finite(flat.get("n_gen", {}).get("chieff", [])),
            ]
        )
        n = min(q_values.size, chieff_values.size)
        if n == 0:
            continue
        q_values = q_values[:n]
        chieff_values = chieff_values[:n]
        valid = (
            np.isfinite(q_values)
            & np.isfinite(chieff_values)
            & (q_values > 0.0)
            & (q_values <= 1.0)
            & (chieff_values >= -1.0)
            & (chieff_values <= 1.0)
        )
        if np.any(valid):
            scenario_points[scenario] = (q_values[valid], chieff_values[valid])

    if not scenario_points:
        return

    x_edges = np.linspace(0.0, 1.0, 36)
    y_edges = np.linspace(-1.0, 1.0, 42)

    histograms = {}
    vmax = 0.0
    for scenario, (q_values, chieff_values) in scenario_points.items():
        counts, _, _ = np.histogram2d(q_values, chieff_values, bins=[x_edges, y_edges])
        with np.errstate(divide="ignore"):
            log_counts = np.log10(counts.T)
        log_counts[counts.T <= 0] = np.nan
        histograms[scenario] = log_counts
        if np.any(np.isfinite(log_counts)):
            vmax = max(vmax, float(np.nanmax(log_counts)))

    if not histograms:
        return

    ncols = len(scenario_names)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(max(9, 8.0 * ncols), 8),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)

    mappable = None
    for ax, scenario in zip(axes, scenario_names):
        points = scenario_points.get(scenario)
        # if points is None:
        #     ax.set_visible(False)
        #     continue
        q_values, chieff_values = points
        label = _scenario_title(scenario, model_labels)
        log_counts = histograms.get(scenario)
        if log_counts is None:
            ax.set_visible(False)
            continue

        mappable = ax.pcolormesh(
            x_edges,
            y_edges,
            log_counts,
            cmap="viridis",
            shading="auto",
            vmin=0.0,
            vmax=max(vmax, 0.0),
        )
        ax.set_title(
            f"{label}\nN={_format_count_label(q_values.size)}", fontweight="bold"
        )
        ax.set_xlabel(r"$q=m_2/m_1$", fontsize=MERGER_AXIS_LABELSIZE)
        ax.set_xlim(8e-1, 1.0)
        ax.set_ylim(8e-1, 1.0)
        ax.grid(True, alpha=0.2, linestyle="--", which="both")

    axes[0].set_ylabel(r"$\chi_{\mathrm{eff}}$", fontsize=MERGER_AXIS_LABELSIZE)
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, location="right", pad=0.02)
        cbar.set_label(r"$\log_{10}(N_{\mathrm{merg}})$ per bin")

    _save(
        fig,
        output_dir,
        _stem_with_suffix("chieff_mass_ratio_heatmap", filename_suffix),
    )
    plt.close(fig)


def plot_smbh_mass_dependence(
    smbh_dependence_data: Mapping[str, Mapping],
    output_dir: str = "fig_damp",
    scenarios: Sequence[str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    model_colors: Mapping[str, Any] | None = None,
    filename_suffix: str | None = None,
) -> None:
    """Plot the first-generation radial distribution conditional on SMBH mass."""
    scenario_names = _scenario_keys(smbh_dependence_data, scenarios)
    if not scenario_names:
        return
    fig, axes = plt.subplots(
        1,
        len(scenario_names),
        figsize=(max(8, 8.5 * len(scenario_names)), 7),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)
    im = None
    scenario_points = {}
    all_log_mass = []
    x_edges = np.linspace(1.0, 6.0, 36)

    for scenario in scenario_names:
        payload = smbh_dependence_data.get(scenario)
        if not payload:
            continue

        event_table = payload.get("event_table", {})
        log_mass, log_aini = _paired_positive_finite_log10(
            event_table.get("M_SMBH", []), event_table.get("Aini", [])
        )
        if log_mass.size == 0:
            continue
        scenario_points[scenario] = (log_mass, log_aini)
        all_log_mass.append(log_mass)

    if all_log_mass:
        global_log_mass = np.concatenate(all_log_mass)
        y_min = float(np.nanmin(global_log_mass))
        y_max = float(np.nanmax(global_log_mass))
        if np.isclose(y_min, y_max):
            y_min -= 0.5
            y_max += 0.5
        y_edges = np.linspace(y_min, y_max, 12)
    else:
        y_edges = np.linspace(5.0, 8.0, 12)

    histograms = {}
    vmax = 0.0
    for scenario, (log_mass, log_aini) in scenario_points.items():
        hist, _, _ = np.histogram2d(log_mass, log_aini, bins=[y_edges, x_edges])
        row_totals = hist.sum(axis=1, keepdims=True)
        hist = np.divide(
            hist,
            row_totals,
            out=np.zeros_like(hist, dtype=float),
            where=row_totals > 0,
        )
        histograms[scenario] = hist
        if np.any(hist):
            vmax = max(vmax, float(np.nanmax(hist)))

    vmax = max(vmax, 0.01)

    for ax, scenario in zip(axes, scenario_names):
        hist = histograms.get(scenario)
        if hist is None:
            ax.set_visible(False)
            continue

        im = ax.imshow(
            hist,
            origin="lower",
            aspect="auto",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(
            _scenario_title(scenario, model_labels),
            fontweight="bold",
            fontsize=RADIAL_TITLE_FONTSIZE,
        )
        ax.set_xlabel(r"$\log(a_{\mathrm{out}}/r_g)$", fontsize=RADIAL_AXIS_LABELSIZE)
        ax.set_xlim([1, 6])
        ax.grid(True, alpha=0.2, linestyle="--", which="both")

    axes[0].set_ylabel(r"$\log(M_{\rm SMBH}/M_\odot)$", fontsize=RADIAL_AXIS_LABELSIZE)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, location="right", pad=0.02)
        cbar.set_label(r"$P(\log a_{\mathrm{out}}\mid \log M_{\rm SMBH})$ per bin")

    # plt.tight_layout()
    _save(fig, output_dir, _stem_with_suffix("smbh_mass_dependence", filename_suffix))
    plt.close(fig)


def _disk_profile_payload(
    smbh_mass: float,
    mdot_per_solar_mass: float | None = None,
    *,
    disk_model: str = "sirko",
    eddington_ratio: float = 0.1,
    alpha: float = 0.01,
    n_resolution: int | None = None,
) -> Dict[str, np.ndarray | float | str]:
    """Compute pAGN disk profiles for one SMBH mass using the main model path."""
    if n_resolution is None:
        n_resolution = int(os.getenv("PAGN_DISK_N_RESOLUTION", "10000"))

    mdot_cgs = None
    if mdot_per_solar_mass is not None:
        mdot_cgs = float(smbh_mass) * float(mdot_per_solar_mass)

    profile = build_pagn_disk_profile(
        smbh_mass,
        model=disk_model,
        n_resolution=n_resolution,
        eddington_ratio=eddington_ratio,
        alpha=alpha,
        b=0,
        mdot_cgs=mdot_cgs,
    )

    r_over_rg = np.asarray(profile.r_list, dtype=float) / profile.rg
    q_values = np.asarray(profile.Q_list, dtype=float)
    r_out_over_rg = _q1_radius_over_rg(
        r_over_rg,
        q_values,
        fallback=float(profile.r_out / profile.rg),
    )

    return {
        "smbh_mass": float(smbh_mass),
        "disk_model": profile.model_name,
        "r_over_rg": r_over_rg,
        "r_out_over_rg": r_out_over_rg,
        "tc": np.asarray(profile.Tc_list, dtype=float),
        "q": q_values,
        "sigma": np.asarray(profile.sigma_list, dtype=float),
        "tau": np.asarray(profile.tau_list, dtype=float),
        "h_over_r": np.asarray(profile.h_list, dtype=float),
        "cs": np.asarray(profile.cs_list, dtype=float),
        "rho": np.asarray(profile.rho_list, dtype=float),
    }


def _disk_torque_payload(
    smbh_mass: float,
    mdot_per_solar_mass: float | None = None,
    *,
    disk_model: str = "sirko",
    eddington_ratio: float = 0.1,
    alpha: float = 0.01,
    n_resolution: int | None = None,
    migrator_mass_msun: float = 10.0,
    smoothing: float = 0.005,
) -> Dict[str, np.ndarray | float | str]:
    """Compute a pAGN torque profile using the main simulation torque model."""
    if n_resolution is None:
        n_resolution = int(os.getenv("PAGN_DISK_N_RESOLUTION", "10000"))

    mdot_cgs = None
    if mdot_per_solar_mass is not None:
        mdot_cgs = float(smbh_mass) * float(mdot_per_solar_mass)

    disk_profile = build_pagn_disk_profile(
        smbh_mass,
        model=disk_model,
        n_resolution=n_resolution,
        eddington_ratio=eddington_ratio,
        alpha=alpha,
        b=0,
        mdot_cgs=mdot_cgs,
    )
    params = MigrationParameters(alpha=alpha)
    torque_profile = compute_torque_profile(
        disk_profile.disk,
        migrator_mass_msun=migrator_mass_msun,
        smoothing=smoothing,
        params=params,
    )
    trap_radii_cm = find_stable_traps(torque_profile)
    torque_radius_over_rg = np.asarray(torque_profile.radius_rg, dtype=float)
    q_radius_over_rg = np.asarray(disk_profile.r_list, dtype=float) / disk_profile.rg
    r_out_over_rg = _q1_radius_over_rg(
        q_radius_over_rg,
        np.asarray(disk_profile.Q_list, dtype=float),
        fallback=float(disk_profile.r_out / disk_profile.rg),
    )

    return {
        "smbh_mass": float(smbh_mass),
        "disk_model": disk_profile.model_name,
        "r_over_rg": torque_radius_over_rg,
        "r_out_over_rg": r_out_over_rg,
        "gamma_tot_cgs": np.asarray(torque_profile.gamma_tot, dtype=float) * 1.0e7,
        "gamma_i_cgs": np.asarray(torque_profile.gamma_i, dtype=float) * 1.0e7,
        "gamma_th_cgs": np.asarray(torque_profile.gamma_th, dtype=float) * 1.0e7,
        "trap_radii_over_rg": (
            np.asarray(trap_radii_cm, dtype=float) / disk_profile.rg
        ),
    }


def _q1_radius_over_rg(
    radius_over_rg: np.ndarray,
    q_values: np.ndarray,
    *,
    fallback: float,
) -> float:
    """Return the outer Q=1 radius in rg units, falling back when absent."""
    radius = np.asarray(radius_over_rg, dtype=float)
    q = np.asarray(q_values, dtype=float)
    n = min(radius.size, q.size)
    if n == 0:
        return float(fallback)

    radius = radius[:n]
    q = q[:n]
    valid = np.isfinite(radius) & np.isfinite(q) & (radius > 0.0) & (q > 0.0)
    radius = radius[valid]
    q = q[valid]
    if radius.size == 0:
        return float(fallback)

    order = np.argsort(radius)
    radius = radius[order]
    q = q[order]
    log_radius = np.log10(radius)
    log_q = np.log10(q)

    exact = np.where(np.isclose(log_q, 0.0, rtol=0.0, atol=1.0e-8))[0]
    if exact.size:
        return float(radius[exact[-1]])

    crossing_indices = np.where(np.sign(log_q[1:]) != np.sign(log_q[:-1]))[0]
    if crossing_indices.size == 0:
        return float(fallback)

    roots = []
    for idx in crossing_indices:
        x0, x1 = log_radius[idx], log_radius[idx + 1]
        y0, y1 = log_q[idx], log_q[idx + 1]
        if y1 == y0:
            continue
        roots.append(10.0 ** (x0 - y0 * (x1 - x0) / (y1 - y0)))

    if not roots:
        return float(fallback)
    return float(np.max(roots))


def _disk_plot_radius_mask(profile: Mapping[str, np.ndarray | float | str], radius):
    """Limit one plotted line to its Q=1 outer radius."""
    radius = np.asarray(radius, dtype=float)
    r_out = float(profile.get("r_out_over_rg", np.nan))
    if not np.isfinite(r_out) or r_out <= 0.0:
        return np.ones_like(radius, dtype=bool)
    return radius <= r_out


def _disk_radius_bounds(
    profiles: Sequence[Mapping[str, np.ndarray | float | str]],
) -> tuple[float, float] | None:
    """Return finite plot limits from solved profile radii and Q=1 outer edges."""
    radii = []
    r_out_values = []
    for profile in profiles:
        radius = np.asarray(profile["r_over_rg"], dtype=float)
        valid_radius = radius[
            np.isfinite(radius) & (radius > 0) & _disk_plot_radius_mask(profile, radius)
        ]
        if valid_radius.size:
            radii.append(valid_radius)

        r_out = float(profile["r_out_over_rg"])
        if np.isfinite(r_out) and r_out > 0:
            r_out_values.append(r_out)

    if not radii:
        return None

    all_radii = np.concatenate(radii)
    r_min = float(np.nanmin(all_radii))
    r_max = (
        float(np.nanmax(r_out_values)) if r_out_values else float(np.nanmax(all_radii))
    )
    if r_min <= 0 or r_max <= r_min:
        return None
    return r_min, r_max


def _semi_analytic_disk_profile(
    r_over_rg: np.ndarray,
    smbh_mass: float,
    alpha: float = 0.01,
) -> Dict[str, np.ndarray]:
    """Semi-analytic AGN disk scalings in the user's supplied convention."""
    radius = np.asarray(r_over_rg, dtype=float)
    x = radius / (10.0**3.2)
    inner = radius <= 10.0**3.2
    mass_factor = smbh_mass / 1.0e8
    alpha_factor = alpha / 0.01

    sigma = (
        7.94e5
        * mass_factor ** (4.0 / 5.0)
        * alpha_factor
        * np.where(inner, x**0.8, x ** (-1.49))
    )
    h_over_r = (
        7.59e-3
        * mass_factor ** (-3.0 / 20.0)
        * alpha_factor ** (-1.0)
        * np.where(inner, x ** (-0.6), x**0.5)
    )
    cs = (
        5.37e6
        * mass_factor ** (3.0 / 2.0)
        * alpha_factor ** (-1.0)
        * np.where(inner, x ** (-1.1), 1.0)
    )

    return {"sigma": sigma, "h_over_r": h_over_r, "cs": cs}


def plot_agn_disk_parameter_profiles(
    output_dir: str = "fig_damp",
    smbh_masses: Sequence[float] = (1e6, 1e7, 1e8),
    mdot_per_solar_mass: float | None = None,
    *,
    disk_model: str = "sirko",
    eddington_ratio: float = 0.1,
    alpha: float = 0.01,
    n_resolution: int | None = None,
) -> None:
    """Plot useful pAGN AGN disk parameters versus radius."""
    profiles = [
        _disk_profile_payload(
            float(mass),
            mdot_per_solar_mass,
            disk_model=disk_model,
            eddington_ratio=eddington_ratio,
            alpha=alpha,
            n_resolution=n_resolution,
        )
        for mass in smbh_masses
    ]

    parameters = [
        # ("tc", r"$T_{\mathrm{c}}\;(\mathrm{K})$"),
        ("sigma", r"$\Sigma\;(\mathrm{g\,cm^{-2}})$"),
        # ("rho", r"$\rho\;(\mathrm{g\,cm^{-3}})$"),
        # ("q", r"$Q$"),
        # ("tau", r"$\tau$"),
        ("h_over_r", r"$H/R$"),
        # ("cs", r"$c_{\mathrm{s}}\;(\mathrm{cm\,s^{-1}})$"),
    ]
    colors = ["#1f77b4", "#BE9A62", "#2A7F62"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    radius_bounds = _disk_radius_bounds(profiles)

    for ax, (key, ylabel) in zip(axes, parameters):
        for profile, color in zip(profiles, colors):
            radius = np.asarray(profile["r_over_rg"], dtype=float)
            values = np.asarray(profile[key], dtype=float)
            mask = (
                np.isfinite(radius) & np.isfinite(values) & (radius > 0) & (values > 0)
            )
            if not np.any(mask):
                continue

            mass_label = (
                rf"$M_\bullet=10^{{{int(np.log10(profile['smbh_mass']))}}}\,M_\odot$"
            )
            ax.plot(
                radius[mask],
                values[mask],
                color=color,
                lw=3.8,
                label=mass_label,
                alpha=0.8,
            )
            # ax.axvline(
            #     profile["r_out_over_rg"],
            #     color=color,
            #     lw=1.8,
            #     ls=":",
            #     alpha=0.75,
            # )

        ax.set_xscale("log")
        ax.set_yscale("log")
        if radius_bounds is not None:
            ax.set_xlim(radius_bounds)
        ax.set_ylabel(ylabel, fontsize=RADIAL_AXIS_LABELSIZE)
        ax.grid(True, alpha=0.25, linestyle="--", which="both")

    # for ax in axes[len(parameters) :]:
    #     ax.set_visible(False)

    # for ax in axes[4:]:
    #     if ax.get_visible():
    #         ax.set_xlabel(r"$R/r_g$")
    for ax in axes:
        ax.set_xlabel(r"$R/r_g$", fontsize=RADIAL_AXIS_LABELSIZE)

    # handles, labels = axes[0].get_legend_handles_labels()
    # if handles:
    #     fig.legend(
    #         handles,
    #         labels,
    #         loc="best",
    #         frameon=False,
    #     )
    axes[0].legend(loc="best", framealpha=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, output_dir, "agn_disk_parameter_profiles")
    plt.close(fig)


def plot_agn_disk_method_comparison(
    output_dir: str = "fig_damp",
    smbh_masses: Sequence[float] = (1e6, 1e7, 1e8),
    mdot_per_solar_mass: float | None = None,
    alpha: float = 0.01,
    *,
    disk_model: str = "sirko",
    eddington_ratio: float = 0.1,
    n_resolution: int | None = None,
) -> None:
    """Compare solved disk profiles with the supplied semi-analytic scalings."""
    profiles = [
        _disk_profile_payload(
            float(mass),
            mdot_per_solar_mass,
            disk_model=disk_model,
            eddington_ratio=eddington_ratio,
            alpha=alpha,
            n_resolution=n_resolution,
        )
        for mass in smbh_masses
    ]

    parameters = [
        ("sigma", r"$\Sigma\;(\mathrm{g\,cm^{-2}})$"),
        ("h_over_r", r"$H/R$"),
        # ("cs", r"$c_{\mathrm{s}}\;(\mathrm{cm\,s^{-1}})$"),
    ]
    colors = ["#1f77b4", "#BE9A62", "#2A7F62"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 20), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    radius_bounds = _disk_radius_bounds(profiles)

    for profile, color in zip(profiles, colors):
        radius = np.asarray(profile["r_over_rg"], dtype=float)
        analytic = _semi_analytic_disk_profile(radius, profile["smbh_mass"], alpha)
        mass_label = (
            rf"$M_\bullet=10^{{{int(np.log10(profile['smbh_mass']))}}}\,M_\odot$"
        )

        for ax, (key, ylabel) in zip(axes, parameters):
            solved_values = np.asarray(profile[key], dtype=float)
            analytic_values = np.asarray(analytic[key], dtype=float)
            solved_mask = (
                np.isfinite(radius)
                & np.isfinite(solved_values)
                & (radius > 0)
                & (solved_values > 0)
            )
            analytic_mask = (
                np.isfinite(radius)
                & np.isfinite(analytic_values)
                & (radius > 0)
                & (analytic_values > 0)
            )
            if np.any(solved_mask):
                ax.plot(
                    radius[solved_mask],
                    solved_values[solved_mask],
                    color=color,
                    lw=3.8,
                    alpha=0.85,
                    label=f"{mass_label} solved",
                )
            if np.any(analytic_mask):
                ax.plot(
                    radius[analytic_mask],
                    analytic_values[analytic_mask],
                    color=color,
                    lw=3.8,
                    ls="--",
                    alpha=0.95,
                    label=f"{mass_label} analytic",
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            if radius_bounds is not None:
                ax.set_xlim(radius_bounds)
            ax.set_ylabel(ylabel, fontsize=RADIAL_AXIS_LABELSIZE)
            ax.grid(True, alpha=0.25, linestyle="--", which="both")

    for profile, color in zip(profiles, colors):
        r_out = float(profile["r_out_over_rg"])
        if not np.isfinite(r_out) or r_out <= 0:
            continue
        for ax in axes:
            ax.axvline(r_out, color=color, lw=1.4, ls=":", alpha=0.75)
    axes[-1].set_xlabel(r"$R/r_g$", fontsize=RADIAL_AXIS_LABELSIZE)

    method_handles = [
        Line2D([0], [0], color="0.2", lw=2.5, ls="-", label="Solved disk"),
        Line2D([0], [0], color="0.2", lw=2.5, ls="--", label="Semi-analytic"),
        Line2D([0], [0], color="0.35", lw=1.4, ls=":", label=r"$R_{\rm out}$"),
    ]
    mass_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            lw=3.0,
            label=rf"$10^{{{int(np.log10(profile['smbh_mass']))}}}\,M_\odot$",
        )
        for profile, color in zip(profiles, colors)
    ]
    axes[0].legend(
        handles=mass_handles + method_handles,
        loc="best",
        framealpha=0.85,
        ncol=2,
    )

    axes[0].set_ylim([1e2, 1e8])
    axes[1].set_ylim([1e-3, 1e0])

    fig.tight_layout()
    _save(fig, output_dir, "agn_disk_method_comparison")
    plt.close(fig)


def plot_agn_disk_torque_profiles(
    output_dir: str = "fig_damp",
    smbh_masses: Sequence[float] = (1e6, 1e7, 1e8),
    mdot_per_solar_mass: float | None = None,
    *,
    disk_model: str = "sirko",
    eddington_ratio: float = 0.1,
    alpha: float = 0.01,
    n_resolution: int | None = None,
    migrator_mass_msun: float = 10.0,
    smoothing: float = 0.005,
) -> None:
    """Plot pAGN migration torque magnitudes using the main simulation model."""
    profiles = [
        _disk_torque_payload(
            float(mass),
            mdot_per_solar_mass,
            disk_model=disk_model,
            eddington_ratio=eddington_ratio,
            alpha=alpha,
            n_resolution=n_resolution,
            migrator_mass_msun=migrator_mass_msun,
            smoothing=smoothing,
        )
        for mass in smbh_masses
    ]

    colors = ["#1f77b4", "#BE9A62", "#2A7F62"]
    radius_bounds = _disk_radius_bounds(profiles)

    fig, ax = plt.subplots(figsize=(14, 8))
    mass_handles = []

    for profile, color in zip(profiles, colors):
        radius = np.asarray(profile["r_over_rg"], dtype=float)
        radius_mask = _disk_plot_radius_mask(profile, radius)
        gamma_tot_cgs = np.asarray(profile["gamma_tot_cgs"], dtype=float)
        gamma_abs_cgs = np.abs(gamma_tot_cgs)
        mask = (
            np.isfinite(radius)
            & np.isfinite(gamma_abs_cgs)
            & (radius > 0)
            & (gamma_abs_cgs > 0)
            & radius_mask
        )
        if not np.any(mask):
            continue

        mass_label = (
            rf"$M_\bullet=10^{{{int(np.log10(profile['smbh_mass']))}}}\,M_\odot$"
        )
        mass_handles.append(Line2D([0], [0], color=color, lw=3.2, label=mass_label))

        gamma_outward = np.where(mask & (gamma_tot_cgs > 0.0), gamma_abs_cgs, np.nan)
        gamma_inward = np.where(mask & (gamma_tot_cgs < 0.0), gamma_abs_cgs, np.nan)
        ax.plot(
            radius,
            gamma_outward,
            color=color,
            lw=3.2,
            ls="-.",
            alpha=0.9,
        )
        ax.plot(
            radius,
            gamma_inward,
            color=color,
            lw=3.2,
            ls="-",
            alpha=0.9,
        )

        r_out = float(profile["r_out_over_rg"])
        trap_radii = np.asarray(profile["trap_radii_over_rg"], dtype=float)
        trap_mask = np.isfinite(trap_radii) & (trap_radii > 0)
        if np.isfinite(r_out) and r_out > 0:
            trap_mask &= trap_radii <= r_out
        trap_radii = trap_radii[trap_mask]
        if trap_radii.size:
            for trap_radius in trap_radii:
                ax.axvline(trap_radius, color=color, lw=1.3, ls="--", alpha=0.6)

        # if np.isfinite(r_out) and r_out > 0:
        #     ax.axvline(r_out, color=color, lw=1.3, ls=":", alpha=0.65)

    ax.set_xscale("log")
    ax.set_yscale("log")
    if radius_bounds is not None:
        ax.set_xlim(radius_bounds)
    ax.set_xlabel(r"$R/r_g$", fontsize=RADIAL_AXIS_LABELSIZE)
    ax.set_ylabel(
        r"$|\Gamma_{\rm tot}|\;(\mathrm{g\,cm^2\,s^{-2}})$",
        fontsize=RADIAL_AXIS_LABELSIZE,
    )
    ax.grid(True, alpha=0.25, linestyle="--", which="both")

    direction_handles = [
        Line2D([0], [0], color="0.2", lw=3.2, ls="-", label="Outward migration"),
        Line2D([0], [0], color="0.2", lw=3.2, ls="-.", label="Inward migration"),
        Line2D([0], [0], color="0.2", lw=1.4, ls="--", label="Stable trap"),
        # Line2D([0], [0], color="0.35", lw=1.4, ls=":", label=r"$Q=1$"),
    ]
    ax.legend(
        handles=mass_handles + direction_handles,
        loc="best",
        framealpha=0.85,
        ncol=2,
    )

    fig.tight_layout()
    _save(fig, output_dir, "agn_disk_torque_profiles")
    plt.close(fig)
