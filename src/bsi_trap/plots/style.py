"""Shared Matplotlib style definitions for BSI-trap figures."""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover
    sns = None


PLOT_RCPARAMS: dict[str, object] = {
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

SCENARIO_COLORS: dict[str, dict[str, str]] = {
    "damp": {"initial": "#1f77b4", "merger": "#1f77b4"},
    "lin": {"initial": "#BE9A62", "merger": "#BE9A62"},
    "log": {"initial": "#6B4E25", "merger": "#6B4E25"},
    "with_gas_damp": {"initial": "#1f77b4", "merger": "#1f77b4"},
    "damp_no_copy": {"initial": "#FF7F0E", "merger": "#FF7F0E"},
    "with_gas_damp_no_copy": {"initial": "#FF7F0E", "merger": "#FF7F0E"},
    "with_gas_lin": {"initial": "#BE9A62", "merger": "#BE9A62"},
    "with_gas_log": {"initial": "#6B4E25", "merger": "#6B4E25"},
    "without_gas_damp": {"initial": "#2CA02C", "merger": "#2CA02C"},
    "without_gas_lin": {"initial": "#9467BD", "merger": "#9467BD"},
    "without_gas_log": {"initial": "#D62728", "merger": "#D62728"},
    "BSI": {"initial": "#17BECF", "merger": "#17BECF"},
}

FALLBACK_COLORS: tuple[str, ...] = (
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

GENERATION_STYLE: dict[str, dict[str, str]] = {
    "first_gen": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
    "n_gen": {"color": "#6382AA", "linestyle": "--", "marker": "s"},
}

RADIAL_AXIS_LABELSIZE = 34
RADIAL_TITLE_FONTSIZE = 30
MERGER_AXIS_LABELSIZE = 34
TIME_HIST_TICK_LABELSIZE = 30
TIME_HIST_LEGEND_FONTSIZE = 28


def apply_plot_style(extra_rc: Mapping[str, object] | None = None) -> None:
    """Apply the project plotting style once at import/runtime."""
    rc = dict(PLOT_RCPARAMS)
    if extra_rc:
        rc.update(extra_rc)
    plt.rcParams.update(rc)
    if sns is not None:
        sns.set_theme(style="whitegrid", rc=plt.rcParams)


def fallback_color(key: str) -> str:
    """Return a deterministic fallback color for an arbitrary scenario key."""
    return FALLBACK_COLORS[abs(hash(key)) % len(FALLBACK_COLORS)]


apply_plot_style()
