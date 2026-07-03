from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Mapping, Any


@dataclass(frozen=True)
class FigureDefinition:
    name: str
    function_name: str
    category: str
    description: str


FIGURE_DEFINITIONS: dict[str, FigureDefinition] = {
    "sma_distribution": FigureDefinition(
        "sma_distribution",
        "plot_sma_distribution",
        "radial",
        "Initial semi-major-axis distribution.",
    ),
    "sma_merger_distribution": FigureDefinition(
        "sma_merger_distribution",
        "plot_sma_merger_distribution",
        "merger",
        "Merger initial semi-major-axis distribution.",
    ),
    "mass_radius_distribution": FigureDefinition(
        "mass_radius_distribution",
        "plot_mass_radius_distribution",
        "radial",
        "BH mass-radius scatter/distribution.",
    ),
    "mass_radius_heatmap": FigureDefinition(
        "mass_radius_heatmap",
        "plot_mass_radius_heatmap",
        "radial",
        "BH mass-radius heatmap.",
    ),
    "aini_timescale_heatmaps": FigureDefinition(
        "aini_timescale_heatmaps",
        "plot_aini_timescale_heatmaps",
        "radial",
        "Aini-timescale heatmaps.",
    ),
    "smbh_mass_dependence": FigureDefinition(
        "smbh_mass_dependence",
        "plot_smbh_mass_dependence",
        "radial",
        "SMBH mass dependence diagnostics.",
    ),
    "bh_trap_evolution": FigureDefinition(
        "bh_trap_evolution",
        "plot_bh_trap_evolution",
        "radial",
        "Trap evolution diagnostics.",
    ),
    "event_rate_comparison": FigureDefinition(
        "event_rate_comparison",
        "plot_event_rate_comparison",
        "merger",
        "Event-rate comparison.",
    ),
    "chieff_mass_ratio_contours": FigureDefinition(
        "chieff_mass_ratio_contours",
        "plot_chieff_mass_ratio_contours",
        "merger",
        "Chi-effective versus mass-ratio contours.",
    ),
    "bsi_delay_timescale": FigureDefinition(
        "bsi_delay_timescale",
        "plot_bsi_delay_timescale",
        "merger",
        "BSI delay/timescale diagnostic.",
    ),
    "trel_tdamp_heatmap": FigureDefinition(
        "trel_tdamp_heatmap",
        "plot_trel_tdamp_heatmap",
        "general",
        "Relaxation/damping heatmap.",
    ),
    "timescales_histogram": FigureDefinition(
        "timescales_histogram",
        "plot_timescales_histogram",
        "general",
        "Timescale histogram.",
    ),
    "tmerg_distribution_histograms": FigureDefinition(
        "tmerg_distribution_histograms",
        "plot_tmerg_distribution_histograms",
        "general",
        "Merger-time histograms.",
    ),
    "tini_distribution_histograms": FigureDefinition(
        "tini_distribution_histograms",
        "plot_tini_distribution_histograms",
        "general",
        "Initial-time histograms.",
    ),
    "agn_disk_parameter_profiles": FigureDefinition(
        "agn_disk_parameter_profiles",
        "plot_agn_disk_parameter_profiles",
        "general",
        "AGN disk parameter profiles.",
    ),
    "agn_disk_torque_profiles": FigureDefinition(
        "agn_disk_torque_profiles",
        "plot_agn_disk_torque_profiles",
        "general",
        "AGN disk torque profiles.",
    ),
    "stop_condition_statistics": FigureDefinition(
        "stop_condition_statistics",
        "plot_stop_condition_statistics",
        "general",
        "Stop-condition statistics.",
    ),
}


def resolve_figure_function(name: str) -> Callable[..., Any]:
    """Resolve a registered plotting function lazily to avoid import cycles."""
    definition = FIGURE_DEFINITIONS[name]
    fig_plot = import_module("bsi_trap.plots.fig_plot")
    return getattr(fig_plot, definition.function_name)


def available_figures(category: str | None = None) -> Mapping[str, FigureDefinition]:
    """Return all figure definitions, optionally filtered by category."""
    if category is None:
        return dict(FIGURE_DEFINITIONS)
    return {
        name: definition
        for name, definition in FIGURE_DEFINITIONS.items()
        if definition.category == category
    }
