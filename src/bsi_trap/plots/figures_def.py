from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Mapping, Any


@dataclass(frozen=True)
class FigureDefinition:
    name: str
    module_name: str
    function_name: str
    category: str
    description: str


FIGURE_DEFINITIONS: dict[str, FigureDefinition] = {
    "bbh_sbh_population_timeline": FigureDefinition(
        "bbh_sbh_population_timeline",
        "bsi_trap.plots.timeline_evolution",
        "plot_bbh_sbh_population_timeline",
        "timeline",
        "Active BBH and single-BH population timeline.",
    ),
    "bbh_parameter_timeline": FigureDefinition(
        "bbh_parameter_timeline",
        "bsi_trap.plots.timeline_evolution",
        "plot_bbh_parameter_timeline",
        "timeline",
        "BBH evolution trajectory timeline.",
    ),
}


def resolve_figure_function(name: str) -> Callable[..., Any]:
    """Resolve a registered plotting function lazily to avoid import cycles."""
    definition = FIGURE_DEFINITIONS[name]
    module = import_module(definition.module_name)
    return getattr(module, definition.function_name)


def available_figures(category: str | None = None) -> Mapping[str, FigureDefinition]:
    """Return all figure definitions, optionally filtered by category."""
    if category is None:
        return dict(FIGURE_DEFINITIONS)
    return {
        name: definition
        for name, definition in FIGURE_DEFINITIONS.items()
        if definition.category == category
    }
