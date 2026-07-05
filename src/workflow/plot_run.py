import json
import logging
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

from bsi_trap.hdf5_load.timeline_catalog import read_timeline_catalogs
from bsi_trap.plots.style import apply_plot_style
from bsi_trap.plots.timeline_evolution import (
    plot_bbh_parameter_timeline,
    plot_bbh_sbh_population_timeline,
)

apply_plot_style()

logger = logging.getLogger(__name__)


def _resolve_output_file(value: object, summary_path: Path) -> Path | None:
    if not value:
        return None

    path = Path(str(value))
    if path.exists():
        return path

    if not path.is_absolute():
        candidates = (
            summary_path.parent / path,
            summary_path.parent.parent / path,
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate

    return None


def _resolve_generation_files(
    results: list[dict[str, Any]],
    summary_path: Path,
) -> list[Path]:
    output_files: list[Path] = []
    for result in results:
        if result.get("success") is not True:
            continue
        generation_files = result.get("generation_files")
        if not generation_files:
            generation_files = [result.get("output_file")]
        for value in generation_files:
            output_file = _resolve_output_file(value, summary_path)
            if output_file is not None:
                output_files.append(output_file)
    return output_files


def run_plotting_workflow(
    *,
    summary_path: Path,
    output_dir: Path,
) -> Path:
    logger.info("Starting plotting workflow")
    if not summary_path.exists():
        raise FileNotFoundError(f"Simulation summary not found: {summary_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)

    raw_results = summary.get("results", [])
    if not isinstance(raw_results, list):
        raise ValueError("Invalid simulation summary: 'results' must be a list.")
    if any(not isinstance(result, dict) for result in raw_results):
        raise ValueError("Invalid simulation summary: every result must be a mapping.")
    results = cast(list[dict[str, Any]], raw_results)

    figure_paths: list[Path] = []
    generation_files = _resolve_generation_files(results, summary_path)

    if generation_files:
        timeline_catalog = read_timeline_catalogs(generation_files)
        for timeline_figure in (
            plot_bbh_sbh_population_timeline(timeline_catalog, output_dir),
            plot_bbh_parameter_timeline(timeline_catalog, output_dir),
        ):
            if timeline_figure is not None:
                figure_paths.append(timeline_figure)

    manifest_path = output_dir / "plot_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "summary_path": str(summary_path),
                "figures": [str(path) for path in figure_paths],
                "generation_files_read": len(generation_files),
            },
            stream,
            indent=2,
        )

    logger.info("Plotting workflow finished: %s", manifest_path)
    return manifest_path
