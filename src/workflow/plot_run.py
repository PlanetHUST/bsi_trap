import json
import logging
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from bsi_trap.hdf5_load.merger_catalog import read_merger_catalogs
from bsi_trap.hdf5_load.timeline_catalog import read_timeline_catalogs
from bsi_trap.plots.merger_histograms import plot_merger_parameter_histograms
from bsi_trap.plots.style import apply_plot_style
from bsi_trap.plots.timeline_evolution import (
    plot_bbh_parameter_timeline,
    plot_trap_population_timeline,
)
from bsi_trap.storage.summary import GenerationSummary, read_generation_summary

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


def _load_generation_summaries(
    results: list[dict[str, Any]],
    summary_path: Path,
) -> list[GenerationSummary]:
    summaries: list[GenerationSummary] = []
    for output_file in _resolve_generation_files(results, summary_path):
        summaries.append(read_generation_summary(output_file))
    return summaries


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


def _concat_arrays(
    summaries: list[GenerationSummary],
    attr: str,
) -> np.ndarray:
    arrays = [getattr(summary, attr) for summary in summaries]
    arrays = [array for array in arrays if array.size > 0]
    if not arrays:
        return np.asarray([], dtype=float)
    return np.concatenate(arrays)


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _write_figure(fig: Figure, output_dir: Path, file_name: str) -> Path:
    path = output_dir / file_name
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _plot_generation_population(
    summaries: list[GenerationSummary],
    output_dir: Path,
) -> Path:
    labels = [
        f"{summary.sample_index}:g{summary.generation}"
        for summary in summaries
    ]
    embedded = [summary.embedded_count for summary in summaries]
    binaries = [summary.bbh_count for summary in summaries]
    singles = [summary.single_count for summary in summaries]
    pairs = [summary.pairing_event_count for summary in summaries]

    x = np.arange(len(labels))
    width = 0.2
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.bar(x - 1.5 * width, embedded, width, label="Embedded BH")
    axis.bar(x - 0.5 * width, binaries, width, label="BBH")
    axis.bar(x + 0.5 * width, singles, width, label="Single BH")
    axis.bar(x + 1.5 * width, pairs, width, label="Pair events")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_xlabel("Sample index")
    axis.set_ylabel("Count")
    axis.set_title("Trap-population counts")
    axis.legend(frameon=False)
    return _write_figure(fig, output_dir, "trap_population_counts.png")


def _plot_generation_masses(
    summaries: list[GenerationSummary],
    output_dir: Path,
) -> Path | None:
    m1 = _finite(_concat_arrays(summaries, "m1_msun"))
    m2 = _finite(_concat_arrays(summaries, "m2_msun"))
    if m1.size == 0 and m2.size == 0:
        return None

    fig, axis = plt.subplots(figsize=(8, 4))
    bins = max(5, min(20, int(np.sqrt(max(m1.size + m2.size, 1)))))
    if m1.size > 0:
        axis.hist(m1, bins=bins, alpha=0.65, label="Primary")
    if m2.size > 0:
        axis.hist(m2, bins=bins, alpha=0.65, label="Secondary")
    axis.set_xlabel(r"BH mass [$M_\odot$]")
    axis.set_ylabel("BBH count")
    axis.set_title("Smoke-run BBH mass distribution")
    axis.legend(frameon=False)
    return _write_figure(fig, output_dir, "bbh_mass_distribution.png")


def _plot_generation_times(
    summaries: list[GenerationSummary],
    output_dir: Path,
) -> Path | None:
    tini = _finite(_concat_arrays(summaries, "tini_myr"))
    tmerg = _finite(_concat_arrays(summaries, "tmerg_myr"))
    if tini.size == 0 and tmerg.size == 0:
        return None

    fig, axis = plt.subplots(figsize=(8, 4))
    bins = max(5, min(20, int(np.sqrt(max(tini.size + tmerg.size, 1)))))
    if tini.size > 0:
        axis.hist(tini, bins=bins, alpha=0.65, label="BBH formation")
    if tmerg.size > 0:
        axis.hist(tmerg, bins=bins, alpha=0.65, label="Merger")
    axis.set_xlabel("Time [Myr]")
    axis.set_ylabel("BBH count")
    axis.set_title("Smoke-run event times")
    axis.legend(frameon=False)
    return _write_figure(fig, output_dir, "bbh_event_times.png")


def _plot_generation_delays(
    summaries: list[GenerationSummary],
    output_dir: Path,
) -> Path | None:
    delays = _finite(_concat_arrays(summaries, "tdel_myr"))
    if delays.size == 0:
        return None

    fig, axis = plt.subplots(figsize=(8, 4))
    bins = max(5, min(20, int(np.sqrt(delays.size))))
    axis.hist(delays, bins=bins, color="tab:purple", alpha=0.75)
    axis.set_xlabel("Evolution delay [Myr]")
    axis.set_ylabel("BBH count")
    axis.set_title("Smoke-run BBH evolution delays")
    return _write_figure(fig, output_dir, "bbh_evolution_delays.png")


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
    sample_labels = [str(r.get("sample_index", i)) for i, r in enumerate(results)]
    success_values = [1 if r.get("success") is True else 0 for r in results]
    merger_counts = [int(r.get("merger_count") or 0) for r in results]
    generation_summaries = _load_generation_summaries(results, summary_path)
    generation_files = _resolve_generation_files(results, summary_path)

    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(sample_labels, success_values, color="tab:blue")
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Sample index")
    axis.set_ylabel("Success")
    axis.set_title("Simulation task status")
    figure_paths.append(_write_figure(fig, output_dir, "simulation_task_status.png"))

    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(sample_labels, merger_counts, color="tab:green")
    axis.set_xlabel("Sample index")
    axis.set_ylabel("Merged BBHs")
    axis.set_title("Merged BBH count by sample")
    figure_paths.append(_write_figure(fig, output_dir, "merger_count_by_sample.png"))

    if generation_summaries:
        figure_paths.append(
            _plot_generation_population(generation_summaries, output_dir)
        )
        for figure_path in (
            _plot_generation_masses(generation_summaries, output_dir),
            _plot_generation_times(generation_summaries, output_dir),
            _plot_generation_delays(generation_summaries, output_dir),
        ):
            if figure_path is not None:
                figure_paths.append(figure_path)

    if generation_files:
        timeline_catalog = read_timeline_catalogs(generation_files)
        for timeline_figure in (
            plot_trap_population_timeline(timeline_catalog, output_dir),
            plot_bbh_parameter_timeline(timeline_catalog, output_dir),
        ):
            if timeline_figure is not None:
                figure_paths.append(timeline_figure)

        merger_catalog = read_merger_catalogs(generation_files)
        merger_histogram = plot_merger_parameter_histograms(
            merger_catalog,
            output_dir,
        )
        if merger_histogram is not None:
            figure_paths.append(merger_histogram)

    manifest_path = output_dir / "plot_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "summary_path": str(summary_path),
                "figures": [str(path) for path in figure_paths],
                "generation_files_read": len(generation_summaries),
            },
            stream,
            indent=2,
        )

    logger.info("Plotting workflow finished: %s", manifest_path)
    return manifest_path
