from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from bsi_trap.constant.constant import Myr
from bsi_trap.hdf5_load.timeline_catalog import TimelineCatalog


def plot_trap_population_timeline(
    catalog: TimelineCatalog,
    output_dir: Path,
) -> Path | None:
    """Plot the trap-reservoir population changes implied by event timelines."""
    if (
        catalog.trap_reach_s.size == 0
        and catalog.pair_formation_s.size == 0
        and catalog.merger_s.size == 0
    ):
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    time_myr, singles, binaries = _population_curves(catalog)
    axes[0].step(time_myr, singles, where="post", label="Trap singles", linewidth=2.0)
    axes[0].step(time_myr, binaries, where="post", label="BBHs", linewidth=2.0)
    axes[0].set_ylabel("Object count")
    axes[0].set_title("Trap Population Timeline")
    axes[0].legend(frameon=False)

    duty_cycle_myr = _duty_cycle_myr(catalog)
    for label, event_s, color in (
        ("Trap reach", catalog.trap_reach_s, "tab:green"),
        ("Pair up", catalog.pair_formation_s, "tab:blue"),
        ("Merge", catalog.merger_s, "tab:red"),
    ):
        event_time_myr, event_count = _cumulative_count(
            event_s,
            terminal_myr=duty_cycle_myr,
        )
        if event_time_myr.size > 0:
            axes[1].step(
                event_time_myr,
                event_count,
                where="post",
                label=label,
                linewidth=2.0,
                color=color,
            )

    axes[1].set_xlabel("Time [Myr]")
    axes[1].set_ylabel("Cumulative events")
    axes[1].legend(frameon=False)

    if np.isfinite(duty_cycle_myr):
        for axis in axes:
            axis.axvline(
                duty_cycle_myr,
                color="0.25",
                linestyle=":",
                linewidth=1.5,
                label=None,
            )
            axis.set_xlim(left=0.0, right=duty_cycle_myr)

    return _write_figure(fig, output_dir, "trap_population_timeline.png")


def plot_bbh_parameter_timeline(
    catalog: TimelineCatalog,
    output_dir: Path,
) -> Path | None:
    """Plot BBH evolution tracks in semimajor-axis/eccentricity phase space."""
    if catalog.binary_count == 0:
        return None

    track_mask = (
        np.isfinite(catalog.start_s)
        & np.isfinite(catalog.end_s)
        & (catalog.end_s >= catalog.start_s)
        & _within_track_duty(catalog)
        & (catalog.sma_initial_cm > 0.0)
        & (catalog.sma_final_cm > 0.0)
        & np.isfinite(catalog.ecc_initial)
        & np.isfinite(catalog.ecc_final)
    )
    if not np.any(track_mask):
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.5, 6.5))

    path_time_myr = _phase_space_color_times(catalog, track_mask)
    time_min = float(np.nanmin(path_time_myr))
    time_max = float(np.nanmax(path_time_myr))
    if time_max <= time_min:
        time_max = time_min + 1.0
    norm = Normalize(vmin=time_min, vmax=time_max)
    color_map = plt.get_cmap("viridis")

    for idx in np.flatnonzero(track_mask):
        track = _binary_track(catalog, idx)
        if track is None:
            track_time_s = np.array([catalog.start_s[idx], catalog.end_s[idx]])
            track_sma_cm = np.array([catalog.sma_initial_cm[idx], catalog.sma_final_cm[idx]])
            track_ecc = np.array([catalog.ecc_initial[idx], catalog.ecc_final[idx]])
        else:
            track_time_s, track_sma_cm, track_ecc = track

        linestyle = "-" if int(catalog.generation[idx]) == 1 else "--"
        color_values = track_time_s / Myr
        _plot_phase_track(
            axis,
            track_sma_cm,
            track_ecc,
            color_values,
            norm=norm,
            color_map=color_map,
            linestyle=linestyle,
        )

        start_color = color_map(norm(color_values[0]))
        end_color = color_map(norm(color_values[-1]))
        axis.scatter(
            track_sma_cm[0],
            track_ecc[0],
            color=start_color,
            marker="o",
            s=26,
            alpha=0.9,
        )
        axis.scatter(
            track_sma_cm[-1],
            track_ecc[-1],
            color=end_color,
            marker="*" if catalog.merged[idx] else "x",
            s=58 if catalog.merged[idx] else 34,
            alpha=0.9,
        )

    axis.autoscale_view()
    axis.set_xscale("log")
    axis.set_xlabel("Semimajor axis [cm]")
    axis.set_ylabel("Eccentricity")
    axis.set_ylim(bottom=-0.02, top=1.02)
    axis.set_title("BBH Phase-space Evolution")
    axis.grid(True, alpha=0.25)

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=color_map)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=axis)
    colorbar.set_label("Evolution time [Myr]")

    legend_handles = [
        plt.Line2D([0], [0], color="0.2", marker="o", linestyle="", label="Initial"),
        plt.Line2D([0], [0], color="0.2", marker="*", linestyle="", label="Merged"),
        plt.Line2D([0], [0], color="0.2", marker="x", linestyle="", label="Stopped"),
        plt.Line2D([0], [0], color="0.2", linestyle="-", label="1g"),
        plt.Line2D([0], [0], color="0.2", linestyle="--", label="ng"),
    ]
    axis.legend(handles=legend_handles, frameon=False, loc="best")

    return _write_figure(fig, output_dir, "bbh_parameter_timeline.png")


def _population_curves(catalog: TimelineCatalog) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times_s = np.concatenate(
        [
            catalog.trap_reach_s,
            catalog.pair_formation_s,
            catalog.merger_s,
        ]
    )
    single_delta = np.concatenate(
        [
            np.ones(catalog.trap_reach_s.size, dtype=np.int64),
            -2 * np.ones(catalog.pair_formation_s.size, dtype=np.int64),
            np.zeros(catalog.merger_s.size, dtype=np.int64),
        ]
    )
    binary_delta = np.concatenate(
        [
            np.zeros(catalog.trap_reach_s.size, dtype=np.int64),
            np.ones(catalog.pair_formation_s.size, dtype=np.int64),
            -1 * np.ones(catalog.merger_s.size, dtype=np.int64),
        ]
    )

    mask = np.isfinite(times_s)
    times_s = times_s[mask]
    single_delta = single_delta[mask]
    binary_delta = binary_delta[mask]
    order = np.argsort(times_s, kind="stable")

    time_myr = np.concatenate(([0.0], times_s[order] / Myr))
    singles = np.concatenate(([0], np.cumsum(single_delta[order])))
    binaries = np.concatenate(([0], np.cumsum(binary_delta[order])))

    duty_cycle_myr = _duty_cycle_myr(catalog)
    if np.isfinite(duty_cycle_myr) and (
        time_myr.size == 0 or time_myr[-1] < duty_cycle_myr
    ):
        time_myr = np.concatenate((time_myr, [duty_cycle_myr]))
        singles = np.concatenate((singles, [singles[-1]]))
        binaries = np.concatenate((binaries, [binaries[-1]]))

    return time_myr, singles, binaries


def _cumulative_count(
    times_s: np.ndarray,
    *,
    terminal_myr: float,
) -> tuple[np.ndarray, np.ndarray]:
    times_s = np.asarray(times_s, dtype=float)
    times_s = np.sort(times_s[np.isfinite(times_s)])
    if times_s.size == 0:
        if np.isfinite(terminal_myr):
            return np.array([0.0, terminal_myr]), np.array([0, 0], dtype=np.int64)
        return np.empty(0), np.empty(0, dtype=np.int64)

    time_myr = np.concatenate(([0.0], times_s / Myr))
    count = np.arange(times_s.size + 1, dtype=np.int64)
    if np.isfinite(terminal_myr) and time_myr[-1] < terminal_myr:
        time_myr = np.concatenate((time_myr, [terminal_myr]))
        count = np.concatenate((count, [count[-1]]))
    return time_myr, count


def _duty_cycle_myr(catalog: TimelineCatalog) -> float:
    duty_cycle_s = catalog.max_duty_cycle_s
    if not np.isfinite(duty_cycle_s):
        return np.nan
    return duty_cycle_s / Myr


def _within_track_duty(catalog: TimelineCatalog) -> np.ndarray:
    valid_duty = np.isfinite(catalog.duty_cycle_s)
    within = np.ones(catalog.binary_count, dtype=bool)
    within[valid_duty] = (
        (catalog.start_s[valid_duty] <= catalog.duty_cycle_s[valid_duty])
        & (catalog.end_s[valid_duty] <= catalog.duty_cycle_s[valid_duty])
    )
    return within


def _phase_space_color_times(
    catalog: TimelineCatalog,
    track_mask: np.ndarray,
) -> np.ndarray:
    if catalog.track_time_s.size > 0:
        values = catalog.track_time_s[np.isfinite(catalog.track_time_s)] / Myr
        if values.size > 0:
            return values
    return catalog.start_s[track_mask] / Myr


def _binary_track(
    catalog: TimelineCatalog,
    binary_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    mask = catalog.track_binary_idx == binary_idx
    if not np.any(mask):
        return None

    time_s = catalog.track_time_s[mask]
    sma_cm = catalog.track_sma_cm[mask]
    ecc = catalog.track_ecc[mask]
    valid = (
        np.isfinite(time_s)
        & np.isfinite(sma_cm)
        & np.isfinite(ecc)
        & (sma_cm > 0.0)
    )
    if np.count_nonzero(valid) < 2:
        return None

    order = np.argsort(time_s[valid], kind="stable")
    return time_s[valid][order], sma_cm[valid][order], ecc[valid][order]


def _plot_phase_track(
    axis,
    sma_cm: np.ndarray,
    ecc: np.ndarray,
    color_values: np.ndarray,
    *,
    norm: Normalize,
    color_map,
    linestyle: str,
) -> None:
    if sma_cm.size < 2:
        return

    points = np.column_stack((sma_cm, ecc))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    collection = LineCollection(
        segments,
        cmap=color_map,
        norm=norm,
        linewidth=1.4,
        alpha=0.72,
        linestyles=linestyle,
    )
    collection.set_array(color_values[:-1])
    axis.add_collection(collection)


def _write_figure(fig, output_dir: Path, file_name: str) -> Path:
    path = output_dir / file_name
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
