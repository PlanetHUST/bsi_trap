from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bsi_trap.constant.constant import Myr, parsec
from bsi_trap.hdf5_load.timeline_catalog import TimelineCatalog

_MILLI_PARSEC_CM = 1.0e-3 * parsec
_POPULATION_OUTPUT_NAME = "bbh_sbh_population_timeline.png"
_PARAMETER_OUTPUT_NAME = "bbh_parameter_timeline.png"
_POPULATION_FIGSIZE = (10, 5.2)
_PARAMETER_FIGSIZE = (6.2, 6.2)

_TRAP_SINGLE_ARRIVAL_DELTA = 1
_TRAP_SINGLE_PAIR_FORMATION_DELTA = -2
_TRAP_SINGLE_MERGER_DELTA = 0
_TRAP_BBH_ARRIVAL_DELTA = 0
_TRAP_BBH_FORMATION_DELTA = 1
_TRAP_BBH_MERGER_DELTA = -1
_EVENT_ORDER_TRAP_REACH = 0
_EVENT_ORDER_FORMATION = 1
_EVENT_ORDER_MERGER = 2

_POPULATION_LINE_WIDTH = 2.4
_DUTY_CYCLE_LINE_WIDTH = 1.5
_TRACK_LINE_WIDTH = 0.55
_TRACK_ALPHA = 0.78
_COLLAPSED_TRACK_EPSILON = 1.0e-10
_COLLAPSED_TRACK_MARKER_SIZE = 4.0
_FINAL_MERGER_MARKER_SIZE = 34
_J_AXIS_FLOOR = 1.0e-3
_J_AXIS_CEILING = 1.0
_DEFAULT_Y_LIMITS = (-2.0, 2.0)
_EQUAL_Y_LIMIT_PAD = 0.5
_Y_AXIS_PAD_MIN = 0.04
_Y_AXIS_PAD_FRACTION = 0.04
_LOG_LIMIT_PAD_MIN = 0.015
_LOG_LIMIT_PAD_FRACTION = 0.04
_LOSS_CONE_SAMPLE_COUNT = 256
_LOSS_CONE_SLOPE = -0.65
_LOSS_CONE_INTERCEPT = -1.35
_LOSS_CONE_COLOR = "#8fb9dd"
_LOSS_CONE_ALPHA = 0.85


def plot_bbh_sbh_population_timeline(
    catalog: TimelineCatalog,
    output_dir: Path,
) -> Path | None:
    """Plot active trap single-BH and BBH counts through the saved timeline."""
    if (
        catalog.trap_reach_s.size == 0
        and catalog.binary_count == 0
        and catalog.merger_s.size == 0
    ):
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=_POPULATION_FIGSIZE)

    time_myr, single_count, binary_count = _population_curves(catalog)
    if time_myr.size == 0:
        plt.close(fig)
        return None

    axis.step(
        time_myr,
        single_count,
        where="post",
        color="tab:orange",
        linewidth=_POPULATION_LINE_WIDTH,
        label="Trap single BHs",
    )
    axis.step(
        time_myr,
        binary_count,
        where="post",
        color="tab:blue",
        linewidth=_POPULATION_LINE_WIDTH,
        label="Trap BBHs",
    )
    axis.set_title("BBH and Single-BH Population Timeline")
    axis.set_xlabel("Time [Myr]")
    axis.set_ylabel("Active count")
    axis.legend(frameon=False)
    axis.grid(True, axis="y", alpha=0.25)

    duty_cycle_myr = _duty_cycle_myr(catalog)
    if np.isfinite(duty_cycle_myr):
        axis.axvline(
            duty_cycle_myr,
            color="0.25",
            linestyle=":",
            linewidth=_DUTY_CYCLE_LINE_WIDTH,
            label=None,
        )
        axis.set_xlim(left=0.0, right=duty_cycle_myr)
    else:
        axis.set_xlim(left=0.0)

    return _write_figure(fig, output_dir, _POPULATION_OUTPUT_NAME)


def plot_bbh_parameter_timeline(
    catalog: TimelineCatalog,
    output_dir: Path,
) -> Path | None:
    """Plot BBH evolution in the BBH analogue of energy-angular momentum space."""
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
    fig, axis = plt.subplots(figsize=_PARAMETER_FIGSIZE)
    color_map = plt.get_cmap("tab20")
    plotted_x: list[np.ndarray] = []
    plotted_y: list[np.ndarray] = []
    collapsed_x: list[float] = []
    collapsed_y: list[float] = []
    collapsed_color: list[object] = []
    final_x: list[float] = []
    final_y: list[float] = []
    final_is_merge: list[bool] = []

    for line_number, idx in enumerate(np.flatnonzero(track_mask)):
        track = _binary_track(catalog, idx)
        if track is None:
            track_sma_cm = np.array(
                [catalog.sma_initial_cm[idx], catalog.sma_final_cm[idx]]
            )
            track_ecc = np.array([catalog.ecc_initial[idx], catalog.ecc_final[idx]])
        else:
            _, track_sma_cm, track_ecc = track

        j_over_jc, y_log_a = _bbh_energy_angular_momentum_coordinates(
            track_sma_cm,
            track_ecc,
        )
        if j_over_jc.size < 2:
            continue

        plotted_x.append(j_over_jc)
        plotted_y.append(y_log_a)
        final_x.append(float(j_over_jc[-1]))
        final_y.append(float(y_log_a[-1]))
        final_is_merge.append(bool(catalog.merged[idx]))
        color = color_map(line_number % color_map.N)
        if _track_extent(j_over_jc, y_log_a) <= _COLLAPSED_TRACK_EPSILON:
            collapsed_x.append(float(j_over_jc[0]))
            collapsed_y.append(float(y_log_a[0]))
            collapsed_color.append(color)
            continue

        axis.plot(
            j_over_jc,
            y_log_a,
            color=color,
            linewidth=_TRACK_LINE_WIDTH,
            alpha=_TRACK_ALPHA,
        )

    if len(plotted_x) == 0 or len(plotted_y) == 0:
        plt.close(fig)
        return None

    _style_bbh_energy_angular_momentum_axis(axis, plotted_x, plotted_y)
    if collapsed_x:
        axis.scatter(
            collapsed_x,
            collapsed_y,
            c=collapsed_color,
            s=_COLLAPSED_TRACK_MARKER_SIZE,
            marker=".",
            alpha=0.65,
            linewidths=0.0,
            zorder=3,
        )
    _plot_final_points(axis, final_x, final_y, final_is_merge)

    return _write_figure(fig, output_dir, _PARAMETER_OUTPUT_NAME)


def _population_curves(
    catalog: TimelineCatalog,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    formation_s = np.asarray(catalog.start_s, dtype=float)
    duty_cycle_s = catalog.max_duty_cycle_s
    if np.isfinite(duty_cycle_s):
        formation_s = formation_s[formation_s <= duty_cycle_s]
    times_s = np.concatenate(
        [
            catalog.trap_reach_s,
            formation_s,
            catalog.merger_s,
        ]
    )
    single_delta = np.concatenate(
        [
            _TRAP_SINGLE_ARRIVAL_DELTA
            * np.ones(catalog.trap_reach_s.size, dtype=np.int64),
            _TRAP_SINGLE_PAIR_FORMATION_DELTA
            * np.ones(formation_s.size, dtype=np.int64),
            _TRAP_SINGLE_MERGER_DELTA
            * np.ones(catalog.merger_s.size, dtype=np.int64),
        ]
    )
    binary_delta = np.concatenate(
        [
            _TRAP_BBH_ARRIVAL_DELTA
            * np.ones(catalog.trap_reach_s.size, dtype=np.int64),
            _TRAP_BBH_FORMATION_DELTA * np.ones(formation_s.size, dtype=np.int64),
            _TRAP_BBH_MERGER_DELTA * np.ones(catalog.merger_s.size, dtype=np.int64),
        ]
    )
    event_order = np.concatenate(
        [
            _EVENT_ORDER_TRAP_REACH
            * np.ones(catalog.trap_reach_s.size, dtype=np.int64),
            _EVENT_ORDER_FORMATION * np.ones(formation_s.size, dtype=np.int64),
            _EVENT_ORDER_MERGER * np.ones(catalog.merger_s.size, dtype=np.int64),
        ]
    )

    mask = np.isfinite(times_s)
    times_s = times_s[mask]
    single_delta = single_delta[mask]
    binary_delta = binary_delta[mask]
    event_order = event_order[mask]
    if times_s.size == 0:
        return np.empty(0), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    order = np.lexsort((event_order, times_s))

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


def _bbh_energy_angular_momentum_coordinates(
    sma_cm: np.ndarray,
    ecc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map BBH ``(a, e)`` tracks to the ``(E, J)``-like plotting plane."""
    sma_cm = np.asarray(sma_cm, dtype=float)
    ecc = np.asarray(ecc, dtype=float)
    valid = (
        np.isfinite(sma_cm)
        & np.isfinite(ecc)
        & (sma_cm > 0.0)
        & (ecc >= 0.0)
        & (ecc < 1.0)
    )
    if np.count_nonzero(valid) < 2:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    ecc_valid = ecc[valid]
    j_over_jc = np.sqrt(np.clip(1.0 - ecc_valid**2, np.finfo(float).tiny, 1.0))
    y_log_a = np.log10(sma_cm[valid] / _MILLI_PARSEC_CM)
    return j_over_jc, y_log_a


def _track_extent(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.nanmax(np.abs(x - x[0])) + np.nanmax(np.abs(y - y[0])))


def _style_bbh_energy_angular_momentum_axis(
    axis,
    plotted_x: list[np.ndarray],
    plotted_y: list[np.ndarray],
) -> None:
    x_values = np.concatenate(plotted_x)
    x_min, x_max = _log_plot_limits(
        x_values,
        lower_floor=_J_AXIS_FLOOR,
        upper_ceiling=_J_AXIS_CEILING,
    )
    y_values = np.concatenate(plotted_y)
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = _DEFAULT_Y_LIMITS
    if y_max <= y_min:
        y_min -= _EQUAL_Y_LIMIT_PAD
        y_max += _EQUAL_Y_LIMIT_PAD
    y_pad = max(_Y_AXIS_PAD_MIN, _Y_AXIS_PAD_FRACTION * (y_max - y_min))

    axis.set_xscale("log")
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min - y_pad, y_max + y_pad)
    axis.set_xlabel(r"$J/J_c$", fontsize=18)
    axis.set_ylabel(r"$\log_{10}(a/\mathrm{mpc})$", fontsize=18)
    axis.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=13,
        length=7,
    )
    axis.tick_params(which="minor", length=4)
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)

    _shade_loss_cone(axis, y_min - y_pad, y_max + y_pad, x_min, x_max)


def _log_plot_limits(
    values: np.ndarray,
    *,
    lower_floor: float,
    upper_ceiling: float,
) -> tuple[float, float]:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid) & (valid > 0.0)]
    if valid.size == 0:
        return lower_floor, upper_ceiling

    data_min = max(float(np.nanmin(valid)), lower_floor)
    data_max = min(float(np.nanmax(valid)), upper_ceiling)
    if data_max <= data_min:
        return max(data_min / 1.5, lower_floor), min(data_min * 1.5, upper_ceiling)

    log_min = np.log10(data_min)
    log_max = np.log10(data_max)
    pad = max(_LOG_LIMIT_PAD_MIN, _LOG_LIMIT_PAD_FRACTION * (log_max - log_min))
    return max(10.0 ** (log_min - pad), lower_floor), min(
        10.0 ** (log_max + pad),
        upper_ceiling,
    )


def _shade_loss_cone(
    axis,
    y_min: float,
    y_max: float,
    x_min: float,
    x_max: float,
) -> None:
    y_values = np.linspace(y_min, y_max, _LOSS_CONE_SAMPLE_COUNT)
    # Simple visual analogue of the loss-cone boundary in the requested plane.
    # Lower-a orbits have a wider loss-cone region.
    loss_cone_boundary = np.clip(
        10.0 ** (_LOSS_CONE_SLOPE * y_values + _LOSS_CONE_INTERCEPT),
        x_min,
        x_max,
    )
    axis.fill_betweenx(
        y_values,
        x_min,
        loss_cone_boundary,
        color=_LOSS_CONE_COLOR,
        alpha=_LOSS_CONE_ALPHA,
        linewidth=0.0,
        zorder=0,
    )


def _plot_final_points(
    axis,
    final_x: list[float],
    final_y: list[float],
    final_is_merge: list[bool],
) -> None:
    if not final_x:
        return

    final_x_arr = np.asarray(final_x, dtype=float)
    final_y_arr = np.asarray(final_y, dtype=float)
    merged = np.asarray(final_is_merge, dtype=bool)

    if np.any(merged):
        axis.scatter(
            final_x_arr[merged],
            final_y_arr[merged],
            color="red",
            s=_FINAL_MERGER_MARKER_SIZE,
            marker="o",
            alpha=0.95,
            linewidths=0.0,
            zorder=5,
        )


def _write_figure(fig, output_dir: Path, file_name: str) -> Path:
    path = output_dir / file_name
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
