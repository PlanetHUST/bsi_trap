"""Plotting orchestrator for combined AGN simulation HDF5 outputs.

This module does not run physics. It resolves which model outputs to plot,
loads combined HDF5 files, attaches model metadata, prepares plot payloads, and
then calls the plotting library in ``fig_plot.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from experiment_grid import load_experiment_grid
from fig_plot import (
    HDF5FigureGenerator,
    _collect_trap_band_limits,
    plot_agn_disk_parameter_profiles,
    plot_agn_disk_torque_profiles,
    plot_aini_timescale_heatmaps,
    plot_bsi_delay_timescale,
    plot_bh_trap_evolution,
    plot_chieff_mass_ratio_contours,
    plot_event_rate_comparison,
    plot_mass_radius_distribution,
    plot_mass_radius_heatmap,
    plot_sma_distribution,
    plot_sma_merger_distribution,
    plot_smbh_mass_dependence,
    plot_stop_condition_statistics,
    plot_tini_distribution_histograms,
    plot_tmerg_distribution_histograms,
    plot_timescales_histogram,
    plot_trel_tdamp_heatmap,
)
from hdf5_load import hdf5_load_method
from model_config import ModelConfig, legacy_model_config
from plot_data_utils import (
    prepare_distribution_data,
    prepare_event_rate_data,
    prepare_mass_radius_data,
    prepare_sma_data,
    prepare_smbh_dependence_data,
    prepare_stop_condition_data,
)

OUTPUT_DIR = "fig_damp"
DATA_DIR = "output"

DEFAULT_MODEL_LABELS = {
    "with_gas_damp": "Damped Simulation",
    "damp_no_copy": "Damped No Copy",
    "with_gas_damp_no_copy": "Damped No Copy",
    "with_gas_lin": "Linear Sampling",
    "with_gas_log": "Log Sampling",
    "without_gas_damp": "No Gas Damped",
    "without_gas_lin": "No Gas Linear",
    "without_gas_log": "No Gas Log",
    "BSI": "BSI",
}

DEFAULT_MODEL_COLORS = {
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


def _config_to_dict(model_config: ModelConfig | Mapping | None) -> dict:
    if model_config is None:
        return {}
    if isinstance(model_config, ModelConfig):
        return model_config.to_dict()
    if isinstance(model_config, Mapping):
        return dict(model_config)
    return {}


def _model_name(model_config: ModelConfig | Mapping) -> str:
    if isinstance(model_config, ModelConfig):
        return model_config.model_name
    return str(model_config.get("model_name", ""))


def resolve_plot_models(
    model_names: Sequence[str] | None = None,
    model_configs: Sequence[ModelConfig | Mapping] | None = None,
    experiment_grid: str | None = None,
) -> list[str]:
    """Resolve the ordered model names that fig_main should load."""
    if model_names is not None:
        return list(model_names)
    if model_configs is not None:
        return [_model_name(model) for model in model_configs if _model_name(model)]
    if experiment_grid:
        return [model.model_name for model in load_experiment_grid(experiment_grid)]
    raise ValueError(
        "fig_main requires model_names, model_configs, or experiment_grid; "
        "default plot scenarios are selected in main.py"
    )


def resolve_model_configs(
    model_names: Sequence[str],
    model_configs: Sequence[ModelConfig | Mapping] | None = None,
    experiment_grid: str | None = None,
) -> dict[str, dict]:
    """Build model_config metadata for selected plots when available."""
    if model_configs is not None:
        return {
            _model_name(model): _config_to_dict(model)
            for model in model_configs
            if _model_name(model) in model_names
        }

    if experiment_grid:
        grid_models = load_experiment_grid(experiment_grid)
        return {
            model.model_name: model.to_dict()
            for model in grid_models
            if model.model_name in model_names
        }

    resolved = {}
    for model_name in model_names:
        try:
            resolved[model_name] = legacy_model_config(model_name).to_dict()
        except ValueError:
            resolved[model_name] = {}
    return resolved


def load_simulation_sets(
    loader: hdf5_load_method,
    model_names: Sequence[str],
    data_dir: str | Path = DATA_DIR,
) -> dict[str, dict]:
    """Load each selected model output keyed by its on-disk model name."""
    datasets = {}
    for model_name in model_names:
        print(f"Loading {model_name} data")
        datasets[model_name] = loader.load_hdf5_data(model_name, data_dir=str(data_dir))
    return datasets


def _model_config_from_loaded(data: Mapping) -> dict:
    metadata = data.get("metadata", {}) if isinstance(data, Mapping) else {}
    sample_table = (
        metadata.get("sample_table", {}) if isinstance(metadata, Mapping) else {}
    )
    config_json = sample_table.get("model_config_json")
    if config_json is None:
        return {}
    try:
        values = list(config_json)
    except TypeError:
        values = [config_json]
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str) and value:
            try:
                parsed = json.loads(value)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def attach_model_configs(
    simulation_data: dict[str, dict], model_configs: Mapping[str, Mapping]
) -> None:
    """Attach per-model config metadata to loaded simulation dictionaries."""
    for model_name, data in simulation_data.items():
        data["model_config"] = dict(
            model_configs.get(model_name) or _model_config_from_loaded(data)
        )


def _single_damped_model(model_names: Sequence[str]) -> str | None:
    for candidate in ("with_gas_damp", "damp"):
        if candidate in model_names:
            return candidate
    return model_names[0] if model_names else None


def _disk_parameter_plot_kwargs(
    model_name: str | None, model_configs: Mapping[str, Mapping]
) -> dict:
    """Use the selected run's pAGN disk settings for standalone disk figures."""
    config = model_configs.get(model_name or "", {})
    disk_model = str(config.get("disk_model", "sirko"))
    return {"disk_model": disk_model}


def _is_bsi_model(model_name: str, model_configs: Mapping[str, Mapping]) -> bool:
    config = model_configs.get(model_name, {})
    return bool(config.get("bsi_enabled", False))


def _is_damp_no_copy_model(model_name: str, config: Mapping) -> bool:
    """Return True for damped models that disable the copy expansion."""
    normalized_name = str(model_name).lower()
    return normalized_name in {"damp_no_copy", "with_gas_damp_no_copy"} or (
        config.get("gas_mode", "with_gas") == "with_gas"
        and config.get("radial_distribution") == "damp"
        and config.get("damping_copy_mode") == "none"
    )


def _is_damp_copy_model(model_name: str, config: Mapping) -> bool:
    """Return True for the damped model that uses copy expansion."""
    normalized_name = str(model_name).lower()
    if normalized_name in {"damp", "with_gas_damp"}:
        return True
    return (
        config.get("gas_mode", "with_gas") == "with_gas"
        and config.get("radial_distribution") == "damp"
        and not _is_damp_no_copy_model(model_name, config)
    )


def _radial_comparison_models(
    model_names: Sequence[str], model_configs: Mapping[str, Mapping]
) -> list[str]:
    """Return the radial/copy comparison set, excluding BSI variants."""
    non_bsi = [name for name in model_names if not _is_bsi_model(name, model_configs)]
    if not non_bsi:
        return list(model_names)

    preferred = []
    seen_radial_modes = set()
    radial_order = ("damp", "linear", "log")
    for radial_mode in radial_order:
        for model_name in non_bsi:
            config = model_configs.get(model_name, {})
            if config.get("radial_distribution") != radial_mode:
                continue
            if config.get("gas_mode", "with_gas") != "with_gas":
                continue
            if radial_mode in seen_radial_modes:
                continue
            preferred.append(model_name)
            seen_radial_modes.add(radial_mode)
            break
    no_copy_models = []
    for model_name in non_bsi:
        config = model_configs.get(model_name, {})
        if _is_damp_no_copy_model(model_name, config) and model_name not in preferred:
            no_copy_models.append(model_name)
    if no_copy_models:
        damp_index = next(
            (
                i
                for i, name in enumerate(preferred)
                if model_configs.get(name, {}).get("radial_distribution") == "damp"
            ),
            -1,
        )
        insert_at = damp_index + 1 if damp_index >= 0 else len(preferred)
        preferred[insert_at:insert_at] = no_copy_models
    return preferred if preferred else non_bsi


def _damp_copy_comparison_models(
    model_names: Sequence[str], model_configs: Mapping[str, Mapping]
) -> list[str]:
    damp_copy = None
    damp_no_copy = None
    for model_name in model_names:
        config = model_configs.get(model_name, {})
        if damp_copy is None and _is_damp_copy_model(model_name, config):
            damp_copy = model_name
        if damp_no_copy is None and _is_damp_no_copy_model(model_name, config):
            damp_no_copy = model_name
    if damp_copy is None or damp_no_copy is None:
        return []
    return [damp_copy, damp_no_copy]


def _radial_distribution_models(
    model_names: Sequence[str], model_configs: Mapping[str, Mapping]
) -> list[str]:
    return [
        model_name
        for model_name in model_names
        if not _is_damp_no_copy_model(model_name, model_configs.get(model_name, {}))
    ]


def _matching_without_bsi_model(
    bsi_model: str,
    model_names: Sequence[str],
    model_configs: Mapping[str, Mapping],
) -> str | None:
    """Find the loaded non-BSI model whose config best matches a BSI variant."""
    candidates = [
        name for name in model_names if not _is_bsi_model(name, model_configs)
    ]
    if not candidates:
        return None

    bsi_config = model_configs.get(bsi_model, {})
    exact_fields = (
        "gas_mode",
        "disk_model",
        "radial_distribution",
        "damping_enabled",
        "damping_copy_mode",
        "trap_mode",
        "fixed_trap_rg",
        "migration_model",
        "remnant_retention",
        "trap_overload_policy",
        "max_generation",
        "agn_lifetime_myr",
    )
    for candidate in candidates:
        candidate_config = model_configs.get(candidate, {})
        if all(
            candidate_config.get(field) == bsi_config.get(field)
            for field in exact_fields
        ):
            return candidate

    radial_distribution = bsi_config.get("radial_distribution")
    gas_mode = bsi_config.get("gas_mode")
    for candidate in candidates:
        candidate_config = model_configs.get(candidate, {})
        if (
            candidate_config.get("radial_distribution") == radial_distribution
            and candidate_config.get("gas_mode") == gas_mode
        ):
            return candidate

    for candidate in ("with_gas_damp", "damp"):
        if candidate in candidates:
            return candidate
    return candidates[0]


def _bsi_comparison_pairs(
    model_names: Sequence[str], model_configs: Mapping[str, Mapping]
) -> list[tuple[str, str]]:
    pairs = []
    for model_name in model_names:
        if not _is_bsi_model(model_name, model_configs):
            continue
        baseline = _matching_without_bsi_model(model_name, model_names, model_configs)
        if baseline is not None:
            pairs.append((baseline, model_name))
    return pairs


def _safe_suffix(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in str(value)]
    return "_".join("".join(chars).split("_")).strip("_")


def _bsi_suffix(bsi_model: str) -> str:
    suffix = _safe_suffix(bsi_model)
    return "bsi_comparison" if suffix == "bsi" else f"bsi_comparison_{suffix}"


def _bsi_comparison_labels(
    model_labels: Mapping[str, str], baseline: str, bsi_model: str
) -> dict[str, str]:
    labels = dict(model_labels)
    baseline_label = labels.get(baseline, baseline.replace("_", " ").title())
    bsi_label = labels.get(bsi_model, bsi_model.replace("_", " ").title())
    labels[baseline] = f"{baseline_label} (no BSI)"
    labels[bsi_model] = f"{bsi_label} (with BSI)"
    return labels


def main(
    model_names: Sequence[str] | None = None,
    model_configs: Sequence[ModelConfig | Mapping] | None = None,
    experiment_grid: str | None = None,
    output_dir: str | Path = OUTPUT_DIR,
    data_dir: str | Path = DATA_DIR,
) -> None:
    """Generate standard figures from existing combined HDF5 files."""
    selected_models = resolve_plot_models(
        model_names=model_names,
        model_configs=model_configs,
        experiment_grid=experiment_grid,
    )
    resolved_configs = resolve_model_configs(
        selected_models,
        model_configs=model_configs,
        experiment_grid=experiment_grid,
    )
    model_labels = {
        name: DEFAULT_MODEL_LABELS.get(name, name.replace("_", " ").title())
        for name in selected_models
    }
    model_colors = {
        name: DEFAULT_MODEL_COLORS[name]
        for name in selected_models
        if name in DEFAULT_MODEL_COLORS
    }

    loader = hdf5_load_method()
    simulation_data = load_simulation_sets(loader, selected_models, data_dir=data_dir)
    attach_model_configs(simulation_data, resolved_configs)
    effective_configs = {
        name: dict(simulation_data.get(name, {}).get("model_config", {}))
        for name in selected_models
    }
    radial_models = _radial_comparison_models(selected_models, effective_configs)
    damp_copy_models = _damp_copy_comparison_models(radial_models, effective_configs)
    radial_distribution_models = _radial_distribution_models(
        radial_models, effective_configs
    )
    bsi_pairs = _bsi_comparison_pairs(selected_models, effective_configs)

    output_root = Path(output_dir)
    radial_output_dir = output_root / "radial_distribution"
    merger_output_dir = output_root / "merger_outcome"
    general_output_dir = output_root / "general"

    # Radial-distribution figures compare the damp/linear/log radial configs.
    radial_sma_data = prepare_sma_data(simulation_data, radial_output_dir)
    radial_trap_band = _collect_trap_band_limits(
        *(radial_sma_data.get(model_name, {}) for model_name in radial_models)
    )
    plot_sma_distribution(
        radial_sma_data,
        radial_trap_band,
        scenarios=radial_models,
        model_labels=model_labels,
        model_colors=model_colors,
    )

    mass_radius_data = prepare_mass_radius_data(simulation_data, radial_output_dir)
    plot_mass_radius_distribution(
        mass_radius_data,
        radial_trap_band,
        scenarios=radial_models,
        model_labels=model_labels,
        model_colors=model_colors,
    )
    heatmap_groups = []
    if damp_copy_models:
        heatmap_groups.append(("copy_comparison", damp_copy_models))
        if radial_distribution_models:
            heatmap_groups.append(("radial_distribution", radial_distribution_models))
    else:
        heatmap_groups.append((None, radial_models))

    for suffix, scenarios in heatmap_groups:
        trap_band = _collect_trap_band_limits(
            *(radial_sma_data.get(model_name, {}) for model_name in scenarios)
        )
        plot_mass_radius_heatmap(
            mass_radius_data,
            trap_band,
            scenarios=scenarios,
            model_labels=model_labels,
            model_colors=model_colors,
            filename_suffix=suffix,
        )
        plot_aini_timescale_heatmaps(
            simulation_data,
            output_dir=str(radial_output_dir),
            scenarios=scenarios,
            model_labels=model_labels,
            filename_suffix=suffix,
        )
    smbh_dependence_data = prepare_smbh_dependence_data(
        simulation_data, radial_output_dir
    )
    for suffix, scenarios in heatmap_groups:
        plot_smbh_mass_dependence(
            smbh_dependence_data,
            output_dir=str(radial_output_dir),
            scenarios=scenarios,
            model_labels=model_labels,
            model_colors=model_colors,
            filename_suffix=suffix,
        )
    plot_bh_trap_evolution(
        simulation_data,
        output_dir=str(radial_output_dir),
        scenarios=radial_models,
        model_labels=model_labels,
        model_colors=model_colors,
        save_generation_detail=False,
    )

    # Merger-outcome figures only compare each BSI-enabled model to its matched
    # no-BSI baseline, so radial-config trends do not get mixed into them.
    figure_generator = HDF5FigureGenerator(
        output_dir=str(merger_output_dir),
        model_configs=effective_configs,
        model_labels=model_labels,
        model_colors=model_colors,
    )
    distribution_data = prepare_distribution_data(simulation_data)
    merger_sma_data = prepare_sma_data(simulation_data, merger_output_dir)
    event_rate_data = prepare_event_rate_data(simulation_data, merger_output_dir)

    if len(radial_distribution_models) >= 2:
        figure_generator.create_distribution_plots(
            distribution_data,
            scenarios=radial_distribution_models,
            model_labels=model_labels,
            filename_suffix="radial_distribution",
        )
        trap_band = _collect_trap_band_limits(
            *(
                merger_sma_data.get(model_name, {})
                for model_name in radial_distribution_models
            )
        )
        plot_sma_merger_distribution(
            merger_sma_data,
            trap_band,
            scenarios=radial_distribution_models,
            model_labels=model_labels,
            model_colors=model_colors,
            filename_suffix="radial_distribution",
        )
    if len(damp_copy_models) >= 2:
        figure_generator.create_distribution_plots(
            distribution_data,
            scenarios=damp_copy_models,
            model_labels=model_labels,
            filename_suffix="copy_comparison",
        )
        trap_band = _collect_trap_band_limits(
            *(merger_sma_data.get(model_name, {}) for model_name in damp_copy_models)
        )
        plot_sma_merger_distribution(
            merger_sma_data,
            trap_band,
            scenarios=damp_copy_models,
            model_labels=model_labels,
            model_colors=model_colors,
            filename_suffix="copy_comparison",
        )

    for baseline, bsi_model in bsi_pairs:
        pair_suffix = None if len(bsi_pairs) == 1 else _bsi_suffix(bsi_model)
        figure_generator.create_distribution_plots(
            distribution_data,
            scenarios=[baseline, bsi_model],
            model_labels=_bsi_comparison_labels(model_labels, baseline, bsi_model),
            filename_suffix=pair_suffix,
        )
        pair = [baseline, bsi_model]
        pair_trap_band = _collect_trap_band_limits(
            *(merger_sma_data.get(model_name, {}) for model_name in pair)
        )
        plot_sma_merger_distribution(
            merger_sma_data,
            pair_trap_band,
            scenarios=pair,
            model_labels=_bsi_comparison_labels(model_labels, baseline, bsi_model),
            model_colors=model_colors,
            filename_suffix=pair_suffix,
        )
        plot_chieff_mass_ratio_contours(
            distribution_data,
            output_dir=str(merger_output_dir),
            scenarios=[baseline, bsi_model],
            model_labels=_bsi_comparison_labels(model_labels, baseline, bsi_model),
            model_colors=model_colors,
            filename_suffix=pair_suffix,
        )
        plot_bsi_delay_timescale(
            simulation_data.get(bsi_model, {}).get("first_gen", {}),
            output_dir=str(merger_output_dir),
            filename_suffix=pair_suffix,
        )

    plot_event_rate_comparison(
        event_rate_data,
        output_dir=str(merger_output_dir),
        scenarios=selected_models,
        model_labels=model_labels,
        model_colors=model_colors,
    )

    # General diagnostics are not part of either comparison family.
    damped_model = _single_damped_model(radial_models)
    if damped_model in simulation_data:
        first_gen = simulation_data[damped_model].get("first_gen", {})
        plot_trel_tdamp_heatmap(first_gen, str(general_output_dir))
        plot_timescales_histogram(first_gen, str(general_output_dir))
    plot_tmerg_distribution_histograms(
        event_rate_data,
        output_dir=str(general_output_dir),
        scenarios=selected_models,
        model_labels=model_labels,
        model_colors=model_colors,
    )
    plot_tini_distribution_histograms(
        event_rate_data,
        output_dir=str(general_output_dir),
        scenarios=selected_models,
        model_labels=model_labels,
        model_colors=model_colors,
    )
    plot_agn_disk_parameter_profiles(
        output_dir=str(general_output_dir),
        **_disk_parameter_plot_kwargs(damped_model, effective_configs),
    )
    plot_agn_disk_torque_profiles(
        output_dir=str(general_output_dir),
        **_disk_parameter_plot_kwargs(damped_model, effective_configs),
    )

    stop_condition_data = prepare_stop_condition_data(
        simulation_data, general_output_dir
    )
    plot_stop_condition_statistics(
        stop_condition_data,
        output_dir=str(general_output_dir),
        scenarios=radial_models,
        model_labels=model_labels,
        model_colors=model_colors,
    )


if __name__ == "__main__":
    from main import DEFAULT_TYPES

    main(model_names=DEFAULT_TYPES)
