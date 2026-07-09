"""Display labels and scenario names used by plot modules."""

from __future__ import annotations

from typing import Mapping

SCENARIO_DISPLAY: dict[str, str] = {
    "damp": "Damped Simulation",
    "lin": "Linear Sampling",
    "log": "Log Sampling",
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

AXIS_LABELS: dict[str, str] = {
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


def scenario_title(key: str, model_labels: Mapping[str, str] | None = None) -> str:
    """Resolve a stable display title for a model/scenario key."""
    if model_labels and key in model_labels:
        return model_labels[key]
    return SCENARIO_DISPLAY.get(key, key.replace("_", " ").title())


def axis_label(key: str) -> str:
    """Resolve a display label for a data field."""
    return AXIS_LABELS.get(key, key)
