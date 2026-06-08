"""Schema-configured derived field evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

MYR_SECONDS = 1.0e6 * 365.25 * 24.0 * 3600.0


@dataclass(frozen=True)
class DerivedFieldRule:
    name: str
    formula: str


def add_derived_fields(
    payload: dict[str, np.ndarray],
    rules: tuple[DerivedFieldRule, ...] = (),
) -> None:
    """Evaluate derived fields declared in the layout.

    A rule is skipped when its formula references a missing field.
    """

    for rule in rules:
        if rule.name in payload:
            continue

        try:
            payload[rule.name] = _evaluate_formula(rule.formula, payload)
        except (NameError, KeyError):
            continue


def safe_divide(numerator: Any, denominator: Any) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator_array != 0, numerator_array / denominator_array, np.nan)


def _evaluate_formula(formula: str, payload: dict[str, np.ndarray]) -> np.ndarray:
    namespace = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "maximum": np.maximum,
        "where": np.where,
        "safe_divide": safe_divide,
        "MYR_SECONDS": MYR_SECONDS,
    }
    namespace.update(payload)

    return np.asarray(eval(formula, {"__builtins__": {}}, namespace))
