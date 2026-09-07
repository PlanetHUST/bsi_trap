"""Human-readable and strict-JSON translations of simulation results."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from bsi_trap.simulation.models import DemoResult
from bsi_trap.units import MYR_S, PC_CM


def _myr(seconds: float) -> float:
    return seconds / MYR_S


def _pc(radius_cm: float) -> float:
    return radius_cm / PC_CM


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def format_summary(result: DemoResult, *, preview: int = 5) -> str:
    """Return a compact summary without recomputing scientific quantities."""
    disk = result.disk
    arrivals = result.arrivals
    state = result.state
    lines = [
        "BSI-trap simulation",
        f"  seed: {result.seed}",
        f"  SMBH mass [Msun]: {result.config.smbh_mass_msun:.6g}",
        f"  AGN lifetime [Myr]: {result.config.agn_lifetime_myr:.6g}",
        f"  disk model: {disk.model_name}",
        f"  disk cells: {disk.radius_cm.size}",
        f"  disk outer Q boundary [pc]: {_pc(disk.outer_radius_cm):.6e}",
        f"  sampled embedded BHs: {result.population.size}",
        f"  reached stable trap: {arrivals.reached_count}",
        f"  binaries formed: {len(state.binaries)}",
        f"  binaries merged: {len(state.merger_records)}",
        f"  BSI events: {len(state.bsi_records)}",
        f"  remaining active singles: {result.single_count}",
    ]
    shown_arrivals = min(max(preview, 0), result.population.size)
    if shown_arrivals:
        lines.append(f"  first {shown_arrivals} migration outcomes by BH ID:")
        for bh_id in result.population.bh_id[:shown_arrivals]:
            index = int(bh_id)
            arrival = arrivals.arrival_time_s[index]
            arrival_text = f"{_myr(float(arrival)):.4e}" if np.isfinite(arrival) else "unreached"
            lines.append(
                "    "
                f"bh_id={index:04d} "
                f"m={result.population.mass_msun[index]:.3f} Msun "
                f"arrival={arrival_text} Myr "
                f"stop={arrivals.stop_reason[index]}"
            )
    shown_binaries = min(max(preview, 0), len(state.binaries))
    if shown_binaries:
        lines.append(f"  first {shown_binaries} binaries:")
        for binary in state.binaries[:shown_binaries]:
            lines.append(
                "    "
                f"binary_id={binary.binary_id:04d} "
                f"components=({binary.bh1_id:04d},{binary.bh2_id:04d}) "
                f"formation={_myr(binary.formation_time_s):.4e} Myr "
                f"status={binary.status}"
            )
    return "\n".join(lines)


def result_to_dict(result: DemoResult) -> dict[str, Any]:
    """Translate a result to JSON-safe primitives with explicit unavailable values."""
    arrivals = result.arrivals
    migration_outcomes = []
    for bh_id in result.population.bh_id:
        index = int(bh_id)
        migration_outcomes.append(
            {
                "bh_id": index,
                "mass_msun": float(result.population.mass_msun[index]),
                "initial_radius_cm": float(result.population.initial_radius_cm[index]),
                "final_radius_cm": float(arrivals.final_radius_cm[index]),
                "arrival_time_s": _finite_or_none(arrivals.arrival_time_s[index]),
                "migration_elapsed_time_s": float(
                    arrivals.migration_elapsed_time_s[index]
                ),
                "trap_radius_cm": _finite_or_none(arrivals.trap_radius_cm[index]),
                "reached_stable_trap": bool(arrivals.reached_mask[index]),
                "stop_reason": str(arrivals.stop_reason[index]),
            }
        )
    return {
        "configuration": asdict(result.config),
        "seed": result.seed,
        "disk": {
            "model_name": result.disk.model_name,
            "cell_count": int(result.disk.radius_cm.size),
            "inner_radius_cm": result.disk.inner_radius_cm,
            "outer_radius_cm": result.disk.outer_radius_cm,
            "gravitational_radius_cm": result.disk.gravitational_radius_cm,
            "schwarzschild_radius_cm": result.disk.schwarzschild_radius_cm,
        },
        "initial_population": {
            "bh_id": result.population.bh_id.tolist(),
            "mass_msun": result.population.mass_msun.tolist(),
            "initial_radius_cm": result.population.initial_radius_cm.tolist(),
        },
        "migration_outcomes": migration_outcomes,
        "arrival_order_bh_id": arrivals.order.tolist(),
        "event_history": [asdict(record) for record in result.state.event_history],
        "binary_records": [asdict(binary) for binary in result.state.binaries],
        "pairing_records": [asdict(record) for record in result.state.pairing_records],
        "bsi_records": [asdict(record) for record in result.state.bsi_records],
        "merger_records": [asdict(record) for record in result.state.merger_records],
        "final_state": {
            "current_time_s": result.state.current_time_s,
            "active_single_mask": result.state.active_single_mask.tolist(),
            "pairing_pending_mask": result.state.pairing_pending_mask.tolist(),
            "paired_mask": result.state.paired_mask.tolist(),
            "merged_mask": result.state.merged_mask.tolist(),
        },
    }


def write_result_json(result: DemoResult, output_path: Path) -> None:
    """Write strict JSON; unavailable migration values become ``null``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result_to_dict(result), stream, indent=2, allow_nan=False)
        stream.write("\n")


__all__ = ["format_summary", "result_to_dict", "write_result_json"]
