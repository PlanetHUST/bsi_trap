"""Text summaries for the compact demo."""

from __future__ import annotations

import numpy as np

from bsi_trap.simulation import DemoResult
from bsi_trap.units import MYR, PC


def _myr(seconds: float | np.ndarray) -> float | np.ndarray:
    return seconds / MYR


def _pc(cm: float | np.ndarray) -> float | np.ndarray:
    return cm / PC


def format_summary(result: DemoResult, *, preview: int = 5) -> str:
    config = result.config
    disk = result.disk
    arrivals = result.arrivals
    pairing = result.pairing
    evolution = result.evolution

    lines = [
        "BSI trap demo",
        f"  seed: {config.seed}",
        f"  SMBH mass [Msun]: {config.smbh_mass_msun:.6g}",
        f"  AGN lifetime [Myr]: {config.agn_lifetime_myr:.6g}",
        f"  disk model: {config.disk_model}",
        f"  disk cells: {disk.radius_cm.size}",
        f"  trap radius [pc]: {_pc(disk.r_trap_cm):.6e}",
        f"  trap Q: {disk.trap_q:.6g}",
        f"  trap H/R: {disk.h_over_r[disk.trap_index]:.6g}",
        f"  sampled embedded BHs: {result.population.size}",
        f"  reached trap: {arrivals.reached_count}",
        f"  BBHs formed: {pairing.size}",
        f"  BBHs merged: {evolution.merger_count}",
        f"  BSI events: {evolution.bsi_count}",
        f"  remaining trap singles: {result.single_count}",
    ]

    shown_arrivals = min(max(preview, 0), arrivals.order.size)
    if shown_arrivals:
        lines.append(f"  first {shown_arrivals} arrivals:")
        for index in arrivals.order[:shown_arrivals]:
            lines.append(
                "    "
                f"idx={int(index):04d} "
                f"t={float(_myr(arrivals.arrival_s[index])):.4f} Myr "
                f"m={result.population.mass_msun[index]:.3f} Msun "
                f"R0={float(_pc(result.population.initial_radius_cm[index])):.3e} pc"
            )

    shown_pairs = min(max(preview, 0), pairing.size)
    if shown_pairs:
        lines.append(f"  first {shown_pairs} BBH formations:")
        for row in range(shown_pairs):
            first = int(pairing.first_idx[row])
            second = int(pairing.second_idx[row])
            lines.append(
                "    "
                f"({first:04d},{second:04d}) "
                f"t_form={float(_myr(pairing.formation_s[row])):.4f} Myr "
                f"delay={float(_myr(pairing.pair_delay_s[row])):.4e} Myr "
                f"m=({result.population.mass_msun[first]:.3f},"
                f"{result.population.mass_msun[second]:.3f}) Msun"
            )

    shown_mergers = min(max(preview, 0), evolution.merger_count)
    if shown_mergers:
        rows = np.flatnonzero(evolution.merged)[:shown_mergers]
        lines.append(f"  first {shown_mergers} BBH mergers:")
        for row in rows:
            row = int(row)
            lines.append(
                "    "
                f"bbh={row:04d} "
                f"t_merge={float(_myr(evolution.merger_s[row])):.4f} Myr "
                f"delay={float(_myr(evolution.delay_s[row])):.4e} Myr "
                f"channel={evolution.channel[row]} "
                f"Mf={evolution.final_mass_msun[row]:.3f} Msun "
                f"chif={evolution.final_spin[row]:.3f}"
            )

    return "\n".join(lines)
