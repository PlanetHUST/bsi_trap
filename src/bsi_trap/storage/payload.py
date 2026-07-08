from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from bsi_trap.simulation.state.models import AGN_properties, BBH_properties, DiskProfile
from bsi_trap.simulation.config.model_config import ModelConfig
from bsi_trap.constant import constant as const
from bsi_trap.simulation.timeline.models import EvolutionTimeline
from bsi_trap.simulation.runtime.remnants import HierarchicalMergerLedger
from bsi_trap.simulation.initial_conditions.trap_population import TrapPopulation

from .contract import HDF5Contract


@dataclass(frozen=True)
class GenerationState:
    agn: AGN_properties
    disk: DiskProfile
    population: TrapPopulation
    evolution_timeline: EvolutionTimeline
    hierarchical_mergers: HierarchicalMergerLedger | None = None
    generation_index: int = 1


def gravitational_radius_cm(agn: AGN_properties) -> float:
    return float(const.G * agn.M_SMBH_msun * const.msun / const.c**2)


def _as_myr(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) / const.Myr


def _nan(size: int) -> np.ndarray:
    return np.full(size, np.nan, dtype=float)


def agn_data(agn: AGN_properties, disk: DiskProfile) -> dict[str, float]:
    return {
        "smbh_mass_msun": float(agn.M_SMBH_msun),
        "M_SMBH_msun": float(agn.M_SMBH_msun),
        "lifetime_myr": float(agn.lifetime_myr),
        "alpha": float(agn.alpha),
        "rg": gravitational_radius_cm(agn),
        "r_in_cm": float(disk.r_in_cm),
        "r_out_cm": float(disk.r_out_cm),
        "r_trap_cm": float(disk.r_trap_cm),
        "r_antitrap_cm": float(disk.r_antitrap_cm),
    }


def bbh_data(
    bbh: BBH_properties,
    evolution_timeline: EvolutionTimeline,
) -> dict[str, np.ndarray]:
    return {
        "m1": bbh.m1_msun,
        "m2": bbh.m2_msun,
        "chi1": bbh.chi1,
        "chi2": bbh.chi2,
        "A": bbh.A_cm,
        "Aini": bbh.A_cm,
        "sma_initial": bbh.sma_initial_cm,
        "ecc_initial": bbh.ecc_initial,
        "sma": bbh.sma_cm,
        "sma_final": bbh.sma_cm,
        "ecc": bbh.ecc,
        "ecc_final": bbh.ecc,
        "ecc10": bbh.ecc10,
        "ecc0p1": bbh.ecc0p1,
        "th1": bbh.th1,
        "th2": bbh.th2,
        "idi": bbh.idi,
        "mmerg": bbh.mmerg,
        "mrem": bbh.mrem_msun,
        "chimerg": bbh.chimerg,
        "chirem": bbh.chirem,
        "vk_cgs": bbh.vk_cgs,
        "generation": bbh.generation,
        "trapreach_time_s": bbh.t_ini_s,
        "bsi_timescale_s": bbh.bsi_timescale_s,
        "bsi_weak_count": bbh.bsi_weak_count,
        "bsi_delta_e_total": bbh.bsi_delta_e_total,
        "bsi_delta_theta_total": bbh.bsi_delta_theta_total,
        "bsi_perturbed_flag": bbh.flags.bsi_event_flag,
        "evolution_start_s": evolution_timeline.start_s,
        "evolution_end_s": evolution_timeline.end_s,
    }


def flag_data(bbh: BBH_properties) -> dict[str, np.ndarray]:
    flags = bbh.flags
    return {
        "merger_flag": flags.merger_flag,
        "GWescape_flag": flags.GWescape_flag,
        "continue_ng_flag": flags.continue_ng_flag,
        "hard": flags.hard,
        "bsi_event_flag": flags.bsi_event_flag,
        "trap_reached_flag": flags.trap_reached_flag,
        "stop_reason_flag": flags.stop_reason_flag,
    }


def time_data(
    bbh: BBH_properties,
    evolution_timeline: EvolutionTimeline,
) -> dict[str, np.ndarray]:
    num = bbh.num
    bsi_event_time_s = np.full(num, np.nan, dtype=float)
    bsi_delay_s = np.asarray(bbh.bsi_timescale_s, dtype=float)
    bsi_applied = (
        np.asarray(bbh.flags.bsi_event_flag, dtype=int) != 0
    ) & np.isfinite(bsi_delay_s) & (bsi_delay_s >= 0.0)
    bsi_event_time_s[bsi_applied] = bbh.t_ini_s[bsi_applied] + bsi_delay_s[bsi_applied]
    return {
        "tform_myr": _nan(num),
        "trel_myr": _nan(num),
        "tdamp_myr": _nan(num),
        "tmig_myr": _nan(num),
        "tini_myr": _as_myr(bbh.t_ini_s),
        "tdel_myr": _as_myr(bbh.t_del_s),
        "tmerg_myr": _as_myr(evolution_timeline.merger_s),
        "t_bsi_myr": _as_myr(bsi_event_time_s),
    }


def bbh_track_data(bbh: BBH_properties) -> dict[str, np.ndarray]:
    return {
        "binary_idx": bbh.track_binary_idx,
        "step_idx": bbh.track_step_idx,
        "time_s": bbh.track_time_s,
        "sma_cm": bbh.track_sma_cm,
        "ecc": bbh.track_ecc,
    }


def trap_population_data(
    population: TrapPopulation,
) -> dict[str, dict[str, Any]]:
    ebh = population.embedded_bh
    pairing = population.pairing_timeline
    singles = population.singles
    return {
        "embedded_bh": {
            "m_msun": ebh.m_msun,
            "chi": ebh.chi,
            "theta": ebh.theta,
            "t_form_s": ebh.t_form_s,
            "t_mig_s": ebh.t_mig_s,
            "t_reach_s": ebh.t_reach_s,
            "t_pair_s": ebh.t_pair_s,
            "R_cm": ebh.R_cm,
            "r_trap_target_cm": ebh.r_trap_target_cm,
            "r_pair_cm": ebh.r_pair_cm,
            "trap_flag": ebh.trap_flag,
            "trap_width_cm": float(ebh.trap_width_cm),
        },
        "pairing_timeline": {
            "first_idx": pairing.first_idx,
            "second_idx": pairing.second_idx,
            "later_arrival_s": pairing.later_arrival_s,
            "encounter_s": pairing.encounter_s,
            "pair_delay_s": pairing.pair_delay_s,
            "formation_s": pairing.formation_s,
        },
        "singles": {
            "m_msun": singles.m_msun,
            "chi": singles.chi,
            "theta": singles.theta,
            "t_ini_s": singles.t_ini_s,
            "generation": singles.generation,
        },
    }


def hierarchical_merger_data(
    ledger: HierarchicalMergerLedger | None,
) -> dict[str, Any]:
    if ledger is None:
        ledger = HierarchicalMergerLedger.empty()

    singles = ledger.next_generation_singles
    return {
        "source_binary_idx": ledger.source_binary_idx,
        "merger_s": ledger.merger_s,
        "remnant_mass_msun": ledger.remnant_mass_msun,
        "remnant_spin": ledger.remnant_spin,
        "kick_velocity_cgs": ledger.kick_velocity_cgs,
        "outer_semimajor_cm": ledger.outer_semimajor_cm,
        "retained_flag": ledger.retained.astype(int),
        "escaped_flag": ledger.escaped.astype(int),
        "trap_overloaded_flag": ledger.trap_overloaded.astype(int),
        "next_generation_singles": {
            "m_msun": singles.m_msun,
            "t_ini_s": singles.t_ini_s,
            "generation": singles.generation,
        },
    }


def generation_data(
    *,
    model_config: ModelConfig,
    state: GenerationState,
    contract: HDF5Contract = HDF5Contract(),
) -> dict[str, object]:
    bbh = state.population.binaries
    return {
        contract.key_agn: agn_data(state.agn, state.disk),
        contract.key_bbh: bbh_data(bbh, state.evolution_timeline),
        contract.key_flags: flag_data(bbh),
        contract.key_timescales: time_data(bbh, state.evolution_timeline),
        contract.key_bbh_tracks: bbh_track_data(bbh),
        contract.key_trap_population: trap_population_data(state.population),
        contract.key_hierarchical_mergers: hierarchical_merger_data(
            state.hierarchical_mergers,
        ),
        contract.key_model_config: model_config,
        contract.key_generation_type: (
            "first_generation"
            if state.generation_index == 1
            else "hierarchical_generation"
        ),
        contract.key_process_id: os.getpid(),
    }


def sample_manifest(
    *,
    model_config: ModelConfig,
    state: GenerationState,
    sample_index: int,
    smbh_mass: float,
    output_file: Path,
    merger_count: int,
    evolved_binary_count: int,
    generation_count: int,
    generation_files: list[Path] | tuple[Path, ...] | None = None,
) -> dict[str, object]:
    population = state.population
    timeline = state.evolution_timeline
    return {
        "sample_index": int(sample_index),
        "simulation_type": model_config.simulation_type,
        "smbh_mass_msun": float(smbh_mass),
        "generation_count": int(generation_count),
        "merger_count": int(merger_count),
        "evolved_binary_count": int(evolved_binary_count),
        "output_file": str(output_file),
        "generation_files": [str(path) for path in generation_files]
        if generation_files is not None
        else [str(output_file)],
        "model_config": model_config.to_dict(),
        "process_id": os.getpid(),
        "completion_time": time.time(),
        "status": "completed",
        "agn": agn_data(state.agn, state.disk),
        "trap_population": {
            "embedded_bh_count": int(population.embedded_bh.num),
            "binary_count": int(population.binaries.num),
            "single_count": int(population.singles.num),
            "pairing_event_count": int(population.pairing_timeline.size),
        },
        "binary_evolution": {
            "evolved_binary_count": int(timeline.size),
            "merger_count": int(merger_count),
            "nonmerger_count": int(timeline.size - merger_count),
            "mean_delay_myr": float(np.nanmean(timeline.delay_s) / const.Myr)
            if timeline.size > 0
            else float("nan"),
        },
        "hierarchical_mergers": {
            "remnant_count": int(state.hierarchical_mergers.size)
            if state.hierarchical_mergers is not None
            else 0,
            "retained_count": int(state.hierarchical_mergers.retained_count)
            if state.hierarchical_mergers is not None
            else 0,
            "escaped_count": int(state.hierarchical_mergers.escaped_count)
            if state.hierarchical_mergers is not None
            else 0,
            "trap_overloaded_count": int(
                state.hierarchical_mergers.trap_overloaded_count
            )
            if state.hierarchical_mergers is not None
            else 0,
            "recycled_count": int(state.hierarchical_mergers.recycled_count)
            if state.hierarchical_mergers is not None
            else 0,
        },
    }


def write_sample_manifest(
    *,
    model_config: ModelConfig,
    state: GenerationState,
    sample_index: int,
    smbh_mass: float,
    output_file: Path,
    merger_count: int,
    evolved_binary_count: int,
    generation_count: int,
    generation_files: list[Path] | tuple[Path, ...] | None = None,
) -> Path:
    manifest_path = output_file.parent / f"manifest_sample_{sample_index:04d}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(
            sample_manifest(
                model_config=model_config,
                state=state,
                sample_index=sample_index,
                smbh_mass=smbh_mass,
                output_file=output_file,
                merger_count=merger_count,
                evolved_binary_count=evolved_binary_count,
                generation_count=generation_count,
                generation_files=generation_files,
            ),
            stream,
            indent=2,
            default=str,
        )
    os.replace(temp_path, manifest_path)
    return manifest_path
