from __future__ import annotations

import numpy as np

from bsi_trap.simulation.state.models import (
    AGN_properties,
    BBH_properties,
    DiskProfile,
    Embedded_BH,
    Single_BH_properties,
)
from bsi_trap.simulation.config.model_config import ModelConfig
from bsi_trap.physics.trap.context import build_trap_context
from bsi_trap.simulation.initial_conditions.bbh_parameters import BBH_params_initialize
from bsi_trap.simulation.initial_conditions.trap_population import TrapPopulation
from bsi_trap.simulation.runtime.singles import copy_singles_by_index
from bsi_trap.simulation.timeline.models import PairingTimeline
from bsi_trap.simulation.timeline.pairing import construct_pairing_step_timeline


def build_bbh_from_pairing(
    source_singles: Single_BH_properties,
    pairing_timeline: PairingTimeline,
) -> BBH_properties:
    bbh = BBH_properties(pairing_timeline.size)
    if bbh.num == 0:
        return bbh

    first_idx = np.asarray(pairing_timeline.first_idx, dtype=np.intp)
    second_idx = np.asarray(pairing_timeline.second_idx, dtype=np.intp)
    first_mass_msun = source_singles.m_msun[first_idx]
    second_mass_msun = source_singles.m_msun[second_idx]
    first_is_primary = first_mass_msun >= second_mass_msun

    primary_idx = np.where(first_is_primary, first_idx, second_idx)
    secondary_idx = np.where(first_is_primary, second_idx, first_idx)
    bbh.idi = primary_idx
    bbh.m1_msun = np.asarray(source_singles.m_msun[primary_idx], dtype=np.float64)
    bbh.m2_msun = np.asarray(source_singles.m_msun[secondary_idx], dtype=np.float64)
    bbh.formation_s = np.asarray(pairing_timeline.formation_s, dtype=np.float64)
    bbh.generation = (
        np.maximum(
            source_singles.generation[first_idx],
            source_singles.generation[second_idx],
        )
        + 1
    ).astype(np.int64)

    primary_has_spin = np.isfinite(source_singles.chi[primary_idx]) & np.isfinite(
        source_singles.theta[primary_idx]
    )
    secondary_has_spin = np.isfinite(source_singles.chi[secondary_idx]) & np.isfinite(
        source_singles.theta[secondary_idx]
    )
    bbh.chi1[primary_has_spin] = source_singles.chi[primary_idx][primary_has_spin]
    bbh.th1[primary_has_spin] = source_singles.theta[primary_idx][primary_has_spin]
    bbh.chi2[secondary_has_spin] = source_singles.chi[secondary_idx][
        secondary_has_spin
    ]
    bbh.th2[secondary_has_spin] = source_singles.theta[secondary_idx][
        secondary_has_spin
    ]
    return bbh


def remaining_singles_after_pairing(
    source_singles: Single_BH_properties,
    pairing_timeline: PairingTimeline,
) -> Single_BH_properties:
    if pairing_timeline.size == 0:
        return copy_singles_by_index(source_singles, np.arange(source_singles.num))

    paired_mask = np.zeros(source_singles.num, dtype=bool)
    paired_mask[np.asarray(pairing_timeline.first_idx, dtype=np.intp)] = True
    paired_mask[np.asarray(pairing_timeline.second_idx, dtype=np.intp)] = True
    return copy_singles_by_index(source_singles, np.flatnonzero(~paired_mask))


def build_pair_event_population(
    *,
    embedded_bh: Embedded_BH,
    source_singles: Single_BH_properties,
    current_time_s: float,
    step_end_s: float,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    rng_pair_up: np.random.Generator,
    rng_orbit: np.random.Generator,
    rng_spin: np.random.Generator,
) -> TrapPopulation:
    trap_context = build_trap_context(disk, model_config)
    pairing_timeline = construct_pairing_step_timeline(
        singles=source_singles,
        trap_context=trap_context,
        model_config=model_config,
        disk=disk,
        smbh_mass_msun=agn.M_SMBH_msun,
        current_time_s=current_time_s,
        step_end_s=step_end_s,
        rng_pair_up=rng_pair_up,
    )

    bbh = build_bbh_from_pairing(source_singles, pairing_timeline)
    singles = remaining_singles_after_pairing(source_singles, pairing_timeline)

    population = TrapPopulation(
        embedded_bh=embedded_bh,
        binaries=bbh,
        singles=singles,
        pairing_timeline=pairing_timeline,
    )

    if bbh.num > 0:
        BBH_params_initialize(
            agn=agn,
            disk=disk,
            bbh=bbh,
            rng_orbit=rng_orbit,
            rng_spin=rng_spin,
            model_config=model_config,
            preserve_existing_spins=True,
        )

    return population
