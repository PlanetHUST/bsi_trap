from dataclasses import dataclass

import numpy as np
from bsi_trap.simulation.state.models import (
    BBH_properties,
    AGN_properties,
    BBHEvolverParams,
    DiskProfile,
)
from bsi_trap.simulation.config.model_config import ModelConfig
from bsi_trap.physics.binary.gas_hardening import evolve_gas
from bsi_trap.physics.binary.gw import evolve_gw_only
from bsi_trap.simulation.timeline.bsi import construct_bsi_timeline
from bsi_trap.simulation.timeline.evolution import construct_bbh_evolution_timeline
from bsi_trap.simulation.timeline.models import EvolutionTimeline
from bsi_trap.physics.binary.tracks import finalize_bbh_tracks
from bsi_trap.constant.stop_reason import StopReasons
from bsi_trap.physics.binary import numeric
from bsi_trap.physics.binary import quantities as pq
from bsi_trap.constant import constant as const
from bsi_trap.simulation.runtime.remnants import (
    HierarchicalMergerLedger,
    build_hierarchical_mergers,
)


@dataclass(frozen=True)
class EvolutionResult:
    evolution_timeline: EvolutionTimeline
    hierarchical_mergers: HierarchicalMergerLedger


def make_generation(
    Evolparams: BBHEvolverParams,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    bbh: BBH_properties,
    rng_orbit: np.random.Generator,
) -> EvolutionResult:
    """Evolve one committed BBH population and package its event ledgers.

    BSI event timing belongs to the top-level runtime clock. This helper only
    applies BSI delays that have already been written to ``bbh.bsi_timescale_s``.
    """

    for i in range(bbh.num):
        t_del_s = evolve_single_binary(Evolparams, model_config, agn, disk, bbh, i)
        bbh.t_del_s[i] = t_del_s

    finalize_bbh_tracks(bbh)

    bsi_timeline = construct_bsi_timeline(bbh)

    hierarchical_mergers = build_hierarchical_mergers(
        agn=agn,
        disk=disk,
        bbh=bbh,
        rng=rng_orbit,
    )

    evolution_timeline = construct_bbh_evolution_timeline(
        bbh,
        bsi_timeline=bsi_timeline,
    )

    return EvolutionResult(
        evolution_timeline=evolution_timeline,
        hierarchical_mergers=hierarchical_mergers,
    )


def _stop_invalid_initial_binary(bbh: BBH_properties, i: int) -> float:
    flags = bbh.flags
    flags.merger_flag[i] = 0
    flags.continue_ng_flag[i] = 0
    flags.stop_reason_flag[i] = StopReasons.STOP_NON_PHYSICAL_ORBIT
    return 0.0


def _stop_initial_binary_too_tight(bbh: BBH_properties, i: int) -> float:
    flags = bbh.flags
    flags.merger_flag[i] = 0
    flags.continue_ng_flag[i] = 0
    flags.stop_reason_flag[i] = StopReasons.STOP_INITIAL_BINARY_TOO_TIGHT
    return 0.0


def _validate_initial_binary(
    Evolparams: BBHEvolverParams,
    bbh: BBH_properties,
    i: int,
) -> int | None:
    m1_g = float(bbh.m1_msun[i] * const.msun)
    m2_g = float(bbh.m2_msun[i] * const.msun)
    a_cm = float(bbh.sma_evolve_cm[i])
    e = float(bbh.ecc_evolve[i])
    t_ini_s = float(bbh.t_ini_s[i])

    if (
        not np.isfinite(t_ini_s)
        or t_ini_s < 0.0
        or not np.isfinite(m1_g)
        or m1_g <= 0.0
        or not np.isfinite(m2_g)
        or m2_g <= 0.0
        or not numeric.is_physical_state(Evolparams, a_cm, e)
    ):
        return StopReasons.STOP_NON_PHYSICAL_ORBIT

    r_th = pq.threshold_radius(m1_g, m2_g)
    if (a_cm - r_th) / r_th <= Evolparams.merger_boundary_fraction:
        return StopReasons.STOP_INITIAL_BINARY_TOO_TIGHT

    return None


def evolve_single_binary(
    Evolparams: BBHEvolverParams,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    bbh: BBH_properties,
    i: int,
    *,
    segment_stop_s: float | None = None,
) -> float:
    """Evolve one BBH and return the delay from BBH formation to termination."""

    # get bbh flags
    t_ini_s = float(bbh.t_ini_s[i])
    bbh.flags.trap_reached_flag[i] = 1

    initial_stop = _validate_initial_binary(Evolparams, bbh, i)
    if initial_stop == StopReasons.STOP_NON_PHYSICAL_ORBIT:
        return _stop_invalid_initial_binary(bbh, i)
    if initial_stop == StopReasons.STOP_INITIAL_BINARY_TOO_TIGHT:
        return _stop_initial_binary_too_tight(bbh, i)

    if model_config.gas_mode == "with_gas":
        t_del_s = evolve_gas(
            Evolparams,
            disk,
            bbh,
            agn,
            i,
            bsi_enabled=model_config.bsi_enabled,
            segment_stop_s=segment_stop_s,
        )

    elif model_config.gas_mode == "without_gas":
        t_del_s = evolve_gw_only(
            Evolparams,
            bbh,
            agn,
            i,
            t_ini_s,
            bsi_enabled=model_config.bsi_enabled,
            segment_stop_s=segment_stop_s,
        )

    else:
        raise ValueError(f"Invalid gas mode specified: {model_config.gas_mode}")

    if not np.isfinite(t_del_s) or t_del_s < 0.0:
        raise ValueError(f"Invalid BBH evolution delay for index {i}: {t_del_s}")

    return t_del_s
