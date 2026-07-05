from dataclasses import dataclass

import numpy as np
from bsi_trap.classes.models import (
    BBH_properties,
    AGN_properties,
    Single_BH_properties,
    BBHEvolverParams,
    DiskProfile,
)
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.evolve.bsi import construct_bsi_timeline, schedule_bsi_events
from bsi_trap.evolve.gas_harden import evolve_gas
from bsi_trap.evolve.gw import evolve_gw_only
from bsi_trap.evolve.timeline import (
    BSI_Timeline,
    BoolArray,
    EvolutionTimeline,
    FloatArray,
    IndexArray,
    IntArray,
)
from bsi_trap.evolve.tracks import finalize_bbh_tracks
from bsi_trap.constant.stop_reason import StopReasons
from bsi_trap.evolve import numeric
from bsi_trap.evolve import physical_quantity as pq
from bsi_trap.constant import constant as const
from bsi_trap.hierarchical.remnants import (
    HierarchicalMergerLedger,
    build_hierarchical_mergers,
)


@dataclass(frozen=True)
class GenerationEvolutionResult:
    evolution_timeline: EvolutionTimeline
    hierarchical_mergers: HierarchicalMergerLedger


def construct_bbh_evolution_timeline(
    bbh: BBH_properties,
    bsi_timeline: BSI_Timeline | None = None,
) -> EvolutionTimeline:
    """Build the absolute-time BBH evolution outcome ledger.

    The BBH object should already contain the physical post-evolution state.
    This function only packages times, merger flags, stop reasons, and the BSI
    event ledger for storage/analysis.
    """
    if bsi_timeline is None:
        bsi_timeline = construct_bsi_timeline(bbh)
    if bbh.num == 0:
        return EvolutionTimeline.empty(bsi_timeline=bsi_timeline)

    binary_idx: IndexArray = np.arange(bbh.num, dtype=np.intp)
    start_s: FloatArray = np.asarray(bbh.t_ini_s, dtype=np.float64)
    delay_s: FloatArray = np.asarray(bbh.t_del_s, dtype=np.float64)
    merged: BoolArray = np.asarray(bbh.flags.merger_flag != 0, dtype=bool)
    end_s: FloatArray = np.asarray(start_s + delay_s, dtype=np.float64)
    merger_s: FloatArray = np.full(bbh.num, np.nan, dtype=np.float64)
    merger_s[merged] = end_s[merged]
    stop_reason: IntArray = np.asarray(
        bbh.flags.stop_reason_flag,
        dtype=np.int64,
    )

    return EvolutionTimeline(
        binary_idx=binary_idx,
        start_s=start_s,
        delay_s=delay_s,
        end_s=end_s,
        merger_s=merger_s,
        merged=merged,
        stop_reason=stop_reason,
        bsi_timeline=bsi_timeline,
    )


def make_generation(
    Evolparams: BBHEvolverParams,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    bbh: BBH_properties,
    sbh: Single_BH_properties,
    rng_orbit: np.random.Generator,
    rng_spin: np.random.Generator,
) -> GenerationEvolutionResult:
    """Evolve the current first-generation BBH population.

    This stage mutates ``bbh`` in place: gas/GW evolution changes orbital
    parameters, and scheduled BSI events may perturb or stop binaries.  The
    returned outcome records both the evolution timeline and the post-merger
    hierarchical-remnant ledger.
    """
    num = bbh.num
    if model_config.bsi_enabled:
        schedule_bsi_events(
            model_config=model_config,
            agn=agn,
            disk=disk,
            bbh=bbh,
            sbh=sbh,
            rng=rng_orbit,
        )

    for i in range(num):
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
    return GenerationEvolutionResult(
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
    a_cm = float(bbh.sma_cm[i])
    e = float(bbh.ecc[i])
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
        )

    elif model_config.gas_mode == "without_gas":
        t_del_s = evolve_gw_only(
            Evolparams,
            bbh,
            agn,
            i,
            t_ini_s,
            bsi_enabled=model_config.bsi_enabled,
        )

    else:
        raise ValueError(f"Invalid gas mode specified: {model_config.gas_mode}")

    if not np.isfinite(t_del_s) or t_del_s < 0.0:
        raise ValueError(f"Invalid BBH evolution delay for index {i}: {t_del_s}")

    return t_del_s
