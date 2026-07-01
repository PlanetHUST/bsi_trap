import numpy as np
from bsi_trap.classes.models import (
    BBH_properties,
    AGN_properties,
    Single_BH_properties,
    BBHEvolverParams,
    DiskProfile,
)
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.evolve.bsi import construct_bsi_timeline
from bsi_trap.evolve.gas_harden import evolve_gas
from bsi_trap.evolve.gw import evolve_gw_only
from bsi_trap.evolve.timeline import BbhEvolutionTimeline, BsiTimeline


def construct_bbh_evolution_timeline(
    bbh: BBH_properties,
    bsi_timeline: BsiTimeline | None = None,
) -> BbhEvolutionTimeline:
    """Build the absolute-time BBH evolution outcome timeline."""
    if bsi_timeline is None:
        bsi_timeline = construct_bsi_timeline(bbh)
    if bbh.num == 0:
        return BbhEvolutionTimeline.empty(bsi_timeline=bsi_timeline)

    binary_idx = np.arange(bbh.num, dtype=np.intp)
    start_s = np.asarray(bbh.t_ini_s, dtype=float)
    delay_s = np.asarray(bbh.t_del_s, dtype=float)
    merged = np.asarray(bbh.flags.merger_flag != 0, dtype=bool)
    end_s = start_s + delay_s
    merger_s = np.full(bbh.num, np.nan, dtype=float)
    merger_s[merged] = end_s[merged]

    return BbhEvolutionTimeline(
        binary_idx=binary_idx,
        start_s=start_s,
        delay_s=delay_s,
        end_s=end_s,
        merger_s=merger_s,
        merged=merged,
        stop_reason=np.asarray(bbh.flags.stop_reason_flag, dtype=np.int64),
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
) -> BbhEvolutionTimeline:
    # main hierarchical mergers evolution workflow
    num = bbh.num

    for i in range(num):
        t_del_s = evolve_single_binary(Evolparams, model_config, agn, disk, bbh, i)
        bbh.t_del_s[i] = t_del_s

    return construct_bbh_evolution_timeline(bbh)


def evolve_single_binary(
    Evolparams: BBHEvolverParams,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    bbh: BBH_properties,
    i: int,
) -> float:
    """in the view of one single bbh, evolue till merge for dutycycle overpassed"""

    # get bbh flags
    t_ini_s = float(bbh.t_ini_s[i])
    bbh.flags.trap_reached_flag[i] = 1

    if model_config.gas_mode == "with_gas":
        t_del_s = evolve_gas(Evolparams, disk, bbh, agn, i)

    elif model_config.gas_mode == "without_gas":
        t_del_s = evolve_gw_only(Evolparams, bbh, agn, i, t_ini_s)

    else:
        raise ValueError(f"Invalid gas mode specified: {model_config.gas_mode}")

    if not np.isfinite(t_del_s) or t_del_s < 0.0:
        raise ValueError(f"Invalid BBH evolution delay for index {i}: {t_del_s}")

    return t_del_s
