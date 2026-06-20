import numpy as np
from bsi_trap.classes.models import (
    BBH_properties,
    AGN_properties,
    Single_BH_properties,
    BBHEvolverParams,
    DiskProfile,
)
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.evolve.gas_harden import evolve_gas
from bsi_trap.evolve.gw import evolve_gw_only


def make_generation(
    Evolparams: BBHEvolverParams,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    bbh: BBH_properties,
    sbh: Single_BH_properties,
    rng_orbit: np.random.Generator,
    rng_spin: np.random.Generator,
) -> BBH_properties:
    # main hierarchical mergers evolution workflow
    num = bbh.num

    for i in range(num):
        t_del_s = evolve_single_binary(Evolparams, model_config, agn, disk, bbh, i)


def evolve_single_binary(
    Evolparams: BBHEvolverParams,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    bbh: BBH_properties,
    i: int,
) -> tuple[float, ...]:
    """in the view of one single bbh, evolue till merge for dutycycle overpassed"""

    # get bbh flags
    flags = bbh.flags
    t_ini_s = float(bbh.t_ini_s[i])

    if model_config.gas_mode == "with_gas":
        bbh, t_del_s, _, flags = evolve_gas(
            Evolparams, disk, bbh, agn, i, flags, t_ini_s
        )
    elif model_config.gas_mode == "without_gas":
        bbh, t_del_s, flags = evolve_gw_only(Evolparams, bbh, agn, i, t_ini_s)
    else:
        raise ValueError(f"Invalid gas mode specified: {model_config.gas_mode}")

    return bbh, t_del_s, flags
