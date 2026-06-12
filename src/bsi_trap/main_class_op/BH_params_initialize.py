import numpy as np
from bsi_trap.classes.classes import (
    BBH_properties,
    Single_BH_properties,
    AGN_properties,
    DiskProfile,
)
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.functions.random_sample import (
    sample_bbh_ecc,
    sample_bbh_semimajor,
    sample_spin_values,
    sample_spin_tilts,
)


def BBH_params_initialize(
    agn: AGN_properties,
    disk: DiskProfile,
    bbh: BBH_properties,
    rng: np.random.Generator,
    model_config: ModelConfig,
) -> None:
    """random initialize {a,e,chi} for bbh"""
    num = bbh.num
    bbh.ecc = sample_bbh_ecc(num, rng)
    bbh.sma_cm = sample_bbh_semimajor(agn, disk, bbh)
    bbh.chi1 = sample_spin_values(rng)
    bbh.chi2 = sample_spin_values(rng)
    bbh.th1 = sample_spin_tilts(bbh, model_config, rng)
    bbh.th2 = sample_spin_tilts(bbh, model_config, rng)


def single_BH_params_initialize():
    pass
