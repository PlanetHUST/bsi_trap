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
    rng_orbit: np.random.Generator,
    rng_spin: np.random.Generator,
    model_config: ModelConfig,
) -> None:
    """Initialize BBH semimajor axes, eccentricities, and component spins."""
    num = bbh.num
    bbh.ecc = sample_bbh_ecc(num, rng_orbit)
    bbh.sma_cm = sample_bbh_semimajor(agn, disk, bbh, rng_orbit)
    bbh.chi1 = sample_spin_values(num, rng_spin)
    bbh.chi2 = sample_spin_values(num, rng_spin)
    bbh.th1, bbh.th2 = sample_spin_tilts(num, model_config, rng_spin)


def single_BH_params_initialize():
    pass
