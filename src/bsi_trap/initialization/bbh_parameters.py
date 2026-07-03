import numpy as np
from bsi_trap.classes.models import (
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
    preserve_existing_spins: bool = False,
) -> None:
    """Initialize BBH semimajor axes, eccentricities, and component spins."""
    num = bbh.num
    bbh.ecc = sample_bbh_ecc(num, rng_orbit)
    bbh.sma_cm = sample_bbh_semimajor(agn, disk, bbh, rng_orbit)
    bbh.sma_initial_cm = np.array(bbh.sma_cm, copy=True)
    bbh.ecc_initial = np.array(bbh.ecc, copy=True)

    if not preserve_existing_spins:
        bbh.chi1 = sample_spin_values(num, rng_spin)
        bbh.chi2 = sample_spin_values(num, rng_spin)
        bbh.th1, bbh.th2 = sample_spin_tilts(num, model_config, rng_spin)
        return

    missing_chi1 = ~np.isfinite(bbh.chi1)
    missing_chi2 = ~np.isfinite(bbh.chi2)
    missing_th1 = ~np.isfinite(bbh.th1)
    missing_th2 = ~np.isfinite(bbh.th2)

    if np.any(missing_chi1):
        bbh.chi1[missing_chi1] = sample_spin_values(
            int(np.count_nonzero(missing_chi1)),
            rng_spin,
        )
    if np.any(missing_chi2):
        bbh.chi2[missing_chi2] = sample_spin_values(
            int(np.count_nonzero(missing_chi2)),
            rng_spin,
        )
    if np.any(missing_th1 | missing_th2):
        th1, th2 = sample_spin_tilts(num, model_config, rng_spin)
        bbh.th1[missing_th1] = th1[missing_th1]
        bbh.th2[missing_th2] = th2[missing_th2]


# def single_BH_params_initialize():
#     pass
