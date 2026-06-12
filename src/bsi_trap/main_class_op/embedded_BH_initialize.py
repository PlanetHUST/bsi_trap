import numpy as np
from typing import Tuple

from bsi_trap.classes.classes import (
    AGN_properties,
    BBH_properties,
    DiskProfile,
    Embedded_BH,
)
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.functions.nsc_population import number_R_NSC
from bsi_trap.functions.random_sample import sample_embedded_radius


def load_filtered_stellar_population(
    fil: str = "input/",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load progenitor/remnant masses and formation times from the input catalog."""
    # similar method as fast-cluster, in order to get the tfrom and mass of stars evolution
    # need some changes for classes
    mzams, m1iii, tformii = np.genfromtxt(
        fil, usecols=(1, 2, 3), skip_header=3, unpack=True
    )

    ok = np.where((m1iii <= 30.0) & (tformii < 100.0))[0]

    return np.copy(mzams[ok]), np.copy(m1iii[ok]), np.copy(tformii[ok])


def embedded_BH_initialize(agn: AGN_properties, disk: DiskProfile) -> Embedded_BH:

    r_out_cm = disk.r_out_cm
    mave_msun = agn.mave_msun
    # 1. calculate the number of embedded BHs
    number_ebh = number_R_NSC(agn.M_SMBH_msun, mave_msun, r_out_cm)

    _, m1_msun, tform_s = load_filtered_stellar_population(number_ebh)

    # 2. random sampling to get radial distribution, inner and outer set as pAGN solver
    R_cm = sample_embedded_radius(number_ebh, agn, disk)

    ebh = Embedded_BH(number_ebh)

    ebh.init_ebh(m1_msun, tform_s, R_cm)

    return ebh
