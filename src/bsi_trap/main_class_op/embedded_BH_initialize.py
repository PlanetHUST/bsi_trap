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
from bsi_trap.functions.random_sample import semimajor_out_random_log


def load_filtered_stellar_population(
    number_ebh: int, fil: str = "input/"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load progenitor/remnant masses and formation times from the input catalog."""
    # similar method as fast-cluster, in order to get the tfrom and mass of stars evolution
    # need some changes for classes
    mzams, m1iii, tformii = np.genfromtxt(
        fil, usecols=(1, 2, 3), skip_header=3, unpack=True
    )

    ok = np.where((m1iii <= 30.0) & (tformii < 100.0))[0]

    return np.copy(mzams[ok]), np.copy(m1iii[ok]), np.copy(tformii[ok])


def assign_spin_tilts(BBH: BBH_properties, model_config: ModelConfig) -> None:
    """Assign spin-tilt angles using the gas/no-gas convention."""

    num = BBH.num

    if model_config.gas_mode == "with_gas":
        theta_1 = np.clip(np.random.normal(loc=1, scale=0.1, size=num), None, 1.0)
        theta_2 = np.clip(np.random.normal(loc=1, scale=0.1, size=num), None, 1.0)

    elif model_config.gas_mode == "without_gas":
        theta_1 = np.random.uniform(-1, 1, size=num)
        theta_2 = np.random.uniform(-1, 1, size=num)
    else:
        raise ValueError(f"Unknown gas_mode: {model_config.gas_mode}")

    th1, th2 = np.arccos(theta_1), np.arccos(theta_2)

    BBH.th1 = th1
    BBH.th2 = th2


def embedded_BH_initialize(
    agn: AGN_properties, disk: DiskProfile, model_config: ModelConfig
) -> Embedded_BH:

    r_out_cm = disk.r_out_cm
    mave_msun = agn.mave_msun
    # 1. calculate the number of embedded BHs
    number_ebh = number_R_NSC(agn.M_SMBH_msun, mave_msun, r_out_cm)

    mzams, m1_msun, tform_s = load_filtered_stellar_population(number_ebh)

    # 2. random sampling to get radial distribution, inner and outer set as pAGN solver
    R_cm = semimajor_out_random_log(
        number_ebh, agn, np.log10(disk.r_in_cm), np.log10(disk.r_out_cm)
    )

    ebh = Embedded_BH(number_ebh)

    ebh.init_ebh(m1_msun, tform_s, R_cm)

    return ebh
