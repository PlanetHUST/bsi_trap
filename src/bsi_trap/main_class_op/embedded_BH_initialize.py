from pathlib import Path
from typing import Tuple

import numpy as np

from bsi_trap.classes.classes import (
    AGN_properties,
    DiskProfile,
    Embedded_BH,
)
from bsi_trap.constant.constant import Myr
from bsi_trap.functions.nsc_population import number_R_NSC
from bsi_trap.functions.random_sample import sample_embedded_radius

DEFAULT_STELLAR_CATALOG = (
    Path(__file__).resolve().parents[3] / "input" / "input_data.txt"
)


def load_filtered_stellar_population(
    catalog_path: str | Path = DEFAULT_STELLAR_CATALOG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load filtered progenitor/remnant masses and formation times in Myr."""
    # similar method as fast-cluster, in order to get the tfrom and mass of stars evolution
    # need some changes for classes
    mzams, m1iii, tformii = np.genfromtxt(
        catalog_path, usecols=(1, 2, 3), skip_header=3, unpack=True
    )

    ok = np.isfinite(m1iii) & np.isfinite(tformii) & (m1iii > 0.0)
    ok &= (m1iii <= 30.0) & (tformii >= 0.0) & (tformii < 100.0)
    if not np.any(ok):
        raise ValueError(f"No valid black-hole remnants found in {catalog_path}")

    return np.copy(mzams[ok]), np.copy(m1iii[ok]), np.copy(tformii[ok])


def embedded_BH_initialize(
    agn: AGN_properties,
    disk: DiskProfile,
    rng_nsc: np.random.Generator,
    catalog_path: str | Path | None = None,
) -> Embedded_BH:
    """Initialize the embedded-BH population from the filtered remnant catalog."""
    if catalog_path is None:
        catalog_path = DEFAULT_STELLAR_CATALOG

    r_out_cm = disk.r_out_cm
    mave_msun = agn.mave_msun

    # 1. calculate the number of embedded BHs
    number_ebh = number_R_NSC(agn.M_SMBH_msun, mave_msun, r_out_cm)
    if number_ebh < 0:
        raise ValueError(
            f"Embedded-BH population size must be nonnegative, got {number_ebh}"
        )

    _, catalog_mass_msun, catalog_tform_myr = load_filtered_stellar_population(
        catalog_path
    )
    sample_idx = rng_nsc.integers(0, catalog_mass_msun.size, size=number_ebh)
    m1_msun = catalog_mass_msun[sample_idx]
    tform_s = catalog_tform_myr[sample_idx] * Myr

    # 2. random sampling to get radial distribution, inner and outer set as pAGN solver
    R_cm = sample_embedded_radius(number_ebh, agn, disk, rng_nsc)

    ebh = Embedded_BH(number_ebh)

    ebh.init_ebh(m1_msun, tform_s, R_cm)

    return ebh
