import logging
import numpy as np

from bsi_trap.functions.random_sample import semimajor_out, ecc
from bsi_trap.functions.nsc_population import stellar_properties
from bsi_trap.classes.classes import AGN_properties, BBH_properties, DiskProfile
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.functions.nsc_population import number_R_NSC


def initialize_stellar(numstar, AGN, fil, gf):
    """Initialize the stellar/BH reservoir that seeds the AGN-disk population."""

    N = numstar
    Stars = stellar_properties(N)

    Stars.idi = np.arange(0, N, 1)

    mzams, m1ii, tformi = load_filtered_stellar_population(fil, gf)

    a = np.random.randint(0, len(m1ii), N)
    Stars.m = mzams[a]
    Stars.mBH = m1ii[a]
    tform = tformi[a]  # tform in Myr

    mBHaverage = np.mean(m1ii[a])  # average BH mass of first generation
    AGN.maverage = mBHaverage
    max1g = max(m1ii[a])  # maximum mass of first generation
    # print(f"the length of 1g Stars is {len(BBH.m1)},selecting from {len(m1ii):3e} BHs")
    # print("Maximum mass for 1g black holes is ",max1g)

    Stars.a = semimajor_out(
        len(Stars.m), AGN, 3, 6
    )  # radius boundary as 10**3 to 10**6 Rg, default gamma=7/4 for thermal distribution

    Stars.ecc = ecc(Stars.m)  # now totally arbitrary

    Stars.cosw = [np.random.random() for _ in range(N)]

    Stars.i = [np.random.random() * np.pi * 16 / 18 + 1 / 18 * np.pi for _ in range(N)]

    return Stars, AGN, mBHaverage, max1g, tform


def load_filtered_stellar_population(fil, gf):
    """Load progenitor/remnant masses and formation times from the input catalog."""
    mzams, m1iii, tformii = np.genfromtxt(
        fil, usecols=(1, 2, 3), skip_header=3, unpack=True
    )
    if gf.Z_list == "0.02":  # historical MOBSE compatibility branch
        ok = np.where((m1iii <= 30.0) & (tformii < 100.0))[0]
    else:  # historical MOBSE compatibility branch
        ok = np.where((m1iii <= 80.0) & (tformii < 100.0))[0]
    return np.copy(mzams[ok]), np.copy(m1iii[ok]), np.copy(tformii[ok])


def assign_spin_tilts(BBH, model_config):
    """Assign spin-tilt angles using the gas/no-gas convention."""

    N=len(BBH.th1)

    if model_config.gas_mode=='with_gas':
        theta_1 = np.clip(np.random.normal(loc=1, scale=0.1, size=N), None, 1.0)
        theta_2 = np.clip(np.random.normal(loc=1, scale=0.1, size=N), None, 1.0)

    elif model_config.gas_mode=='without_gas':
        theta_1 = np.random.uniform(-1, 1, size=N)
        theta_2 = np.random.uniform(-1, 1, size=N)
    else:
        raise ValueError(f"Unknown gas_mode: {model_config.gas_mode}")

    th1, th2 = np.arccos(theta_1), np.arccos(theta_2)

    BBH.th1 = th1
    BBH.th2 = th2

def embedded_BH_initialize(AGN:AGN_properties, Disk:DiskProfile, model_config: ModelConfig)->BBH_properties:
    raise NotImplementedError(
        "Embedded BBH population construction is not implemented yet; "
        "use initialize_stellar and an explicit pair-up model."
    )
