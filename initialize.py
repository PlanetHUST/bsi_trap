"""Population initialization helpers for first- and higher-generation BBHs.

This module is intentionally lightweight: it samples the input stellar/remnant
catalog and assigns initial orbital/spin properties before the detailed
evolution is handled elsewhere.
"""

import numpy as np
import auxiliary as auxi
import general_class as agn
import agn_function as agn_f
from model_config import ModelConfig, legacy_model_config


def _load_filtered_stellar_population(fil, gf):
    """Load progenitor/remnant masses and formation times from the input catalog."""
    mzams, m1iii, tformii = np.genfromtxt(
        fil, usecols=(1, 2, 3), skip_header=3, unpack=True
    )
    if gf.Z_list == "0.02":  # historical MOBSE compatibility branch
        ok = np.where((m1iii <= 30.0) & (tformii < 100.0))[0]
    else:  # historical MOBSE compatibility branch
        ok = np.where((m1iii <= 80.0) & (tformii < 100.0))[0]
    return np.copy(mzams[ok]), np.copy(m1iii[ok]), np.copy(tformii[ok])


def _gas_mode(model_config: ModelConfig | str) -> str:
    if isinstance(model_config, ModelConfig):
        return model_config.gas_mode
    return legacy_model_config(str(model_config)).gas_mode


def _assign_spin_tilts(BBH, N, model_config):
    """Assign spin-tilt angles using the gas/no-gas convention."""
    th1 = np.random.rand(N)  # 0 means aligned with orbital angular momentum
    th2 = np.random.rand(N)
    gas_mode = _gas_mode(model_config)
    if gas_mode == "without_gas":
        BBH.th1 = auxi.set_theta(th1)
        BBH.th2 = auxi.set_theta(th2)
    elif gas_mode == "with_gas":
        BBH.th1 = auxi.set_theta_gas(th1)
        BBH.th2 = auxi.set_theta_gas(th2)
    else:
        raise ValueError(f"Unknown gas_mode: {gas_mode}")


# ---------------------------------------------------------------------------
# Higher-generation initialization
# ---------------------------------------------------------------------------


def initialize_nthgen(BBH, AGN, max1g, gf, type):  # initializes N-gen binary
    """Initialize a higher-generation BBH population from merger remnants."""

    N = len(BBH.m1)

    for i in range(N):
        if gf.ngng == True:
            BBH.m2[i] = auxi.secondary(BBH.m1[i], 0.9 * 2.0 * BBH.m2min)
        else:
            BBH.m2[i] = auxi.secondary(BBH.m1[i], BBH.m2min)

        if (gf.ngng == True) or (BBH.m2[i] > max1g):
            pp = np.random.randint(0, N)
            BBH.chi2[i] = BBH.chi1[pp]
        else:
            BBH.chi2[i] = auxi.spin(gf.spinrms)

    BBH.sma = auxi.semimajor(BBH.m1, BBH.m2, AGN)  # now totally arbitrary
    BBH.ecc = auxi.ecc(BBH.m1)  # now totally arbitrary

    _assign_spin_tilts(BBH, N, type)

    return BBH


def initialize_stellar(numstar, AGN, fil, gf):
    """Initialize the stellar/BH reservoir that seeds the AGN-disk population."""

    N = numstar
    Stars = agn.stellar_properties(N)

    Stars.idi = np.arange(0, N, 1)

    mzams, m1ii, tformi = _load_filtered_stellar_population(fil, gf)
    # print("Removed ",len(m1iii)-len(m1ii)," black holes")

    # print(len(m1ii),N)

    a = np.random.randint(0, len(m1ii), N)
    Stars.m = mzams[a]
    Stars.mBH = m1ii[a]
    tform = tformi[a]  # tform in Myr

    mBHaverage = np.mean(m1ii[a])  # average BH mass of first generation
    AGN.maverage = mBHaverage
    max1g = max(m1ii[a])  # maximum mass of first generation
    # print(f"the length of 1g Stars is {len(BBH.m1)},selecting from {len(m1ii):3e} BHs")
    # print("Maximum mass for 1g black holes is ",max1g)

    Stars.a = agn_f.semimajor_out(
        len(Stars.m), AGN, 3, 6
    )  # radius boundary as 10**3 to 10**6 Rg, default gamma=7/4 for thermal distribution

    Stars.ecc = auxi.ecc(Stars.m)  # now totally arbitrary

    Stars.cosw = [np.random.random() for _ in range(N)]

    Stars.i = [np.random.random() * np.pi * 16 / 18 + 1 / 18 * np.pi for _ in range(N)]

    return Stars, AGN, mBHaverage, max1g, tform
