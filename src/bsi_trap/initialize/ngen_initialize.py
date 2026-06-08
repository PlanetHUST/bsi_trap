import numpy as np

from bsi_trap.functions.random_sample import semimajor, ecc, secondary, spin
from bsi_trap.initialize.embedded_BH_initialize import assign_spin_tilts


def initialize_nthgen(BBH, AGN, max1g, gf, type):

    N = len(BBH.m1)

    for i in range(N):
        if gf.ngng == True:
            BBH.m2[i] = secondary(BBH.m1[i], 0.9 * 2.0 * BBH.m2min)
        else:
            BBH.m2[i] = secondary(BBH.m1[i], BBH.m2min)

        if (gf.ngng == True) or (BBH.m2[i] > max1g):
            pp = np.random.randint(0, N)
            BBH.chi2[i] = BBH.chi1[pp]
        else:
            BBH.chi2[i] = spin(gf.spinrms)

    BBH.sma = semimajor(BBH.m1, BBH.m2, AGN)  # now totally arbitrary
    BBH.ecc = ecc(BBH.m1)  # now totally arbitrary

    assign_spin_tilts(BBH, type)

    return BBH
