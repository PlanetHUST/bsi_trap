import numpy as np
import logging

from bsi_trap.functions import jimenez as ji
import bsi_trap.constant.constant as const
from bsi_trap.functions import nsc_population as nsc


def final_spin(chi1, chi2, m1, m2):
    # generates final spin of the merger remnant from jimenez-forteza et al 2017
    # input masses in msun
    if m1 < m2:
        logging.info("m1 <m2 I FLIP ", m1, m2)
        m1, m2 = m2, m1
        chi1, chi2 = chi2, chi1

    chif = ji.bbh_final_spin_non_precessing_UIB2016(m1, m2, chi1, chi2, "v2")
    if chif > 1.0:
        logging.info("horror= ", chif)

    return chif


################# mass of the merger remnant###############################
# generates final mass of the merger remnant from jimenez-forteza et al 2017
def final_mass(m1, m2, chi1, chi2):
    # input masses in msun
    if m1 < m2:
        logging.info("m1 <m2 I FLIP ", m1, m2)
        m1, m2 = m2, m1
        chi1, chi2 = chi2, chi1
    mf = ji.bbh_final_mass_non_precessing_UIB2016(m1, m2, chi1, chi2, "v2")
    if mf > (m1 + m2):
        logging.info("horror mf= ", mf, m1 + m2)
    return mf


def final_semimajor(AGN, vfin):
    rtrap = AGN.rtrap  # rtrap in cm
    M_tot_rtrap = (AGN.M_SMBH + nsc.mass_R_NSC(AGN.M_SMBH, rtrap)) * const.msun
    Const = rtrap * vfin**2 / const.G / M_tot_rtrap
    Afin = const.G * M_tot_rtrap / vfin**2 * (Const / (2 - Const))
    # print(
    #     f" rtrap: {rtrap:3e} cm \n M_tot: {M_tot_rtrap:3e} g \n vfin: {vfin:3e} cm/s \n Const: {Const:3e} \n Afin: {Afin:3e} cm \n"
    # )
    return Afin


def relativistic_kick(m1, m2, a1, a2, th1, th2, version="maggiore"):

    # m1, m2 input masses in msun
    # a1, a2 input dimensionless spins [0,1]
    # th1, th2 inclination angles in rad (wrt orbital angular momentum!)
    # version indicates the fitting formula to calculate kicks

    if (version == "campanelli") or (version == "lousto1") or (version == "lousto2"):
        if m1 > m2:
            logging.info("m1 > m2 I FLIP ", m1, m2)
            m1, m2 = m2, m1  # lousto use m1<m2
            a1, a2 = a2, a1  # lousto use m1<m2
            th1, th2 = th2, th1  # lousto use m1<m2

        q = m1 / m2  # q=m1/m2 with m1<m2 Lousto & Zlochower 2011

    if version == "maggiore":
        if m1 < m2:
            logging.info("m1 <m2 I FLIP ", m1, m2)
            m1, m2 = m2, m1  # maggiore use m1>m2
            a1, a2 = a2, a1  # maggiore use m1>m2
            th1, th2 = th2, th1  # maggiore use m1>m2

        q = m2 / m1  # astrophysicists use m1>m2

    eta = q / (1.0 + q) ** 2

    # parallel is perpendicular to orbital plane
    # perpendicular 1 and 2 are in the orbital plane
    a1per = a1 * np.sin(th1)
    a1par = a1 * np.cos(th1)

    a2per = a2 * np.sin(th2)
    a2par = a2 * np.cos(th2)

    phi1 = np.random.rand() * 2.0 * np.pi

    Am = 1.2e4  # A
    Bm = -0.93  # B
    H = 6.9e3
    K = 6.0e4

    x = 145.0 / 360.0 * 2.0 * np.pi

    ###VERSION CAMPANELLI ET AL 2007
    if version == "campanelli":

        vm = Am * eta * eta * (1.0 - 4.0 * eta) ** 0.5 * (1.0 + Bm * eta)
        vper = H * eta * eta / (1.0 + q) * (a2par - q * a1par)
        vpar = K * eta * eta / (1.0 + q) * (a2per - q * a1per) * np.cos(phi1)

    ###VERSION LOUSTO ET AL. 2012 eq 4
    if version == "lousto1":

        HS = 0.0  # note lousto never writes it down explicitly
        KS = -4.254
        BK = 0.0
        BH = 0.0

        phi2 = np.random.rand() * 2.0 * np.pi

        vm = Am * eta * eta * (1.0 - q) / (1.0 + q) * (1.0 + Bm * eta)

        vper = (
            H
            * eta
            * eta
            / (1.0 + q)
            * (
                (1.0 + BH * eta) * (a2par - q * a1par)
                + HS * (1 - q) / (1 + q) ** 2 * (a2par + q * q * a1par)
            )
        )

        vpar = (
            K
            * eta
            * eta
            / (1.0 + q)
            * (
                (1.0 + BK * eta) * abs(a2per - q * a1per) * np.cos(phi1)
                + KS
                * (1.0 - q)
                / (1.0 + q) ** 2
                * abs(a2per + q * q * a1per)
                * np.cos(phi2)
            )
        )

    ###VERSION LOUSTO ET AL. 2012 eq 12
    if version == "lousto2":

        V11 = 3678.0  # lousto et al. 2012
        VA = 2.481e3
        VB = 1.792e3
        VC = 1.507e3
        xx = 2.0 * (a2par + q * q * a1par) / (1.0 + q) ** 2

        vm = Am * eta * eta * (1.0 - q) / (1.0 + q) * (1.0 + Bm * eta)

        vper = H * eta * eta / (1.0 + q) * (a2par - q * a1par)

        vpar = (
            16.0
            * eta
            * eta
            / (1.0 + q)
            * (V11 + VA * xx + VB * xx * xx + VC * xx * xx * xx)
            * abs(a2per - q * a1per)
            * np.cos(phi1)
        )

    ###VERSION MAGGIORE BOOK GW, 2 eqs 14.203,4,5
    # note eta := q/(1+q)**2 --> q**2/(1+q)**5 = eta * eta /(1+q)**5

    if version == "maggiore":

        Cmagg = 457.0  # km/s
        Dmagg = 3750.0  # km/s

        vm = Am * eta * eta * (1.0 - q) / (1.0 + q) * (1.0 + Bm * eta)

        vper = Cmagg * 16.0 * eta * eta / (1.0 + q) * abs(a1par - q * a2par)

        vpar = (
            Dmagg * 16.0 * eta * eta / (1.0 + q) * abs(a1per - q * a2per) * np.cos(phi1)
        )

    vk = (vm * vm + vper * vper + 2.0 * vm * vper * np.cos(x) + vpar * vpar) ** 0.5

    # print(m1,m2,a1,a2,th1,th2,vk)

    ##if((version=="campanelli") or (version=="lousto1") or  (version=="lousto2")):
    ##    th1,th2=th2,th1

    return 1e5 * vk
