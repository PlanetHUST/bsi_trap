import numpy as np
import jimenez as ji
import constants as const


def semimajor(m1, m2, AGN):
    N = len(m1)
    a_min = const.Rsun
    m_SM = AGN.M_SMBH * const.msun
    rtrap = AGN.rtrap
    a_max = rtrap * ((m1.max() + m2.max()) * const.msun / m_SM) ** (1 / 3)
    x = np.random.rand(N)
    sma = ((a_max ** (11 / 2) - a_min ** (11 / 2)) * x + a_min ** (11 / 2)) ** (2 / 11)
    return sma  # sma in cm


def check_hard(m1, m2, sma, maverage, sigma):
    # input masses in msun, sma in cm, sigma in cm/s
    Eb = const.G * m1 * m2 * const.msun * const.msun / (2.0 * sma)
    Ek = 0.5 * maverage * const.msun * sigma * sigma

    flag = np.zeros(len(Eb), dtype=int)  # zeros for soft, ones for hard
    a = np.where(Eb >= Ek)
    # print("Warning: binary is hard")
    flag[a[0]] = np.ones(len(a[0]), dtype=int)
    # print(len(flag)-len(flag[a[0]]))
    return flag


def ecc(m1):
    # generates initial eccentricity
    # now P(e)=2 e arbitrary
    N = len(m1)
    x = np.random.rand(N)
    e = np.sqrt(x)
    return e


###################generate secondary (o'leary 2016)###############
def secondary(m1, m2min):
    # generates secondary mass (o'leary 2016)
    # N=len(m1)
    C = (2.0 * m1) ** 5.0 - (m1 + m2min) ** 5.0
    if C < 1e-5:
        print(f"horror C is zero, with C={C}, m1={m1}, m2min={m2min}")
    C = 1.0 / C
    y = np.random.rand()
    m2 = (y / C + (m1 + m2min) ** 5.0) ** (1.0 / 5.0) - m1
    return m2


#################spin of the merger remnant ###############################
# generates final spin of the merger remnant from jimenez-forteza et al 2017
def final_spin(chi1, chi2, m1, m2):
    # input masses in msun
    if m1 < m2:
        print("m1 <m2 I FLIP ", m1, m2)
        m1, m2 = m2, m1
        chi1, chi2 = chi2, chi1

    chif = ji.bbh_final_spin_non_precessing_UIB2016(m1, m2, chi1, chi2, "v2")
    if chif > 1.0:
        print("horror= ", chif)

    return chif


################# mass of the merger remnant###############################
# generates final mass of the merger remnant from jimenez-forteza et al 2017
def final_mass(m1, m2, chi1, chi2):
    # input masses in msun
    if m1 < m2:
        print("m1 <m2 I FLIP ", m1, m2)
        m1, m2 = m2, m1
        chi1, chi2 = chi2, chi1
    mf = ji.bbh_final_mass_non_precessing_UIB2016(m1, m2, chi1, chi2, "v2")
    if mf > (m1 + m2):
        print("horror mf= ", mf, m1 + m2)
    return mf


########################## SPIN MAGNITUDE ################################
# generates spin magnitude from a maxwellian
def spin(rms=0.1):
    a = 2.0
    # print(rms)
    while a > 1.0:
        # default_scale =0.1
        # alternative_scale=0.01
        n1 = np.random.normal(loc=0.0, scale=rms)
        n2 = np.random.normal(loc=0.0, scale=rms)
        n3 = np.random.normal(loc=0.0, scale=rms)

        a = (n1 * n1 + n2 * n2 + n3 * n3) ** 0.5
    return a


########################## SPIN TILT ################################
# def set_theta(x): #generate spin tilt theta
#     theta=x*2.-1.0
#     theta=np.arccos(theta)
#     return theta


def set_theta_gas(x):  # generate spin tilt theta
    theta = np.clip(np.random.normal(loc=1, scale=0.1, size=len(x)), None, 1.0)
    theta = np.arccos(theta)
    return theta


def set_theta(x):  # generate spin tilt theta
    theta = np.random.uniform(-1, 1, size=len(x))
    theta = np.arccos(theta)
    return theta


######################### RELATIVISTIC KICKS ################################
def relativistic_kick(m1, m2, a1, a2, th1, th2, version="maggiore"):

    # m1, m2 input masses in msun
    # a1, a2 input dimensionless spins [0,1]
    # th1, th2 inclination angles in rad (wrt orbital angular momentum!)
    # version indicates the fitting formula to calculate kicks

    if (version == "campanelli") or (version == "lousto1") or (version == "lousto2"):
        if m1 > m2:
            # print("m1 <m2 I FLIP ", m1, m2)
            m1, m2 = m2, m1  # lousto use m1<m2
            a1, a2 = a2, a1  # lousto use m1<m2
            th1, th2 = th2, th1  # lousto use m1<m2

        q = m1 / m2  # q=m1/m2 with m1<m2 Lousto & Zlochower 2011

    if version == "maggiore":
        if m1 < m2:
            print("m1 <m2 I FLIP ", m1, m2)
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
