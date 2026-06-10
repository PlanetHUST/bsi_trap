import logging
import numpy as np
from dataclasses import dataclass

######### Environment Global Parameters ##########


class AGN_properties(object):
    "AGN disk profile"

    def __init__(self, M_SMBH_msun, lifetime_myr):
        self.M_SMBH_msun = M_SMBH_msun
        self.alpha = None
        self.lifetime_myr = lifetime_myr
        self.mave_msun = float
        self.disk_profile = None
        self.torque_profile = None


@dataclass(frozen=True)
class DiskProfile:
    radius_cm: np.ndarray
    h_over_r: np.ndarray
    sigma_cgs: np.ndarray
    cs_cgs: np.ndarray
    r_in_cm: float
    r_out_cm: float
    r_trap_cm: float
    r_antitrap_cm: float

    def __post_init__(self):
        if self.radius_cm.size == 0:
            raise ValueError("DiskProfile.radius_cm must not be empty")

        if not (
            self.radius_cm.size
            == self.h_over_r.size
            == self.sigma_cgs.size
            == self.cs_cgs.size
        ):
            raise ValueError("All disk profile arrays must have the same length")

        if not np.all(np.isfinite(self.radius_cm)):
            raise ValueError("DiskProfile.radius_cm contains non-finite values")

        if not np.all(self.radius_cm > 0):
            raise ValueError("DiskProfile.radius_cm must be positive")

    def closest_index(self, r_cm: float) -> int:
        idx = np.searchsorted(self.radius_cm, r_cm)

        if idx == 0:
            return 0
        if idx == len(self.radius_cm):
            return len(self.radius_cm) - 1

        left = idx - 1
        right = idx

        if abs(self.radius_cm[left] - r_cm) <= abs(self.radius_cm[right] - r_cm):
            return left
        return right

    def get_h(self, r_cm: float) -> float:
        return float(self.h_over_r[self.closest_index(r_cm)])

    def get_sigma(self, r_cm: float) -> float:
        return float(self.sigma_cgs[self.closest_index(r_cm)])

    def get_cs(self, r_cm: float) -> float:
        return float(self.cs_cgs[self.closest_index(r_cm)])


############ Torque Profile #############


@dataclass
class TorqueProfile:
    radius: np.ndarray
    radius_cm: np.ndarray
    radius_rg: np.ndarray
    radius_rs: np.ndarray
    surface_density: np.ndarray
    pressure: np.ndarray
    dsigma_dlogr: np.ndarray
    dtemp_dlogr: np.ndarray
    dpressure_dlogr: np.ndarray
    corotation_coeff: np.ndarray
    lindblad_coeff: np.ndarray
    thermal_coeff: np.ndarray
    gap_parameter: np.ndarray
    gamma_0: np.ndarray
    gamma_i: np.ndarray
    gamma_th: np.ndarray
    gamma_tot: np.ndarray
    tau_mig: np.ndarray
    tau_e: np.ndarray
    tau_i: np.ndarray


########## BH Dynamics Classes ############


class Embedded_BH(object):
    """embedded BHs in in the disk"""

    def __init__(self, num):
        self.num = num
        self.m_msun = np.zeros(num, float)
        self.t_form_s = np.zeros(num, float)
        self.t_ini_s = np.zeros(num, float)
        self.R_cm = np.zeros(num, float)
        self.trap_flag = np.zeros(
            num, int
        )  # -1: trap not reach; -2: single stars; 0-num: binary pair-up with index

    def init_ebh(self, m_msun, t_form_s, R_cm):
        if len(m_msun) == len(t_form_s) == len(R_cm) == self.num:
            self.m_msun = m_msun
            self.t_form_s = t_form_s
            self.R_cm = R_cm
        else:
            logging.error(
                f"unmatched length of variable with len(m_msun)={len(m_msun)}; len(t_form_s)={len(t_form_s)}; len(R_cm)={len(R_cm)}; num = {self.num}"
            )


class BBH_properties(object):
    """BBH population in the trap"""

    def __init__(self, num):
        self.num = num
        self.idi = np.zeros(num, int)
        self.m1_msun = np.zeros(num, float)
        self.m2_msun = np.zeros(num, float)
        self.sma_cm = np.zeros(num, float)
        self.ecc = np.zeros(num, float)

        self.ecc10 = np.zeros(num, float)  # eccentricity at fGW=10 Hz
        self.ecc0p1 = np.zeros(num, float)  # eccentricity at fGW=0.1 Hz

        self.chi1 = np.zeros(num, float)
        self.chi2 = np.zeros(num, float)
        self.th1 = np.zeros(num, float)
        self.th2 = np.zeros(num, float)

        self.A_cm = np.zeros(num, float)  # outer orbital semimajor axis in cm

        self.mrem_msun = np.zeros(num, float)  # mass of merger remnant
        self.mmerg = self.mrem_msun
        self.chirem = np.zeros(num, float)
        self.chimerg = self.chirem
        self.vk_cgs = np.zeros(num, float)  # kick velocity

        self.t_ini_s = np.zeros(num, float)
        self.t_del_s = np.zeros(num, float)
        self.bbh_flag = None


class single_BH_properties(object):
    "single BH population properties in the trap"

    def __init__(self, num):
        self.num = num
        self.m_msun = np.zeros(num, float)
        self.t_ini_s = np.zeros(num, float)


class BBHEvolverParams:
    "BBH evolution class for simulation parameters"

    def __init__(self, rlist, sigmalist, hlist, cslist):
        self.rlist = rlist
        self.sigmalist = sigmalist
        self.hlist = hlist
        self.cslist = cslist

        # Parameters for weak BSI
        self.bsi_eta_weak = 2.0  # Maximum impact parameter in units of a for weak BSI
        self.bsi_eta_strong = (
            0.5  # Maximum impact parameter in units of a for strong BSI
        )
        self.bsi_sigma_e0 = 0.1  # Base eccentricity change for weak BSI

        # Eccentricity limits to avoid unphysical values
        self.ecc_floor = 1e-5
        self.ecc_ceiling = 0.9999
        self.tol_a = 1.0e-3
        self.tol_e = 1.0e-3

        self.trap_bsi_context = None  # To store local conditions for BSI calculations


class bbh_flags:
    def __init__(self, num):
        self.merger_flag = np.zeros(num, int)  # 1 if merger occurred, 0 otherwise
        self.GWescape_flag = np.zeros(num, int)
        self.continue_ng_flag = np.zeros(num, int)
        self.hard = np.ones(num, int)
        self.bsi_event_flag = np.zeros(num, int)
        self.bsi_exchange_flag = np.zeros(num, int)
        self.bsi_ionization_flag = np.zeros(num, int)
        self.bsi_ejection_flag = np.zeros(num, int)
        self.trap_reached_flag = np.zeros(num, int)  # 1 if trap reached, 0 otherwise
        self.stop_reason_flag = np.zeros(num, int)  # code for why evolution stopped


@dataclass(frozen=True)
class MigrationParameters:
    alpha: float = 0.01
    gamma_adiabat: float = 5.0 / 3.0
    hydrogen_fraction: float = 0.7
    kappa_t: float = 20.0
    kappa_norm: float = 25.0


class TrapContext:
    def __ini__(self):
        pass
