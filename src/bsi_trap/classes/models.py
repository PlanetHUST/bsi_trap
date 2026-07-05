"""Core disk, population, and trap-domain models."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Any

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

######### Environment Global Parameters ##########


class AGN_properties(object):
    "AGN disk profile"

    def __init__(self, M_SMBH_msun, lifetime_myr):
        self.M_SMBH_msun = M_SMBH_msun
        self.alpha: float = 0.01
        self.lifetime_myr = lifetime_myr
        self.mave_msun: float = 1.0
        self.disk_profile: DiskProfile | None = None
        self.torque_profile: TorqueProfile | None = None
        self.pagn_disk: Any | None = None


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
            return int(left)
        return int(right)

    def get_h_over_r(self, r_cm: float) -> float:
        """Return the local disk aspect ratio H/R."""
        return float(self.h_over_r[self.closest_index(r_cm)])

    def get_scale_height_cm(self, r_cm: float) -> float:
        """Return the local gas scale height H in cm."""
        return self.get_h_over_r(r_cm) * r_cm

    def get_h(self, r_cm: float) -> float:
        """Return the local disk aspect ratio H/R.

        Kept for existing migration code. New dimensional calculations should
        call :meth:`get_scale_height_cm` explicitly.
        """
        return self.get_h_over_r(r_cm)

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
    """
    ``trap_flag`` records the state after initial pairing:
    ``-1`` did not reach the trap, ``-2`` reached but remains single, and a
    non-negative value is the index of the selected binary companion.
    ``t_pair_s`` is the shared BBH formation time for paired entries and NaN
    for all others.
    """

    def __init__(self, num: int):
        self.num: int = num
        self.m_msun: FloatArray = np.zeros(num, dtype=np.float64)
        self.t_form_s: FloatArray = np.zeros(num, dtype=np.float64)
        self.t_mig_s: FloatArray = np.zeros(num, dtype=np.float64)
        self.t_reach_s: FloatArray = np.zeros(num, dtype=np.float64)
        self.t_pair_s: FloatArray = np.full(num, np.nan, dtype=np.float64)
        self.R_cm: FloatArray = np.zeros(num, dtype=np.float64)
        self.r_trap_target_cm: FloatArray = np.full(num, np.nan, dtype=np.float64)
        self.r_pair_cm: FloatArray = np.full(num, np.nan, dtype=np.float64)
        self.chi: FloatArray = np.full(num, np.nan, dtype=np.float64)
        self.theta: FloatArray = np.full(num, np.nan, dtype=np.float64)
        self.trap_width_cm: float = np.nan
        self.trap_flag: IntArray = np.full(num, -1, dtype=np.int64)

    def init_ebh(
        self,
        m_msun: npt.ArrayLike,
        t_form_s: npt.ArrayLike,
        R_cm: npt.ArrayLike,
    ) -> None:
        m_msun = np.asarray(m_msun, dtype=np.float64)
        t_form_s = np.asarray(t_form_s, dtype=np.float64)
        R_cm = np.asarray(R_cm, dtype=np.float64)
        if not (len(m_msun) == len(t_form_s) == len(R_cm) == self.num):
            raise ValueError(
                "Embedded-BH arrays must all match the declared population size "
                f"{self.num}; got {len(m_msun)}, {len(t_form_s)}, and {len(R_cm)}"
            )
        if np.any(m_msun <= 0.0) or np.any(t_form_s < 0.0) or np.any(R_cm <= 0.0):
            raise ValueError(
                "Embedded-BH masses/radii must be positive and times nonnegative"
            )

        self.m_msun = m_msun
        self.t_form_s = t_form_s
        self.R_cm = R_cm


class BBH_properties(object):
    """BBH population in the trap"""

    def __init__(self, num):
        self.num = num
        self.idi = np.zeros(num, int)
        self.m1_msun = np.zeros(num, float)
        self.m2_msun = np.zeros(num, float)
        self.sma_initial_cm = np.full(num, np.nan, float)  # for copies
        self.ecc_initial = np.full(num, np.nan, float)  # for copies
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
        self.bsi_timescale_s = np.full(num, np.nan, float)
        self.bsi_applied_flag = np.zeros(num, int)
        self.bsi_weak_count = np.zeros(num, int)
        self.bsi_delta_e_total = np.zeros(num, float)
        self.bsi_delta_theta_total = np.zeros(num, float)
        self.track_binary_idx = np.empty(0, dtype=np.int64)
        self.track_step_idx = np.empty(0, dtype=np.int64)
        self.track_time_s = np.empty(0, dtype=np.float64)
        self.track_sma_cm = np.empty(0, dtype=np.float64)
        self.track_ecc = np.empty(0, dtype=np.float64)
        self._track_binary_idx_chunks: list[IntArray] = []
        self._track_step_idx_chunks: list[IntArray] = []
        self._track_time_s_chunks: list[FloatArray] = []
        self._track_sma_cm_chunks: list[FloatArray] = []
        self._track_ecc_chunks: list[FloatArray] = []
        self.flags = BBH_flags(num)


class Single_BH_properties(object):
    "single BH population properties in the trap"

    def __init__(self, num):
        self.num = num
        self.m_msun = np.zeros(num, float)
        self.t_ini_s = np.zeros(num, float)
        self.chi = np.full(num, np.nan, float)
        self.theta = np.full(num, np.nan, float)
        self.r_pair_cm = np.full(num, np.nan, float)


class BBHEvolverParams:
    "BBH evolution class for simulation parameters"

    def __init__(self, rlist, sigmalist, hlist, cslist):
        self.rlist = rlist
        self.sigmalist = sigmalist
        self.hlist = hlist
        self.cslist = cslist

        # Parameters for reduced weak BSI
        self.bsi_eta_weak = 2.0  # Maximum impact parameter in units of a for weak BSI
        self.bsi_sigma_e0 = 0.1  # Base eccentricity change for weak BSI

        # Eccentricity limits to avoid unphysical values
        self.ecc_floor = 1e-5
        self.ecc_ceiling = 0.9999
        self.tol_a = 1.0e-3
        self.tol_e = 1.0e-3

        # Numerical/evolution policy for the reduced BBH hardening model.
        self.gw_initial_step_fraction = 1.0e-2
        self.gas_initial_step_fraction = 1.0e-6
        self.merger_radius_factor = 1.1
        self.merger_boundary_fraction = 1.0e-1
        self.max_evolution_loops = 1_000_000
        self.track_max_smooth_points = 512
        self.eccentricity_probe_low_hz = 0.1
        self.eccentricity_probe_high_hz = 10.0

        self.trap_bsi_context = None  # To store local conditions for BSI calculations


class BBH_flags:
    def __init__(self, num):
        self.merger_flag = np.zeros(num, int)  # 1 if merger occurred, 0 otherwise
        self.GWescape_flag = np.zeros(num, int)
        self.continue_ng_flag = np.zeros(num, int)
        self.hard = np.ones(num, int)
        self.bsi_event_flag = np.zeros(num, int)
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
    """Local trap geometry used by the initial pairing event model.

    Parameters are all CGS quantities. ``volumn_cgs`` remains as a temporary
    compatibility alias for existing code; use ``volume_cm3`` in new code.
    """

    def __init__(self, width_cm: float, scale_height_cm: float, volume_cm3: float):
        if width_cm <= 0.0 or scale_height_cm <= 0.0 or volume_cm3 <= 0.0:
            raise ValueError("TrapContext dimensions and volume must be positive")

        self.width_cm = float(width_cm)
        self.scale_height_cm = float(scale_height_cm)
        self.volume_cm3 = float(volume_cm3)
        self.volumn_cgs = self.volume_cm3
