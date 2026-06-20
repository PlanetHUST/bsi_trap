"""Merger remnant and recoil helper functions.

This module is intentionally scalar: each function evaluates one BBH event.
Array-level loops/vectorization should live in the workflow/evolution layer.

Units
-----
- BH masses passed to ``final_mass``, ``final_spin``, and ``relativistic_kick``
  are in solar masses.
- Spin magnitudes are dimensionless.
- Spin tilt angles are in radians.
- ``relativistic_kick`` returns cm/s.
- ``final_semimajor`` returns cm.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from numpy.random import Generator

import bsi_trap.constant.constant as const
from bsi_trap.classes.models import AGN_properties, DiskProfile
from bsi_trap.functions import jimenez as ji
from bsi_trap.functions import nsc_population as nsc

logger = logging.getLogger(__name__)

KickFormula = Literal["maggiore", "campanelli", "lousto1", "lousto2"]
FinalStateFitVersion = Literal["v1", "v2"]


def _as_float(value: object, name: str) -> float:
    """Convert a scalar-like value to float and reject non-scalar arrays."""
    arr = np.asarray(value)

    if arr.shape != ():
        raise TypeError(f"{name} must be scalar, got shape {arr.shape}")

    out = float(arr.item())

    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")

    return out


def _validate_mass_pair(m1_msun: float, m2_msun: float) -> tuple[float, float]:
    if m1_msun <= 0.0:
        raise ValueError(f"m1_msun must be positive, got {m1_msun}")

    if m2_msun <= 0.0:
        raise ValueError(f"m2_msun must be positive, got {m2_msun}")

    return m1_msun, m2_msun


def _validate_spin(spin: float, name: str) -> float:
    if not -1.0 <= spin <= 1.0:
        raise ValueError(f"{name} must be in [-1, 1], got {spin}")

    return spin


def _validate_tilt(theta_rad: float, name: str) -> float:
    # The recoil formulas only require sin/cos, but finite angles avoid NaN propagation.
    if not np.isfinite(theta_rad):
        raise ValueError(f"{name} must be finite, got {theta_rad}")

    return theta_rad


def _ordered_for_jimenez(
    m1_msun: float,
    m2_msun: float,
    chi1: float,
    chi2: float,
) -> tuple[float, float, float, float]:
    """Jimenez-Forteza fits assume the first object is the primary."""
    if m1_msun < m2_msun:
        logger.debug(
            "Swapping binary components for Jimenez fit: m1=%s, m2=%s",
            m1_msun,
            m2_msun,
        )
        return m2_msun, m1_msun, chi2, chi1

    return m1_msun, m2_msun, chi1, chi2


def final_spin(
    chi1: float,
    chi2: float,
    m1_msun: float,
    m2_msun: float,
    *,
    version: FinalStateFitVersion = "v2",
) -> float:
    """Return the dimensionless aligned-spin remnant spin.

    The numerical fit is delegated to
    ``jimenez.bbh_final_spin_non_precessing_UIB2016``.
    """
    m1_msun = _as_float(m1_msun, "m1_msun")
    m2_msun = _as_float(m2_msun, "m2_msun")
    chi1 = _validate_spin(_as_float(chi1, "chi1"), "chi1")
    chi2 = _validate_spin(_as_float(chi2, "chi2"), "chi2")
    m1_msun, m2_msun = _validate_mass_pair(m1_msun, m2_msun)

    m1_fit, m2_fit, chi1_fit, chi2_fit = _ordered_for_jimenez(
        m1_msun,
        m2_msun,
        chi1,
        chi2,
    )

    spin = float(
        np.asarray(
            ji.bbh_final_spin_non_precessing_UIB2016(
                m1_fit,
                m2_fit,
                chi1_fit,
                chi2_fit,
                version,
            )
        ).item()
    )

    if spin > 1.0:
        logger.warning("Final spin exceeds unity: chif=%s", spin)

    return spin


def final_mass(
    m1_msun: float,
    m2_msun: float,
    chi1: float,
    chi2: float,
    *,
    version: FinalStateFitVersion = "v2",
) -> float:
    """Return the remnant mass in solar masses."""
    m1_msun = _as_float(m1_msun, "m1_msun")
    m2_msun = _as_float(m2_msun, "m2_msun")
    chi1 = _validate_spin(_as_float(chi1, "chi1"), "chi1")
    chi2 = _validate_spin(_as_float(chi2, "chi2"), "chi2")
    m1_msun, m2_msun = _validate_mass_pair(m1_msun, m2_msun)

    m1_fit, m2_fit, chi1_fit, chi2_fit = _ordered_for_jimenez(
        m1_msun,
        m2_msun,
        chi1,
        chi2,
    )

    mass = float(
        np.asarray(
            ji.bbh_final_mass_non_precessing_UIB2016(
                m1_fit,
                m2_fit,
                chi1_fit,
                chi2_fit,
                version,
            )
        ).item()
    )

    if mass > m1_msun + m2_msun:
        logger.warning(
            "Final mass exceeds initial total mass: Mf=%s, Mtot=%s",
            mass,
            m1_msun + m2_msun,
        )

    return mass


def final_semimajor(
    agn: AGN_properties,
    disk: DiskProfile,
    vfin_cgs: float,
) -> float:
    """Return the post-recoil outer semimajor axis in cm.

    ``vfin_cgs`` is the remnant velocity relative to the SMBH/NSC potential.
    If the orbit is formally unbound by this two-body estimate, ``np.inf`` is
    returned instead of a negative semimajor axis.
    """
    vfin_cgs = _as_float(vfin_cgs, "vfin_cgs")

    if vfin_cgs <= 0.0:
        raise ValueError(f"vfin_cgs must be positive, got {vfin_cgs}")

    r_trap_cm = _as_float(disk.r_trap_cm, "disk.r_trap_cm")
    m_smbh_msun = _as_float(agn.M_SMBH_msun, "agn.M_SMBH_msun")

    if r_trap_cm <= 0.0:
        raise ValueError(f"disk.r_trap_cm must be positive, got {r_trap_cm}")

    if m_smbh_msun <= 0.0:
        raise ValueError(f"agn.M_SMBH_msun must be positive, got {m_smbh_msun}")

    m_tot_g = (m_smbh_msun + nsc.mass_enclosed_NSC(m_smbh_msun, r_trap_cm)) * const.msun

    if m_tot_g <= 0.0:
        raise ValueError(f"Total enclosed mass must be positive, got {m_tot_g}")

    energy_parameter = r_trap_cm * vfin_cgs**2 / (const.G * m_tot_g)
    denominator = 2.0 - energy_parameter

    if denominator <= 0.0:
        return float(np.inf)

    return float(const.G * m_tot_g / vfin_cgs**2 * (energy_parameter / denominator))


def _draw_phase(rng: Generator | None) -> float:
    generator = rng if rng is not None else np.random.default_rng()
    return float(generator.uniform(0.0, 2.0 * np.pi))


def _reorder_for_recoil_formula(
    m1_msun: float,
    m2_msun: float,
    a1: float,
    a2: float,
    th1: float,
    th2: float,
    version: KickFormula,
) -> tuple[float, float, float, float, float, float, float]:
    """Return reordered components plus mass ratio q for the selected formula."""
    if version in {"campanelli", "lousto1", "lousto2"}:
        # Lousto/Campanelli convention: m1 <= m2, q = m1/m2.
        if m1_msun > m2_msun:
            logger.debug("Swapping binary components for %s recoil convention", version)
            m1_msun, m2_msun = m2_msun, m1_msun
            a1, a2 = a2, a1
            th1, th2 = th2, th1

        q = m1_msun / m2_msun
        return m1_msun, m2_msun, a1, a2, th1, th2, q

    if version == "maggiore":
        # Maggiore convention: m1 >= m2, q = m2/m1.
        if m1_msun < m2_msun:
            logger.debug("Swapping binary components for Maggiore recoil convention")
            m1_msun, m2_msun = m2_msun, m1_msun
            a1, a2 = a2, a1
            th1, th2 = th2, th1

        q = m2_msun / m1_msun
        return m1_msun, m2_msun, a1, a2, th1, th2, q

    raise ValueError(f"Unknown recoil formula version: {version!r}")


def relativistic_kick(
    m1_msun: float,
    m2_msun: float,
    a1: float,
    a2: float,
    th1: float,
    th2: float,
    version: KickFormula = "maggiore",
    *,
    rng: Generator | None = None,
) -> float:
    """Return the gravitational-wave recoil velocity in cm/s.

    Parameters
    ----------
    m1_msun, m2_msun:
        Component masses in solar masses.
    a1, a2:
        Dimensionless spin magnitudes. They are validated in [0, 1].
    th1, th2:
        Spin-tilt angles in radians, measured relative to the orbital angular
        momentum.
    version:
        Recoil prescription: ``"maggiore"``, ``"campanelli"``, ``"lousto1"``,
        or ``"lousto2"``.
    rng:
        Optional NumPy random generator used for recoil phase angles. Pass this
        from ``RNGStreams.dynamics`` for reproducible Monte Carlo runs.
    """
    m1_msun = _as_float(m1_msun, "m1_msun")
    m2_msun = _as_float(m2_msun, "m2_msun")
    a1 = _as_float(a1, "a1")
    a2 = _as_float(a2, "a2")
    th1 = _validate_tilt(_as_float(th1, "th1"), "th1")
    th2 = _validate_tilt(_as_float(th2, "th2"), "th2")
    m1_msun, m2_msun = _validate_mass_pair(m1_msun, m2_msun)

    if not 0.0 <= a1 <= 1.0:
        raise ValueError(f"a1 must be in [0, 1], got {a1}")

    if not 0.0 <= a2 <= 1.0:
        raise ValueError(f"a2 must be in [0, 1], got {a2}")

    m1_msun, m2_msun, a1, a2, th1, th2, q = _reorder_for_recoil_formula(
        m1_msun,
        m2_msun,
        a1,
        a2,
        th1,
        th2,
        version,
    )

    eta = q / (1.0 + q) ** 2

    a1_perp = a1 * np.sin(th1)
    a1_parallel = a1 * np.cos(th1)
    a2_perp = a2 * np.sin(th2)
    a2_parallel = a2 * np.cos(th2)

    phi1 = _draw_phase(rng)

    # Common coefficients. Velocities are in km/s before the final cgs conversion.
    a_mass = 1.2e4
    b_mass = -0.93
    h_spin = 6.9e3
    k_spin = 6.0e4
    mixing_angle = 145.0 / 360.0 * 2.0 * np.pi

    if version == "campanelli":
        v_mass = a_mass * eta**2 * np.sqrt(1.0 - 4.0 * eta) * (1.0 + b_mass * eta)
        v_perp = h_spin * eta**2 / (1.0 + q) * (a2_parallel - q * a1_parallel)
        v_parallel = (
            k_spin * eta**2 / (1.0 + q) * (a2_perp - q * a1_perp) * np.cos(phi1)
        )

    elif version == "lousto1":
        h_s = 0.0
        k_s = -4.254
        b_k = 0.0
        b_h = 0.0
        phi2 = _draw_phase(rng)

        v_mass = a_mass * eta**2 * (1.0 - q) / (1.0 + q) * (1.0 + b_mass * eta)
        v_perp = (
            h_spin
            * eta**2
            / (1.0 + q)
            * (
                (1.0 + b_h * eta) * (a2_parallel - q * a1_parallel)
                + h_s * (1.0 - q) / (1.0 + q) ** 2 * (a2_parallel + q**2 * a1_parallel)
            )
        )
        v_parallel = (
            k_spin
            * eta**2
            / (1.0 + q)
            * (
                (1.0 + b_k * eta) * abs(a2_perp - q * a1_perp) * np.cos(phi1)
                + k_s
                * (1.0 - q)
                / (1.0 + q) ** 2
                * abs(a2_perp + q**2 * a1_perp)
                * np.cos(phi2)
            )
        )

    elif version == "lousto2":
        v_11 = 3678.0
        v_a = 2.481e3
        v_b = 1.792e3
        v_c = 1.507e3
        xi = 2.0 * (a2_parallel + q**2 * a1_parallel) / (1.0 + q) ** 2

        v_mass = a_mass * eta**2 * (1.0 - q) / (1.0 + q) * (1.0 + b_mass * eta)
        v_perp = h_spin * eta**2 / (1.0 + q) * (a2_parallel - q * a1_parallel)
        v_parallel = (
            16.0
            * eta**2
            / (1.0 + q)
            * (v_11 + v_a * xi + v_b * xi**2 + v_c * xi**3)
            * abs(a2_perp - q * a1_perp)
            * np.cos(phi1)
        )

    elif version == "maggiore":
        c_maggiore = 457.0
        d_maggiore = 3750.0

        v_mass = a_mass * eta**2 * (1.0 - q) / (1.0 + q) * (1.0 + b_mass * eta)
        v_perp = (
            c_maggiore * 16.0 * eta**2 / (1.0 + q) * abs(a1_parallel - q * a2_parallel)
        )
        v_parallel = (
            d_maggiore
            * 16.0
            * eta**2
            / (1.0 + q)
            * abs(a1_perp - q * a2_perp)
            * np.cos(phi1)
        )

    else:
        # This branch is unreachable for a valid KickFormula but helps static analyzers.
        raise ValueError(f"Unknown recoil formula version: {version!r}")

    v_kick_km_s = np.sqrt(
        v_mass**2
        + v_perp**2
        + 2.0 * v_mass * v_perp * np.cos(mixing_angle)
        + v_parallel**2
    )

    return float(1.0e5 * v_kick_km_s)
