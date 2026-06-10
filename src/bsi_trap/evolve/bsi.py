"""need more details here"""

# bsi.py, contains functions for binary-single interactions in the trap
from typing import Dict, Tuple
import numpy as np

from bsi_trap.classes.classes import (
    BBHEvolver,
    BBH_properties,
    single_BH_properties,
    TrapContext,
)
import bsi_trap.constant.constant as const
from bsi_trap.evolve.numeric import _safe_eccentricity, _is_physical_state, _scalar


def focusing_factor(m1, m2, m3, b, v_rel):
    """Calculate the gravitational focusing factor for a given impact parameter."""
    if not np.isfinite(m1) or m1 <= 0.0:
        return 1.0
    if not np.isfinite(m2) or m2 <= 0.0:
        return 1.0
    if not np.isfinite(m3) or m3 <= 0.0:
        return 1.0
    if not np.isfinite(b) or b <= 0.0:
        return 1.0
    if not np.isfinite(v_rel) or v_rel <= 0.0:
        return 1.0

    menc = m1 + m2 + m3

    # focusing = 1 + (2 * G * M_enc) / (b * v_rel^2)
    focusing = 1.0 + 2.0 * const.G * menc / (b * v_rel**2)
    return max(float(focusing), 1.0)


def rate_compute(focus, bmax, number_density, v_rel):
    """Compute the encounter rate given focusing, number density, and relative velocity."""
    if not np.isfinite(focus) or focus <= 0.0:
        return 0.0
    if not np.isfinite(number_density) or number_density <= 0.0:
        return 0.0
    if not np.isfinite(v_rel) or v_rel <= 0.0:
        return 0.0

    # \sigma = \pi b^2 * focusing, where b is the impact parameter for the encounter
    sigma = np.pi * bmax**2 * np.max(float(focus), 1.0)

    rate = number_density * sigma * v_rel
    return max(float(rate), 0.0)


def bsi_rate(
    m1_g: float, m2_g: float, a_cm: float, trap_context: dict
) -> Dict[float, float]:
    """Return the trap-local weak and strong BSI rates in s^-1."""
    pass


def single_weak_bsi(m1, m2, a, e, th1, th2, trap_context):
    """Apply one weak trap encounter, conserving inner angular momentum."""
    pass


def single_strong_bsi(state, spin_state, trap_context):
    """Placeholder for future strong BSI physics; currently a no-op."""
    return state, spin_state


def apply_weak_bsi(m1, m2, a, e, th1, th2, dt, trap_context):
    """Monte Carlo weak BSI event step for one integration timestep."""
    rate, _ = bsi_rate(m1, m2, a, trap_context)
    if rate <= 0.0 or not np.isfinite(rate):
        return a, e, th1, th2, False

    p_event = 1.0 - np.exp(-rate * dt)
    p_event = min(float(p_event), 0.5)
    if np.random.random() >= p_event:
        return a, e, th1, th2, False

    a, e, th1, th2 = single_weak_bsi(m1, m2, a, e, th1, th2, trap_context)
    return a, e, th1, th2, True


def apply_bsi(
    BBH: BBH_properties,
    sBH: single_BH_properties,
    dt: float,
    trap_context: TrapContext,
) -> Tuple[BBH_properties, single_BH_properties]:
    """apply weak and stong BSI change to BBH in time line"""
    m1_g = BBH.m1_msun * const.msun
    m2_g = BBH.m2_msun * const.msun
    m3_g = sBH.m3_msun * const.msun
    a_cm = BBH.sma_cm
    e = BBH.ecc
    th1 = BBH.th1
    th2 = BBH.th2

    anew_cm, e_new, th1_new, th2_new = apply_weak_bsi(
        m1_g, m2_g, a_cm, e, th1, th2, trap_context
    )


def bsi_timescale():
    """Store t_BSI = 1/Gamma_BSI for hard binaries active at the trap."""
    pass
