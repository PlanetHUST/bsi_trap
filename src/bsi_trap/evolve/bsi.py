'''need more details here'''
# bsi.py, contains functions for binary-single interactions in the trap
from typing import Dict
import numpy as np

from bsi_trap.classes.classes import BBHEvolver
import bsi_trap.constant.constant as const
from bsi_trap.evolve.numeric import _safe_eccentricity, _is_physical_state, _scalar

Evolparams = BBHEvolver(rlist=None, sigmalist=None, hlist=None, cslist=None)


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
    sigma = np.pi *bmax**2 * np.max(focus, 1.0)

    rate = number_density * sigma * v_rel
    return max(float(rate), 0.0)

def bsi_rate(m1:float, m2:float, a:float, trap_context:dict)-> Dict[float, float]:
    """Return the trap-local weak and strong BSI rates in s^-1."""

    if trap_context is None:
        return 0.0, 0.0

    number_density = _scalar(trap_context.get("number_density", 0.0), 0.0)

    v_rel = _scalar(trap_context.get("v_rel", np.nan), np.nan)
    m3 = _scalar(trap_context.get("perturber_mass", np.nan), np.nan)

    if not np.isfinite(number_density) or number_density <= 0.0:
        return 0.0, 0.0
    if not np.isfinite(v_rel) or v_rel <= 0.0:
        return 0.0, 0.0
    if not np.isfinite(m3) or m3 <= 0.0:
        return 0.0, 0.0     
    if not np.isfinite(a) or a <= 0.0:
        return 0.0, 0.0

    bweak = Evolparams.bsi_eta_weak * a
    bstrong = Evolparams.bsi_eta_strong * a

    focusing_weak = focusing_factor(m1, m2, m3, bweak, v_rel)
    rate_weak = rate_compute(focusing_weak, bweak, number_density, v_rel)

    focusing_strong = focusing_factor(m1, m2, m3, bstrong, v_rel)
    rate_strong = rate_compute(focusing_strong, bstrong, number_density, v_rel)

    return float(rate_weak-rate_strong), float(rate_strong)


def single_weak_bsi(m1, m2, a, e, th1, th2, trap_context):
    """Apply one weak trap encounter, conserving inner angular momentum."""
    m3 = _scalar(trap_context.get("perturber_mass", np.nan), np.nan)
    if not np.isfinite(m3) or m3 <= 0.0:
        return a, e, th1, th2

    bweak = Evolparams.bsi_eta_weak * a

    if not np.isfinite(bweak) or bweak <= a:
        return a, e, th1, th2

    u = np.random.random()
    rp = np.sqrt(a**2 + u * (bweak**2 - a**2))

    if not np.isfinite(rp) or rp <= 0.0:
        return a, e, th1, th2

    m12_msun = (m1 + m2) / const.msun
    eps = (m3 / m12_msun) * (a / rp) ** 2.5
    eps = float(np.clip(eps, 0.0, 0.2))

    e_old = _safe_eccentricity(e)
    sigma_e = Evolparams.bsi_sigma_e0 * eps * max(0.0, 1.0 - e_old**2)
    e_new = e_old + np.random.normal(0.0, sigma_e)
    e_new = float(np.clip(e_new, Evolparams.ecc_floor, Evolparams.ecc_ceiling))

    denom = max(1.0 - e_new**2, 1.0e-8)
    a_new = a * (1.0 - e_old**2) / denom
    if not np.isfinite(a_new) or a_new <= 0.0:
        a_new = a

    sigma_theta = min(0.05, 0.5 * eps)
    th1_new = float(np.clip(th1 + np.random.normal(0.0, sigma_theta), 0.0, np.pi))
    th2_new = float(np.clip(th2 + np.random.normal(0.0, sigma_theta), 0.0, np.pi))

    return float(a_new), e_new, th1_new, th2_new


def single_strong_bsi(state, spin_state, trap_context):
    """Placeholder for future strong BSI physics; currently a no-op."""
    return state, spin_state


def apply_weak_bsi(m1, m2, a, e, th1, th2, dt, trap_context):
    """Monte Carlo weak BSI event step for one integration timestep."""
    rate,_ = bsi_rate(m1, m2, a, trap_context)
    if rate <= 0.0 or not np.isfinite(rate):
        return a, e, th1, th2, False

    p_event = 1.0 - np.exp(-rate * dt)
    p_event = min(float(p_event), 0.5)
    if np.random.random() >= p_event:
        return a, e, th1, th2, False

    a, e, th1, th2 = single_weak_bsi(m1, m2, a, e, th1, th2, trap_context)
    return a, e, th1, th2, True


def apply_bsi(BBH, flags, i, a, e, m1, m2, dt, bsi_active=True):
    """Apply trap-local weak BSI without changing masses or destroying BBHs."""
    if not bsi_active:
        return a, e, m1, m2, None

    if Evolparams.trap_bsi_context is None:
        return a, e, m1, m2, None

    if getattr(flags, "hard", np.zeros(i + 1, dtype=int))[i] != 1:
        return a, e, m1, m2, None

    if getattr(flags, "trap_reached_flag", np.zeros(i + 1, dtype=int))[i] != 1:
        return a, e, m1, m2, None

    if flags.stop_reason_flag[i] != const.STOP_NONE:
        return a, e, m1, m2, None

    if not (_is_physical_state(a, e) and np.isfinite(dt) and dt > 0.0):
        return a, e, m1, m2, None

    a_new, e_new, th1_new, th2_new, event_happened = apply_weak_bsi(
        m1,
        m2,
        a,
        e,
        BBH.th1[i],
        BBH.th2[i],
        dt,
        Evolparams.trap_bsi_context,
    )
    if not event_happened:
        return a, e, m1, m2, None

    BBH.th1[i] = th1_new
    BBH.th2[i] = th2_new
    return a_new, e_new, m1, m2, None


_apply_bsi_step = apply_bsi


def bsi_timescale():
    """Store t_BSI = 1/Gamma_BSI for hard binaries active at the trap."""
    pass
