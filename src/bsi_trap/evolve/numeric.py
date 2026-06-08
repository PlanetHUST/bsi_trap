import numpy as np
import logging

from bsi_trap.classes.classes import BBHEvolver

Evolparams = BBHEvolver(rlist=None, sigmalist=None, hlist=None, cslist=None)


def safe_eccentricity(e):
    """Clamp roundoff-level eccentricity excursions to the physical range."""
    if not np.isfinite(e):
        logging.warning(f"Non-finite eccentricity detected: e={e}")
        return np.nan
    return float(np.clip(e, 0.0, Evolparams.ecc_ceiling))


def safe_sma(a):
    """Return a finite positive semi-major axis or NaN for invalid states."""
    if not np.isfinite(a) or a <= 0.0:
        logging.warning(f"Non-finite or non-positive semi-major axis detected: a={a}")
        return np.nan
    return float(a)


def is_physical_state(a, e):
    return np.isfinite(a) and a > 0.0 and np.isfinite(e) and 0.0 <= e < 1.0


def adaptive_timestep(a, e, ka, ke, h):
    # guarantee physical parameters
    if not (
        np.isfinite(a)
        and a > 0.0
        and np.isfinite(e)
        and np.isfinite(ka)
        and np.isfinite(ke)
        and np.isfinite(h)
        and h > 0.0
    ):
        logging.warning(
            f"Non-physical state detected: a={a}, e={e}, ka={ka}, ke={ke}, h={h}"
        )
        return np.nan

    while np.abs(ka * h / a) > 2 * Evolparams.tol_a:
        h /= 10
    if np.abs(ke * h) > 2 * Evolparams.tol_e:
        h /= 2

    e_for_rel = max(abs(e), Evolparams.ecc_floor)

    if (
        np.abs(ka * h / a) < 0.1 * Evolparams.tol_a
        and np.abs(ke * h / e_for_rel) < 0.1 * Evolparams.tol_e
    ):
        h *= 5
    return h


def rk4_step(m1, m2, a, e, h, ode_func, *ode_args):
    """4th order Runge-Kutta integration step"""
    if not is_physical_state(a, e):
        logging.warning(f"Attempting RK4 step from non-physical state: a={a}, e={e}")
        return np.nan, np.nan, np.nan

    h_try = h

    for _ in range(30):
        k1_a, k1_e = ode_func(m1, m2, a, e, *ode_args)
        h_step = adaptive_timestep(a, e, k1_a, k1_e, h_try)
        if not np.isfinite(h_step) or h_step <= 0.0:
            h_try *= 0.1
            continue

        a2 = a + 0.5 * h_step * k1_a
        e2 = e + 0.5 * h_step * k1_e
        if not is_physical_state(a2, e2):
            h_try = 0.1 * h_step
            continue
        k2_a, k2_e = ode_func(m1, m2, a2, e2, *ode_args)
        if not np.all(np.isfinite([k2_a, k2_e])):
            h_try = 0.1 * h_step
            continue

        a3 = a + 0.5 * h_step * k2_a
        e3 = e + 0.5 * h_step * k2_e
        if not is_physical_state(a3, e3):
            h_try = 0.1 * h_step
            continue
        k3_a, k3_e = ode_func(m1, m2, a3, e3, *ode_args)
        if not np.all(np.isfinite([k3_a, k3_e])):
            h_try = 0.1 * h_step
            continue

        a4 = a + h_step * k3_a
        e4 = e + h_step * k3_e
        if not is_physical_state(a4, e4):
            h_try = 0.1 * h_step
            continue
        k4_a, k4_e = ode_func(m1, m2, a4, e4, *ode_args)
        if not np.all(np.isfinite([k4_a, k4_e])):
            h_try = 0.1 * h_step
            continue

        anew = a + (h_step / 6) * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)
        enew = e + (h_step / 6) * (k1_e + 2 * k2_e + 2 * k3_e + k4_e)
        if is_physical_state(anew, enew):
            return float(anew), float(enew), float(h_step)

        h_try = 0.1 * h_step

    return np.nan, np.nan, np.nan


def para_evo(m1, m2, a, e, h, ode_func, *ode_args):
    # Take two half steps
    a_half, e_half, hhalf = rk4_step(m1, m2, a, e, h / 2, ode_func, *ode_args)
    if not (is_physical_state(a_half, e_half) and np.isfinite(hhalf) and hhalf > 0.0):
        return np.nan, np.nan, np.nan
    h = 2 * hhalf
    # print(f'the first half parameter computed: a={a_half:.3e}, e={e_half:.3e}')
    a_full, e_full, hhalf = rk4_step(m1, m2, a_half, e_half, h / 2, ode_func, *ode_args)
    if not (is_physical_state(a_full, e_full) and np.isfinite(hhalf) and hhalf > 0.0):
        return np.nan, np.nan, np.nan
    return a_full, e_full, 2 * hhalf


def interp_profile(radius, values, target, default=np.nan):
    "interpolate disk profile based on radius vs values to get value on target radius"
    radius = np.asarray(radius, dtype=float)
    values = np.asarray(values, dtype=float)
    # sort finite mask
    finite = np.isfinite(radius) & np.isfinite(values)
    if not np.any(finite) or not np.isfinite(target):
        return float(default)
    radius = radius[finite]
    values = values[finite]
    order = np.argsort(radius)
    return float(np.interp(float(target), radius[order], values[order]))


def scalar(value, default=np.nan):
    """Read a scalar from code paths that sometimes store one-item arrays."""
    try:
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            return default
        return float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return default


_safe_eccentricity = safe_eccentricity
_safe_sma = safe_sma
_is_physical_state = is_physical_state
_scalar = scalar
