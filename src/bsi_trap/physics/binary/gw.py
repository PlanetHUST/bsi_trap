"""GW-only BBH hardening and its wrapper around the shared orbit loop."""

import numpy as np
import logging

from bsi_trap.simulation.state.models import (
    BBHEvolverParams,
    BBH_properties,
    AGN_properties,
)
import bsi_trap.constant.constant as const
from bsi_trap.physics.binary.evolution import evolve_binary_orbit
from bsi_trap.physics.binary import numeric


def ode_gw_only(Evolparams, m1, m2, a, e):
    """Return Peters (1964) GW hardening rates."""
    a = numeric.safe_sma(Evolparams, a)
    e = numeric.safe_eccentricity(Evolparams, e)
    if not numeric.is_physical_state(Evolparams, a, e):
        logging.warning(f"Non-physical state detected in GW evolution: a={a}, e={e}")
        return np.nan, np.nan

    factor = const.G3 * m1 * m2 * (m1 + m2) / (const.c5 * a**3 * (1 - e**2) ** 3.5)

    da_dt = -64 / 5 * factor * (1 + 73 / 24 * e**2 + 37 / 96 * e**4)
    de_dt = -304 / 15 * e * factor / (a * (1 - e**2) ** -1) * (1 + 121 / 304 * e**2)

    return da_dt, de_dt


def evolve_gw_only(
    Evolparams: BBHEvolverParams,
    bbh: BBH_properties,
    agn: AGN_properties,
    i: int,
    t_ini_s: float,
    *,
    bsi_enabled: bool = False,
    segment_stop_s: float | None = None,
) -> float:
    """Evolve one BBH with GW emission only and return elapsed time."""
    return evolve_binary_orbit(
        Evolparams,
        bbh,
        agn,
        i,
        t_ini_s=t_ini_s,
        initial_step_fraction=Evolparams.gw_initial_step_fraction,
        ode_func=ode_gw_only,
        bsi_enabled=bsi_enabled,
        segment_stop_s=segment_stop_s,
    )
