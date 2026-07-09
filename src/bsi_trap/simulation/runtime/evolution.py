from __future__ import annotations

import numpy as np

from bsi_trap.simulation.state.models import (
    AGN_properties,
    BBHEvolverParams,
    DiskProfile,
)
from bsi_trap.simulation.config.model_config import ModelConfig
from bsi_trap.constant.constant import Myr
from bsi_trap.simulation.runtime.state import EventLog_binary, StateLog
from bsi_trap.physics.binary.pairing import (
    conjunction_rate,
    is_strong_gravity_encounter,
)

__all__ = [
    "EventLog_binary",
    "StateLog",
    "evolve_trap_runtime",
]


def evolve_trap_runtime(
    *,
    Evolparams: BBHEvolverParams,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    initial_state: StateLog,
    rng_pair_up: np.random.Generator,
    rng_orbit: np.random.Generator,
    rng_spin: np.random.Generator,
):
    """Evolve the trap population with one top-level physical event clock."""

    end_time_s = float(agn.lifetime_myr * Myr)

    t_reach_s = arrive_pop.t_reach_s
    m_msun = arrive_pop.m_msun

    pop_state = initial_state

    next_arrive_ind = np.argpartition(t_reach_s, 4)[:4]
    mask = np.argsort(t_reach_s[next_arrive_ind])
    next_arrive_treach_s = t_reach_s[mask]
    next_arrive_m_msun = m_msun[mask]

    # pair-up -> add one bbh, delete two sbh
    # evolve -> add eventlog
    # merge -> delete one bbh
