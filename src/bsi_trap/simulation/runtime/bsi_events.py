from __future__ import annotations

import numpy as np

from bsi_trap.simulation.state.models import (
    AGN_properties,
    DiskProfile,
    Single_BH_properties,
)
from bsi_trap.simulation.config.model_config import ModelConfig
from bsi_trap.physics.binary.bsi import binary_single_rate_s_inv
from bsi_trap.simulation.runtime.state import ActiveBinary, RuntimeEvent
from bsi_trap.simulation.timeline.models import FloatArray


def active_single_masses_at(
    singles: Single_BH_properties,
    current_time_s: float,
) -> FloatArray:
    active_mask = (
        np.isfinite(singles.available_from_s)
        & (singles.available_from_s <= current_time_s)
        & np.isfinite(singles.m_msun)
        & (singles.m_msun > 0.0)
    )
    return np.asarray(singles.m_msun[active_mask], dtype=np.float64)


def sample_next_bsi_event(
    *,
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    active_binaries: list[ActiveBinary],
    live_singles: Single_BH_properties,
    current_time_s: float,
    step_end_s: float,
    rng_orbit: np.random.Generator,
) -> RuntimeEvent | None:

    if (
        not model_config.bsi_enabled
        or step_end_s <= current_time_s
        or not active_binaries
    ):
        return None

    active_single_masses_msun = active_single_masses_at(
        live_singles,
        current_time_s,
    )

    candidates: list[RuntimeEvent] = []
    for active_index, active in enumerate(active_binaries):
        bbh = active.population.binaries
        for binary_idx in range(bbh.num):
            if bbh.bsi_applied_flag[binary_idx] != 0:
                continue

            existing_delay_s = float(bbh.bsi_timescale_s[binary_idx])
            if np.isfinite(existing_delay_s) and existing_delay_s >= 0.0:
                event_s = float(bbh.t_ini_s[binary_idx] + existing_delay_s)
            else:
                if active_single_masses_msun.size == 0:
                    continue
                rate_s_inv = binary_single_rate_s_inv(
                    model_config,
                    agn,
                    disk,
                    bbh,
                    binary_idx,
                    active_single_masses_msun,
                )
                if not np.isfinite(rate_s_inv) or rate_s_inv <= 0.0:
                    continue
                event_s = float(
                    current_time_s + rng_orbit.exponential(1.0 / rate_s_inv)
                )

            if current_time_s < event_s < step_end_s:
                candidates.append(
                    RuntimeEvent(
                        kind="bsi",
                        time_s=event_s,
                        generation=active.generation,
                        record_index=active.record_index,
                        live_single_count=int(live_singles.num),
                        active_binary_count=len(active_binaries),
                        binary_idx=binary_idx,
                        active_index=active_index,
                    )
                )

    if not candidates:
        return None
    return min(candidates, key=lambda event: event.time_s)


def prime_bsi_event(active: ActiveBinary, event: RuntimeEvent) -> None:
    bbh = active.population.binaries
    event_delay_s = float(event.time_s - bbh.t_ini_s[event.binary_idx])
    if not np.isfinite(event_delay_s) or event_delay_s < 0.0:
        raise ValueError("Runtime BSI event cannot precede BBH formation")
    bbh.bsi_timescale_s[event.binary_idx] = event_delay_s
