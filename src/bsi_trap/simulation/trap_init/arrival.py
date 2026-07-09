from __future__ import annotations

import numpy as np

from bsi_trap.constant.constant import Myr
from bsi_trap.physics.trap.migration import resolve_migration
from bsi_trap.simulation.state.models import (
    Arrive_timeline,
    AGN_properties,
    DiskProfile,
    Embedded_BH,
)


def construct_arrive_timeline(
    ebh: Embedded_BH, agn: AGN_properties, disk: DiskProfile
) -> Arrive_timeline:

    end_time_s = agn.lifetime_myr * Myr
    if not np.isfinite(end_time_s) or end_time_s <= 0.0:
        raise ValueError("AGN lifetime must define a finite, positive end time")

    migration_results = resolve_migration(
        agn=agn,
        disk=disk,
        ebh=ebh,
    )
    migration_times_s = migration_results.migration_times_s

    if migration_times_s.shape != (ebh.num,):
        raise ValueError("Migration solver returned an invalid population shape")
    if np.any(np.isnan(migration_times_s)) or np.any(migration_times_s < 0.0):
        raise ValueError(
            "Migration times must be nonnegative finite values or infinity"
        )

    pre_reach_time_s = ebh.t_form_s + migration_times_s
    reached_mask = migration_results.stable_mask & (pre_reach_time_s < end_time_s)
    valid_reach_time_s = pre_reach_time_s[reached_mask]
    m_sun = ebh.m_msun[reached_mask]

    return Arrive_timeline(valid_reach_time_s, m_sun)
