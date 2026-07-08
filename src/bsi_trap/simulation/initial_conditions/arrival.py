from __future__ import annotations

import numpy as np

from bsi_trap.constant.constant import Myr
from bsi_trap.physics.trap.context import build_trap_context
from bsi_trap.physics.trap.migration import resolve_migration
from bsi_trap.simulation.config.model_config import ModelConfig
from bsi_trap.simulation.initial_conditions.conversion import init_sbh
from bsi_trap.simulation.initial_conditions.embedded_bh import embedded_BH_initialize
from bsi_trap.simulation.initial_conditions.population import TrapArrivalState
from bsi_trap.simulation.state.models import AGN_properties, DiskProfile, Embedded_BH


def make_trap_arrival_state(
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    rng_nsc: np.random.Generator,
) -> TrapArrivalState:
    """Build embedded BHs and live trap singles before any pairing occurs."""

    embedded_bh = embedded_BH_initialize(agn, disk, rng_nsc)
    prepare_trap_arrival_state(
        model_config=model_config,
        agn=agn,
        disk=disk,
        ebh=embedded_bh,
    )
    return TrapArrivalState(
        embedded_bh=embedded_bh,
        singles=init_sbh(embedded_bh),
    )


def prepare_trap_arrival_state(
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    ebh: Embedded_BH,
) -> np.ndarray:
    """Resolve migration and mark reached trap objects as live singles."""
    _validate_embedded(ebh)

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

    ebh.t_mig_s = migration_times_s
    ebh.t_reach_s = ebh.t_form_s + migration_times_s
    ebh.r_trap_target_cm = migration_results.target_radii_cm

    reached_mask = migration_results.stable_mask & (ebh.t_reach_s < end_time_s)
    trap_context = build_trap_context(disk, model_config)
    ebh.trap_width_cm = trap_context.width_cm

    ebh.trap_flag.fill(-1)
    ebh.trap_flag[reached_mask] = -2
    ebh.t_pair_s.fill(np.nan)
    return reached_mask


def _validate_embedded(ebh: Embedded_BH) -> None:
    """Check the array contract required before migration and pairing."""
    expected_shape = (ebh.num,)
    for name in ("m_msun", "t_form_s", "R_cm"):
        values = getattr(ebh, name)
        if values.shape != expected_shape:
            raise ValueError(f"Embedded-BH {name} must have shape {expected_shape}")

    if np.any(~np.isfinite(ebh.m_msun)) or np.any(ebh.m_msun <= 0.0):
        raise ValueError("Embedded-BH masses must be finite and positive")
    if np.any(~np.isfinite(ebh.t_form_s)) or np.any(ebh.t_form_s < 0.0):
        raise ValueError("Embedded-BH formation times must be finite and nonnegative")
    if np.any(~np.isfinite(ebh.R_cm)) or np.any(ebh.R_cm <= 0.0):
        raise ValueError("Embedded-BH initial radii must be finite and positive")
