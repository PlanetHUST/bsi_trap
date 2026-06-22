"""Establish and validate the initial single-BH and BBH trap condition."""

from __future__ import annotations

import numpy as np

from bsi_trap.classes.models import (
    AGN_properties,
    BBH_properties,
    DiskProfile,
    Embedded_BH,
    Single_BH_properties,
)
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.constant.constant import Myr
from bsi_trap.evolve.trap_migration import (
    resolve_mass_dependent_trap_migration,
)
from bsi_trap.evolve.trap_context import build_trap_context
from bsi_trap.evolve.pairing import construct_pairing_up_timeline


def _validate_embedded_population(ebh: Embedded_BH) -> None:
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


def _validate_trap_state(ebh: Embedded_BH, end_time_s: float) -> None:
    """Verify that pairing flags, completion times, and population counts agree."""
    paired = ebh.trap_flag >= 0
    singles = ebh.trap_flag == -2
    unreached = ebh.trap_flag == -1
    if not np.all(paired | singles | unreached):
        raise ValueError("Embedded-BH trap flags contain an unknown state")
    if np.any(np.isfinite(ebh.t_pair_s[singles | unreached])):
        raise ValueError("Only paired BHs may have a pair-completion time")

    for first in np.flatnonzero(paired):
        second = int(ebh.trap_flag[first])
        if second >= ebh.num or ebh.trap_flag[second] != first:
            raise ValueError("Embedded-BH pairing flags must be symmetric indices")
        formation_time_s = ebh.t_pair_s[first]
        if not np.isfinite(formation_time_s) or formation_time_s > end_time_s:
            raise ValueError("Paired BHs must form before the trap end time")
        if not np.isclose(formation_time_s, ebh.t_pair_s[second]):
            raise ValueError("Both members of a BBH pair must share one formation time")
        if formation_time_s < max(ebh.t_reach_s[first], ebh.t_reach_s[second]):
            raise ValueError("BBH formation cannot precede either component arrival")


def construct_initial_trap_condition(
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    ebh: Embedded_BH,
    rng_pair_up: np.random.Generator,
) -> tuple[BBH_properties, Single_BH_properties]:
    """Build and validate the first-generation BBH and single-BH populations.

    This is the initial-condition boundary of the workflow:

    1. validate the embedded population;
    2. calculate migration and trap arrival times;
    3. build explicit local trap geometry;
    4. construct the stochastic pairing timeline; and
    5. convert the verified trap state into BBH and single-BH containers.
    """
    _validate_embedded_population(ebh)

    end_time_s = agn.lifetime_myr * Myr
    if not np.isfinite(end_time_s) or end_time_s <= 0.0:
        raise ValueError("AGN lifetime must define a finite, positive end time")

    migration = resolve_mass_dependent_trap_migration(
        agn=agn,
        disk=disk,
        ebh=ebh,
    )
    migration_times_s = migration.migration_times_s
    if migration_times_s.shape != (ebh.num,):
        raise ValueError("Migration solver returned an invalid population shape")
    if np.any(np.isnan(migration_times_s)) or np.any(migration_times_s < 0.0):
        raise ValueError("Migration times must be nonnegative finite values or infinity")

    ebh.t_mig_s = migration_times_s
    ebh.t_reach_s = ebh.t_form_s + migration_times_s
    ebh.r_trap_target_cm = migration.target_radii_cm

    reached_mask = migration.stable_mask & (ebh.t_reach_s < end_time_s)
    trap_context = build_trap_context(disk, model_config)
    ebh.trap_width_cm = trap_context.width_cm
    construct_pairing_up_timeline(
        embedded_bh=ebh,
        reached_mask=reached_mask,
        trap_context=trap_context,
        model_config=model_config,
        disk=disk,
        smbh_mass_msun=agn.M_SMBH_msun,
        end_time_s=end_time_s,
        rng_pair_up=rng_pair_up,
    )

    _validate_trap_state(ebh, end_time_s)
    bbh = init_bbh(ebh)
    sbh = init_sbh(ebh)
    reached_count = int(np.count_nonzero(reached_mask))
    if 2 * bbh.num + sbh.num != reached_count:
        raise ValueError("Initial trap population does not conserve reached BHs")
    return bbh, sbh


def embedded_to_BBH(
    model_config: ModelConfig,
    agn: AGN_properties,
    disk: DiskProfile,
    ebh: Embedded_BH,
    rng_pair_up: np.random.Generator,
) -> tuple[BBH_properties, Single_BH_properties]:
    """Compatibility wrapper for :func:`construct_initial_trap_condition`."""
    return construct_initial_trap_condition(
        model_config=model_config,
        agn=agn,
        disk=disk,
        ebh=ebh,
        rng_pair_up=rng_pair_up,
    )


def init_bbh(ebh: Embedded_BH) -> BBH_properties:
    """Create one BBH entry per symmetric pairing recorded on ``ebh``."""
    first_idx: list[int] = []
    second_idx: list[int] = []

    for first, second in enumerate(ebh.trap_flag):
        if second < 0:
            continue
        if second >= ebh.num or ebh.trap_flag[second] != first:
            raise ValueError("Embedded-BH pairing flags must be symmetric indices")
        if first < second:
            first_idx.append(first)
            second_idx.append(int(second))

    first_idx_array = np.asarray(first_idx, dtype=int)
    second_idx_array = np.asarray(second_idx, dtype=int)
    bbh = BBH_properties(first_idx_array.size)
    if bbh.num == 0:
        return bbh

    first_mass_msun = ebh.m_msun[first_idx_array]
    second_mass_msun = ebh.m_msun[second_idx_array]
    first_is_primary = first_mass_msun >= second_mass_msun

    primary_idx = np.where(first_is_primary, first_idx_array, second_idx_array)
    secondary_idx = np.where(first_is_primary, second_idx_array, first_idx_array)
    formation_times_s = ebh.t_pair_s[first_idx_array]

    if np.any(~np.isfinite(formation_times_s)):
        raise ValueError("Every selected BBH pair must have a finite formation time")
    if not np.allclose(formation_times_s, ebh.t_pair_s[second_idx_array]):
        raise ValueError("Both members of a BBH pair must share one formation time")

    bbh.idi = primary_idx
    bbh.m1_msun = ebh.m_msun[primary_idx]
    bbh.m2_msun = ebh.m_msun[secondary_idx]
    bbh.t_ini_s = formation_times_s
    return bbh


def init_sbh(ebh: Embedded_BH) -> Single_BH_properties:
    """Create the single-BH population from reached, unpaired trap objects."""
    single_idx = np.flatnonzero(ebh.trap_flag == -2)
    sbh = Single_BH_properties(single_idx.size)
    sbh.m_msun = ebh.m_msun[single_idx]
    sbh.t_ini_s = ebh.t_reach_s[single_idx]
    return sbh
