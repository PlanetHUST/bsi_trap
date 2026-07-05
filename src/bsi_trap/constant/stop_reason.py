from __future__ import annotations

from dataclasses import dataclass


# These integer codes are persisted to HDF5 as ``stop_reason_flag``. Keep the
# numeric values stable once output files have been produced, because plotting
# and post-processing read old files by code.
class StopReasons:
    # no terminal stop
    STOP_NONE = 0

    # trap population and pairing
    STOP_SOFT_BINARY = 1
    STOP_GLOBAL_TRAP_OVERLOAD = 2
    STOP_SINGLE_TRAP_OVERLOAD = 3
    STOP_TRAP_NOT_REACHED = 4

    # binary evolution integration
    STOP_INITIAL_BINARY_TOO_TIGHT = 5
    STOP_HUBBLE_LIMIT = 6
    STOP_INTEGRATION_LIMIT = 7
    STOP_AGN_LIFETIME = 11
    STOP_NON_PHYSICAL_ORBIT = 12
    STOP_TIME_BUDGET = 13
    STOP_POST_AGN_ISOLATED_MERGER = 14
    STOP_GENERATION_WALLTIME_LIMIT = 15

    # hierarchical-remnant outcome
    STOP_REMNANT_TRAP_OVERLOAD = 8
    STOP_GW_ESCAPE = 9
    STOP_DISK_RETENTION = 10


class StopReasonType:
    NONE = "none"
    TRAP_POPULATION = "trap_population"
    BINARY_EVOLUTION = "binary_evolution"
    HIERARCHICAL_REMNANT = "hierarchical_remnant"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StopReasonInfo:
    code: int
    name: str
    category: str
    description: str


STOP_REASON_INFO: dict[int, StopReasonInfo] = {
    StopReasons.STOP_NONE: StopReasonInfo(
        StopReasons.STOP_NONE,
        "STOP_NONE",
        StopReasonType.NONE,
        "No terminal stop condition was triggered.",
    ),
    StopReasons.STOP_SOFT_BINARY: StopReasonInfo(
        StopReasons.STOP_SOFT_BINARY,
        "STOP_SOFT_BINARY",
        StopReasonType.TRAP_POPULATION,
        "The candidate binary was too soft to keep as a BBH.",
    ),
    StopReasons.STOP_GLOBAL_TRAP_OVERLOAD: StopReasonInfo(
        StopReasons.STOP_GLOBAL_TRAP_OVERLOAD,
        "STOP_GLOBAL_TRAP_OVERLOAD",
        StopReasonType.TRAP_POPULATION,
        "The trap population exceeded the configured global capacity.",
    ),
    StopReasons.STOP_SINGLE_TRAP_OVERLOAD: StopReasonInfo(
        StopReasons.STOP_SINGLE_TRAP_OVERLOAD,
        "STOP_SINGLE_TRAP_OVERLOAD",
        StopReasonType.TRAP_POPULATION,
        "Single-object trapping exceeded the configured local capacity.",
    ),
    StopReasons.STOP_TRAP_NOT_REACHED: StopReasonInfo(
        StopReasons.STOP_TRAP_NOT_REACHED,
        "STOP_TRAP_NOT_REACHED",
        StopReasonType.TRAP_POPULATION,
        "The object did not reach the migration trap in time.",
    ),
    StopReasons.STOP_INITIAL_BINARY_TOO_TIGHT: StopReasonInfo(
        StopReasons.STOP_INITIAL_BINARY_TOO_TIGHT,
        "STOP_INITIAL_BINARY_TOO_TIGHT",
        StopReasonType.BINARY_EVOLUTION,
        "The initialized BBH was already too close to the merger boundary.",
    ),
    StopReasons.STOP_HUBBLE_LIMIT: StopReasonInfo(
        StopReasons.STOP_HUBBLE_LIMIT,
        "STOP_HUBBLE_LIMIT",
        StopReasonType.BINARY_EVOLUTION,
        "The BBH evolution exceeded the Hubble-time limit.",
    ),
    StopReasons.STOP_INTEGRATION_LIMIT: StopReasonInfo(
        StopReasons.STOP_INTEGRATION_LIMIT,
        "STOP_INTEGRATION_LIMIT",
        StopReasonType.BINARY_EVOLUTION,
        "The numerical integrator hit an invalid or unresolved state.",
    ),
    StopReasons.STOP_AGN_LIFETIME: StopReasonInfo(
        StopReasons.STOP_AGN_LIFETIME,
        "STOP_AGN_LIFETIME",
        StopReasonType.BINARY_EVOLUTION,
        "The BBH reached the end of the AGN active lifetime.",
    ),
    StopReasons.STOP_NON_PHYSICAL_ORBIT: StopReasonInfo(
        StopReasons.STOP_NON_PHYSICAL_ORBIT,
        "STOP_NON_PHYSICAL_ORBIT",
        StopReasonType.BINARY_EVOLUTION,
        "The binary orbit had non-finite or non-physical parameters.",
    ),
    StopReasons.STOP_TIME_BUDGET: StopReasonInfo(
        StopReasons.STOP_TIME_BUDGET,
        "STOP_TIME_BUDGET",
        StopReasonType.BINARY_EVOLUTION,
        "The configured physical time budget was exhausted.",
    ),
    StopReasons.STOP_POST_AGN_ISOLATED_MERGER: StopReasonInfo(
        StopReasons.STOP_POST_AGN_ISOLATED_MERGER,
        "STOP_POST_AGN_ISOLATED_MERGER",
        StopReasonType.BINARY_EVOLUTION,
        "The BBH would merge only after the AGN disk phase.",
    ),
    StopReasons.STOP_GENERATION_WALLTIME_LIMIT: StopReasonInfo(
        StopReasons.STOP_GENERATION_WALLTIME_LIMIT,
        "STOP_GENERATION_WALLTIME_LIMIT",
        StopReasonType.BINARY_EVOLUTION,
        "The code wall-time guard stopped the generation loop.",
    ),
    StopReasons.STOP_REMNANT_TRAP_OVERLOAD: StopReasonInfo(
        StopReasons.STOP_REMNANT_TRAP_OVERLOAD,
        "STOP_REMNANT_TRAP_OVERLOAD",
        StopReasonType.HIERARCHICAL_REMNANT,
        "The merger remnant is retained but too massive for the trap model.",
    ),
    StopReasons.STOP_GW_ESCAPE: StopReasonInfo(
        StopReasons.STOP_GW_ESCAPE,
        "STOP_GW_ESCAPE",
        StopReasonType.HIERARCHICAL_REMNANT,
        "The merger remnant is ejected by gravitational-wave recoil.",
    ),
    StopReasons.STOP_DISK_RETENTION: StopReasonInfo(
        StopReasons.STOP_DISK_RETENTION,
        "STOP_DISK_RETENTION",
        StopReasonType.HIERARCHICAL_REMNANT,
        "The merger remnant remains bound to the disk/trap environment.",
    ),
}


def stop_reason_info(code: int) -> StopReasonInfo:
    return STOP_REASON_INFO.get(
        int(code),
        StopReasonInfo(
            int(code),
            f"UNKNOWN_{int(code)}",
            StopReasonType.UNKNOWN,
            "Unknown stop-reason code.",
        ),
    )


def stop_reason_name(code: int) -> str:
    return stop_reason_info(code).name


def stop_reason_type(code: int) -> str:
    return stop_reason_info(code).category
