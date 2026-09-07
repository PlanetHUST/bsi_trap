"""Scientific stages for disks, populations, migration, and binary evolution."""

from bsi_trap.physics.disk import build_disk_profile, build_pagn_disk_profile
from bsi_trap.physics.evolution import evolve_binary_peters
from bsi_trap.physics.migration import migrate_object
from bsi_trap.physics.migration_torque import (
    classify_torque_traps,
    compute_torque_profile,
)
from bsi_trap.physics.population import construct_arrivals, sample_embedded_population

__all__ = [
    "build_disk_profile",
    "build_pagn_disk_profile",
    "classify_torque_traps",
    "compute_torque_profile",
    "construct_arrivals",
    "evolve_binary_peters",
    "migrate_object",
    "sample_embedded_population",
]
