"""Physical pieces for the compact BSI-trap demo.

The active physics story is intentionally short:

1. solve one AGN disk with pAGN,
2. sample embedded remnant BHs,
3. assign migration arrival times at one trap,
4. pair reached singles into first-generation BBHs,
5. evolve BBHs through gas/GW hardening, BSI perturbations, and merger.
"""

from bsi_trap.physics.disk import build_disk_profile
from bsi_trap.physics.evolution import (
    apply_bsi_perturbation,
    binary_single_delay_s,
    build_binary_population,
    evolve_kernal,
    gas_hardening_time_s,
    gw_merger_time_s,
    merge_remnant,
    merger_radius_cm,
)
from bsi_trap.physics.pairing import pairing_kernal, pair_delay_s
from bsi_trap.physics.population import construct_arrivals, sample_embedded_population

__all__ = [
    "apply_bsi_perturbation",
    "binary_single_delay_s",
    "build_disk_profile",
    "build_binary_population",
    "construct_arrivals",
    "pairing_kernal",
    "evolve_kernal",
    "gas_hardening_time_s",
    "gw_merger_time_s",
    "merge_remnant",
    "merger_radius_cm",
    "pair_delay_s",
    "sample_embedded_population",
]
