"""Small data containers for the compact demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DemoConfig:
    smbh_mass_msun: float = 1.0e6
    agn_lifetime_myr: float = 10.0
    population_size: int = 500
    seed: int = 42
    disk_model: str = "sirko"
    disk_resolution: int = 256
    alpha: float = 0.01
    catalog_rows: int | None = 50_000
    migration_time_myr: float = 0.7
    pair_delay_scale: float = 1.0
    initial_bbh_sma_au: float = 0.01
    initial_bbh_eccentricity: float = 0.3
    bsi_cross_section_sma: float = 3.0
    bsi_hardening_fraction: float = 0.15
    bsi_eccentricity_pump: float = 0.2


@dataclass(frozen=True)
class DiskProfile:
    radius_cm: np.ndarray
    sigma_g_cm2: np.ndarray
    cs_cm_s: np.ndarray
    h_over_r: np.ndarray
    toomre_q: np.ndarray
    r_trap_cm: float
    r_in_cm: float
    r_out_cm: float

    def closest_index(self, radius_cm: float) -> int:
        return int(np.argmin(np.abs(self.radius_cm - radius_cm)))

    @property
    def trap_index(self) -> int:
        return self.closest_index(self.r_trap_cm)

    @property
    def trap_scale_height_cm(self) -> float:
        return float(self.h_over_r[self.trap_index] * self.r_trap_cm)

    @property
    def trap_sigma_g_cm2(self) -> float:
        return float(self.sigma_g_cm2[self.trap_index])

    @property
    def trap_cs_cm_s(self) -> float:
        return float(self.cs_cm_s[self.trap_index])

    @property
    def trap_q(self) -> float:
        return float(self.toomre_q[self.trap_index])


@dataclass(frozen=True)
class EmbeddedPopulation:
    mass_msun: np.ndarray
    initial_radius_cm: np.ndarray

    @property
    def size(self) -> int:
        return int(self.mass_msun.size)


@dataclass(frozen=True)
class ArrivalTimeline:
    arrival_s: np.ndarray
    reached_mask: np.ndarray
    order: np.ndarray

    @property
    def reached_count(self) -> int:
        return int(np.count_nonzero(self.reached_mask))


@dataclass(frozen=True)
class PairingTimeline:
    first_idx: np.ndarray
    second_idx: np.ndarray
    later_arrival_s: np.ndarray
    pair_delay_s: np.ndarray
    formation_s: np.ndarray

    @property
    def size(self) -> int:
        return int(self.formation_s.size)


@dataclass(frozen=True)
class BinaryPopulation:
    first_idx: np.ndarray
    second_idx: np.ndarray
    m1_msun: np.ndarray
    m2_msun: np.ndarray
    semi_major_axis_cm: np.ndarray
    eccentricity: np.ndarray
    formation_s: np.ndarray

    @property
    def size(self) -> int:
        return int(self.formation_s.size)


@dataclass(frozen=True)
class EvolutionTimeline:
    binary_idx: np.ndarray
    start_s: np.ndarray
    gas_delay_s: np.ndarray
    gw_delay_s: np.ndarray
    bsi_event_s: np.ndarray
    delay_s: np.ndarray
    end_s: np.ndarray
    merger_s: np.ndarray
    merged: np.ndarray
    final_mass_msun: np.ndarray
    final_spin: np.ndarray
    channel: np.ndarray

    @property
    def size(self) -> int:
        return int(self.start_s.size)

    @property
    def merger_count(self) -> int:
        return int(np.count_nonzero(self.merged))

    @property
    def bsi_count(self) -> int:
        return int(np.count_nonzero(np.isfinite(self.bsi_event_s)))


@dataclass(frozen=True)
class DemoResult:
    config: DemoConfig
    disk: DiskProfile
    population: EmbeddedPopulation
    arrivals: ArrivalTimeline
    pairing: PairingTimeline
    binaries: BinaryPopulation
    evolution: EvolutionTimeline
    paired_mask: np.ndarray

    @property
    def single_count(self) -> int:
        singles = self.arrivals.reached_mask & ~self.paired_mask
        return int(np.count_nonzero(singles))


__all__ = [
    "ArrivalTimeline",
    "BinaryPopulation",
    "DemoConfig",
    "DemoResult",
    "DiskProfile",
    "EmbeddedPopulation",
    "EvolutionTimeline",
    "PairingTimeline",
]
