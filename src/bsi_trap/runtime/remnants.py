from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from bsi_trap.classes.models import (
    AGN_properties,
    BBH_properties,
    DiskProfile,
    Single_BH_properties,
)
from bsi_trap.constant import constant as const
from bsi_trap.constant.stop_reason import StopReasons
from bsi_trap.evolve.physical_quantity import migration_threshold
from bsi_trap.functions import merger_properties as merger
from bsi_trap.functions.nsc_population import mass_enclosed_NSC
from bsi_trap.timeline.models import BoolArray, FloatArray, IndexArray


@dataclass(frozen=True)
class HierarchicalMergerLedger:
    """Population-level ledger for merger remnants that may seed later BBHs."""

    source_binary_idx: IndexArray
    merger_s: FloatArray
    remnant_mass_msun: FloatArray
    remnant_spin: FloatArray
    kick_velocity_cgs: FloatArray
    outer_semimajor_cm: FloatArray
    retained: BoolArray
    escaped: BoolArray
    trap_overloaded: BoolArray
    next_generation_singles: Single_BH_properties

    @property
    def size(self) -> int:
        return int(self.source_binary_idx.size)

    @property
    def retained_count(self) -> int:
        return int(np.count_nonzero(self.retained))

    @property
    def escaped_count(self) -> int:
        return int(np.count_nonzero(self.escaped))

    @property
    def trap_overloaded_count(self) -> int:
        return int(np.count_nonzero(self.trap_overloaded))

    @property
    def recycled_count(self) -> int:
        return int(self.next_generation_singles.num)

    @classmethod
    def empty(cls) -> "HierarchicalMergerLedger":
        return cls(
            source_binary_idx=np.empty(0, dtype=np.intp),
            merger_s=np.empty(0, dtype=np.float64),
            remnant_mass_msun=np.empty(0, dtype=np.float64),
            remnant_spin=np.empty(0, dtype=np.float64),
            kick_velocity_cgs=np.empty(0, dtype=np.float64),
            outer_semimajor_cm=np.empty(0, dtype=np.float64),
            retained=np.empty(0, dtype=bool),
            escaped=np.empty(0, dtype=bool),
            trap_overloaded=np.empty(0, dtype=bool),
            next_generation_singles=Single_BH_properties(0),
        )


def _enclosed_mass_msun(agn: AGN_properties, disk: DiskProfile) -> float:
    return float(
        agn.M_SMBH_msun
        + mass_enclosed_NSC(
            agn.M_SMBH_msun,
            disk.r_trap_cm,
        )
    )


def _retention_orbit(
    agn: AGN_properties,
    disk: DiskProfile,
    kick_velocity_cgs: float,
    rmax_cm: float,
) -> tuple[bool, float]:
    enclosed_mass_g = _enclosed_mass_msun(agn, disk) * const.msun
    v_kep = float(np.sqrt(const.G * enclosed_mass_g / disk.r_trap_cm))
    v_esc = float(np.sqrt(2.0 * const.G * enclosed_mass_g / disk.r_trap_cm))
    remnant_velocity_cgs = kick_velocity_cgs + v_kep

    if remnant_velocity_cgs > v_esc:
        return False, float("inf")

    outer_semimajor_cm = merger.final_semimajor(
        agn,
        disk,
        remnant_velocity_cgs,
    )
    retained = bool(np.isfinite(outer_semimajor_cm) and outer_semimajor_cm <= rmax_cm)
    return retained, outer_semimajor_cm


def _make_next_generation_singles(
    remnant_mass_msun: FloatArray,
    remnant_spin: FloatArray,
    merger_s: FloatArray,
    source_generation: IndexArray,
    recycled: BoolArray,
) -> Single_BH_properties:
    singles = Single_BH_properties(int(np.count_nonzero(recycled)))
    if singles.num == 0:
        return singles

    singles.m_msun = np.asarray(remnant_mass_msun[recycled], dtype=np.float64)
    singles.available_from_s = np.asarray(merger_s[recycled], dtype=np.float64)
    singles.generation = np.asarray(source_generation[recycled], dtype=np.int64)
    recycled_spin = np.asarray(remnant_spin[recycled], dtype=np.float64)
    singles.chi = np.abs(recycled_spin)
    singles.theta = np.where(recycled_spin >= 0.0, 0.0, np.pi).astype(np.float64)
    return singles


def build_hierarchical_mergers(
    *,
    agn: AGN_properties,
    disk: DiskProfile,
    bbh: BBH_properties,
    rng: Generator,
    rmax_cm: float | None = None,
) -> HierarchicalMergerLedger:
    """Compute remnant dynamics and next-generation seeds for merged BBHs.

    The BBH evolution integrator decides whether a binary merges.  This
    function handles the post-merger AGN step: final mass/spin, GW recoil,
    retention in the disk potential, and whether a retained remnant can be
    recycled as a higher-generation single object.
    """
    source_binary_idx = np.flatnonzero(bbh.flags.merger_flag != 0).astype(np.intp)
    if source_binary_idx.size == 0:
        return HierarchicalMergerLedger.empty()

    rmax = float(disk.r_out_cm if rmax_cm is None else rmax_cm)
    migration_limit_msun = float(migration_threshold(disk, agn, disk.r_trap_cm))

    merger_s = np.asarray(
        bbh.t_ini_s[source_binary_idx] + bbh.t_del_s[source_binary_idx],
        dtype=np.float64,
    )
    remnant_mass_msun = np.empty(source_binary_idx.size, dtype=np.float64)
    remnant_spin = np.empty(source_binary_idx.size, dtype=np.float64)
    kick_velocity_cgs = np.empty(source_binary_idx.size, dtype=np.float64)
    outer_semimajor_cm = np.empty(source_binary_idx.size, dtype=np.float64)
    retained = np.zeros(source_binary_idx.size, dtype=bool)
    escaped = np.zeros(source_binary_idx.size, dtype=bool)
    trap_overloaded = np.zeros(source_binary_idx.size, dtype=bool)

    for out_idx, binary_idx in enumerate(source_binary_idx):
        m1 = float(bbh.m1_msun[binary_idx])
        m2 = float(bbh.m2_msun[binary_idx])
        chi1 = float(bbh.chi1[binary_idx])
        chi2 = float(bbh.chi2[binary_idx])
        th1 = float(bbh.th1[binary_idx])
        th2 = float(bbh.th2[binary_idx])

        remnant_mass_msun[out_idx] = merger.final_mass(m1, m2, chi1, chi2)
        remnant_spin[out_idx] = merger.final_spin(chi1, chi2, m1, m2)
        kick_velocity_cgs[out_idx] = merger.relativistic_kick(
            m1,
            m2,
            abs(chi1),
            abs(chi2),
            th1,
            th2,
            rng=rng,
        )
        retained[out_idx], outer_semimajor_cm[out_idx] = _retention_orbit(
            agn,
            disk,
            kick_velocity_cgs[out_idx],
            rmax,
        )
        escaped[out_idx] = not retained[out_idx]
        trap_overloaded[out_idx] = bool(
            retained[out_idx] and remnant_mass_msun[out_idx] > migration_limit_msun
        )

        bbh.mrem_msun[binary_idx] = remnant_mass_msun[out_idx]
        bbh.chirem[binary_idx] = remnant_spin[out_idx]
        bbh.vk_cgs[binary_idx] = kick_velocity_cgs[out_idx]
        bbh.A_cm[binary_idx] = outer_semimajor_cm[out_idx]

        if escaped[out_idx]:
            bbh.flags.GWescape_flag[binary_idx] = 1
            bbh.flags.continue_ng_flag[binary_idx] = 0
            bbh.flags.stop_reason_flag[binary_idx] = StopReasons.STOP_GW_ESCAPE
        elif trap_overloaded[out_idx]:
            bbh.flags.GWescape_flag[binary_idx] = 0
            bbh.flags.continue_ng_flag[binary_idx] = 0
            bbh.flags.stop_reason_flag[binary_idx] = (
                StopReasons.STOP_REMNANT_TRAP_OVERLOAD
            )
        else:
            bbh.flags.GWescape_flag[binary_idx] = 0
            bbh.flags.continue_ng_flag[binary_idx] = 1

    recycled = retained & ~trap_overloaded
    return HierarchicalMergerLedger(
        source_binary_idx=source_binary_idx,
        merger_s=merger_s,
        remnant_mass_msun=remnant_mass_msun,
        remnant_spin=remnant_spin,
        kick_velocity_cgs=kick_velocity_cgs,
        outer_semimajor_cm=outer_semimajor_cm,
        retained=retained,
        escaped=escaped,
        trap_overloaded=trap_overloaded,
        next_generation_singles=_make_next_generation_singles(
            remnant_mass_msun,
            remnant_spin,
            merger_s,
            np.asarray(bbh.generation[source_binary_idx], dtype=np.intp),
            recycled,
        ),
    )
