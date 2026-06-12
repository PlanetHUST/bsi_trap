# bsi_trap/evolve/timeline.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import numpy as np


@dataclass
class BBHStateSnapshot:
    t_s: float
    m1_msun: float
    m2_msun: float
    sma_cm: float
    ecc: float
    chi1: float
    chi2: float
    th1: float
    th2: float
    A_cm: float
    mrem_msun: float
    chirem: float
    vk_cgs: float

    merger_flag: int
    GWescape_flag: int
    continue_ng_flag: int
    hard: int
    bsi_event_flag: int
    bsi_exchange_flag: int
    bsi_ionization_flag: int
    bsi_ejection_flag: int
    trap_reached_flag: int
    stop_reason_flag: int


@dataclass
class TimelineEvent:
    sample_index: int
    generation: int
    bbh_id: int
    event_id: int
    event_type: str
    t_s: float

    state_before: BBHStateSnapshot | None
    state_after: BBHStateSnapshot | None

    single_bh_id: int | None = None
    single_bh_mass_msun: float | None = None

    bsi_type: str | None = None
    bsi_probability: float | None = None
    bsi_impact_parameter_cm: float | None = None
    bsi_velocity_cms: float | None = None
    bsi_delta_a_cm: float | None = None
    bsi_delta_ecc: float | None = None
    bsi_delta_chi1: float | None = None
    bsi_delta_chi2: float | None = None
    bsi_delta_th1: float | None = None
    bsi_delta_th2: float | None = None

    stop_reason: str | None = None
    metadata: dict[str, Any] | None = None


class TimelineCollector:
    def __init__(self, sample_index: int):
        self.sample_index = sample_index
        self.events: list[TimelineEvent] = []
        self._event_counter = 0

    def snapshot(self, bbh, i: int, t_s: float) -> BBHStateSnapshot:
        return BBHStateSnapshot(
            t_s=float(t_s),
            m1_msun=float(bbh.m1_msun[i]),
            m2_msun=float(bbh.m2_msun[i]),
            sma_cm=float(bbh.sma_cm[i]),
            ecc=float(bbh.ecc[i]),
            chi1=float(bbh.chi1[i]),
            chi2=float(bbh.chi2[i]),
            th1=float(bbh.th1[i]),
            th2=float(bbh.th2[i]),
            A_cm=float(bbh.A_cm[i]),
            mrem_msun=float(bbh.mrem_msun[i]),
            chirem=float(bbh.chirem[i]),
            vk_cgs=float(bbh.vk_cgs[i]),
            merger_flag=int(bbh.flags.merger_flag[i]),
            GWescape_flag=int(bbh.flags.GWescape_flag[i]),
            continue_ng_flag=int(bbh.flags.continue_ng_flag[i]),
            hard=int(bbh.flags.hard[i]),
            bsi_event_flag=int(bbh.flags.bsi_event_flag[i]),
            bsi_exchange_flag=int(bbh.flags.bsi_exchange_flag[i]),
            bsi_ionization_flag=int(bbh.flags.bsi_ionization_flag[i]),
            bsi_ejection_flag=int(bbh.flags.bsi_ejection_flag[i]),
            trap_reached_flag=int(bbh.flags.trap_reached_flag[i]),
            stop_reason_flag=int(bbh.flags.stop_reason_flag[i]),
        )

    def record(
        self,
        generation: int,
        bbh_id: int,
        event_type: str,
        t_s: float,
        state_before=None,
        state_after=None,
        single_bh_id=None,
        single_bh_mass_msun=None,
        bsi_type=None,
        bsi_probability=None,
        bsi_impact_parameter_cm=None,
        bsi_velocity_cms=None,
        bsi_delta_a_cm=None,
        bsi_delta_ecc=None,
        bsi_delta_chi1=None,
        bsi_delta_chi2=None,
        bsi_delta_th1=None,
        bsi_delta_th2=None,
        stop_reason=None,
        metadata=None,
    ) -> None:
        event = TimelineEvent(
            sample_index=self.sample_index,
            generation=generation,
            bbh_id=int(bbh_id),
            event_id=self._event_counter,
            event_type=event_type,
            t_s=float(t_s),
            state_before=state_before,
            state_after=state_after,
            single_bh_id=single_bh_id,
            single_bh_mass_msun=single_bh_mass_msun,
            bsi_type=bsi_type,
            bsi_probability=bsi_probability,
            bsi_impact_parameter_cm=bsi_impact_parameter_cm,
            bsi_velocity_cms=bsi_velocity_cms,
            bsi_delta_a_cm=bsi_delta_a_cm,
            bsi_delta_ecc=bsi_delta_ecc,
            bsi_delta_chi1=bsi_delta_chi1,
            bsi_delta_chi2=bsi_delta_chi2,
            bsi_delta_th1=bsi_delta_th1,
            bsi_delta_th2=bsi_delta_th2,
            stop_reason=stop_reason,
            metadata=metadata,
        )

        self.events.append(event)
        self._event_counter += 1

    def to_records(self) -> list[dict[str, Any]]:
        records = []

        for event in self.events:
            row = asdict(event)
            records.append(row)

        return records
