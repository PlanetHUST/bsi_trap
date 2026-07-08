"""Evolution event-ledger construction."""

from __future__ import annotations

import numpy as np

from bsi_trap.simulation.state.models import BBH_properties

from .bsi import construct_bsi_timeline
from .models import (
    BSI_Timeline,
    BoolArray,
    EvolutionTimeline,
    FloatArray,
    IndexArray,
    IntArray,
)


def construct_bbh_evolution_timeline(
    bbh: BBH_properties,
    bsi_timeline: BSI_Timeline | None = None,
) -> EvolutionTimeline:
    """Build the absolute-time BBH evolution outcome ledger.

    The BBH object should already contain the post-evolution physical state.
    This function only packages times, merger flags, stop reasons, and the BSI
    event ledger for storage and analysis.
    """

    if bsi_timeline is None:
        bsi_timeline = construct_bsi_timeline(bbh)

    if bbh.num == 0:
        return EvolutionTimeline.empty(bsi_timeline=bsi_timeline)

    binary_idx: IndexArray = np.arange(bbh.num, dtype=np.intp)
    start_s: FloatArray = np.asarray(bbh.t_ini_s, dtype=np.float64)
    delay_s: FloatArray = np.asarray(bbh.t_del_s, dtype=np.float64)
    merged: BoolArray = np.asarray(bbh.flags.merger_flag != 0, dtype=bool)
    end_s: FloatArray = np.asarray(start_s + delay_s, dtype=np.float64)
    merger_s: FloatArray = np.full(bbh.num, np.nan, dtype=np.float64)
    merger_s[merged] = end_s[merged]
    stop_reason: IntArray = np.asarray(
        bbh.flags.stop_reason_flag,
        dtype=np.int64,
    )

    return EvolutionTimeline(
        binary_idx=binary_idx,
        start_s=start_s,
        delay_s=delay_s,
        end_s=end_s,
        merger_s=merger_s,
        merged=merged,
        stop_reason=stop_reason,
        bsi_timeline=bsi_timeline,
    )
