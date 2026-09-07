"""Physical constant ownership and conversion-factor checks."""

from __future__ import annotations

import pytest

from bsi_trap.constant import constant as legacy
from bsi_trap.units import (
    C_CM_S,
    G_CGS,
    KG_M2_TO_G_CM2,
    KG_M3_TO_G_CM3,
    M2_KG_TO_CM2_G,
    MSUN_G,
    N_M_TO_DYN_CM,
)


def test_si_to_cgs_conversion_factors() -> None:
    assert KG_M2_TO_G_CM2 == pytest.approx(0.1)
    assert KG_M3_TO_G_CM3 == pytest.approx(1.0e-3)
    assert M2_KG_TO_CM2_G == pytest.approx(10.0)
    assert N_M_TO_DYN_CM == pytest.approx(1.0e7)


def test_legacy_constants_alias_authoritative_cgs_values() -> None:
    assert legacy.G == G_CGS
    assert legacy.c == C_CM_S
    assert legacy.msun == MSUN_G
