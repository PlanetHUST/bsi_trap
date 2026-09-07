"""Shared physical constants and explicit SI-to-CGS conversions."""

from __future__ import annotations

# CGS constants. These are the sole numerical owners used by simulation code.
G_CGS = 6.6743e-8  # cm^3 g^-1 s^-2
C_CM_S = 2.99792458e10
MSUN_G = 1.98847e33
PC_CM = 3.08568e18
AU_CM = 1.495978707e13
YEAR_S = 3.15576e7
MYR_S = 1.0e6 * YEAR_S

# Explicit external-library conversion factors.
M_TO_CM = 100.0
KG_TO_G = 1.0e3
KG_M2_TO_G_CM2 = 0.1
KG_M3_TO_G_CM3 = 1.0e-3
M2_KG_TO_CM2_G = 10.0
N_M_TO_DYN_CM = 1.0e7

# Compatibility aliases for the original compact-demo imports.
G = G_CGS
C = C_CM_S
MSUN = MSUN_G
PC = PC_CM
AU = AU_CM
YR = YEAR_S
MYR = MYR_S
METER_TO_CM = M_TO_CM

__all__ = [
    "AU",
    "AU_CM",
    "C",
    "C_CM_S",
    "G",
    "G_CGS",
    "KG_M2_TO_G_CM2",
    "KG_M3_TO_G_CM3",
    "KG_TO_G",
    "M2_KG_TO_CM2_G",
    "METER_TO_CM",
    "MSUN",
    "MSUN_G",
    "MYR",
    "MYR_S",
    "M_TO_CM",
    "N_M_TO_DYN_CM",
    "PC",
    "PC_CM",
    "YEAR_S",
    "YR",
]
