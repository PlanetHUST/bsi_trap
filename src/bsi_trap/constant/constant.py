"""Compatibility aliases for the authoritative constants in ``bsi_trap.units``."""

from bsi_trap.constant.stop_reason import StopReasons
from bsi_trap.units import AU_CM, C_CM_S, G_CGS, MSUN_G, MYR_S, PC_CM, YEAR_S

tHubble = 13.6e3
parsec = PC_CM
msun = MSUN_G
AU = AU_CM
Rsun = 6.95700e10
G = G_CGS
c = C_CM_S
c5 = c**5
G3 = G**3
yr = YEAR_S
Myr = MYR_S

STOP_HUBBLE_LIMIT = StopReasons.STOP_HUBBLE_LIMIT
STOP_INTEGRATION_LIMIT = StopReasons.STOP_INTEGRATION_LIMIT

__all__ = [
    "AU",
    "G",
    "G3",
    "Myr",
    "Rsun",
    "STOP_HUBBLE_LIMIT",
    "STOP_INTEGRATION_LIMIT",
    "StopReasons",
    "c",
    "c5",
    "msun",
    "parsec",
    "tHubble",
    "yr",
]
