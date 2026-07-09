"""Small CGS constants module kept for simple imports."""

from bsi_trap.constant.stop_reason import StopReasons

tHubble = 13.6e3
parsec = 3.08568e18
msun = 1.98892e33
AU = 1.495978707e13
Rsun = 6.95700e10
G = 6.6743015e-8
c = 2.99792458e10
c5 = c**5
G3 = G**3
yr = 3.1557600e7
Myr = 1.0e6 * yr

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
