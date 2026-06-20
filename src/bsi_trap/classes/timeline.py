# bsi_trap/evolve/timeline.py

from __future__ import annotations

import numpy as np


class BBHStateCollector:
    # Track BBH parameters during evolution, for the phase plot
    # one array for full time evolution for one BBH
    def __init__(self, num: int = 0):
        self.num = num
        self.index = np.zeros(self.num, float)
        self.t_s = np.zeros(self.num, float)
        self.m1_msun = np.zeros(self.num, float)
        self.m2_msun = np.zeros(self.num, float)
        self.sma_cm = np.zeros(self.num, float)
        self.ecc = np.zeros(self.num, float)
        self.chi1 = np.zeros(self.num, float)
        self.chi2 = np.zeros(self.num, float)
        self.th1 = np.zeros(self.num, float)
        self.th2 = np.zeros(self.num, float)

    def insert_state(self, t_s: float) -> int:
        """Insert an empty state at time ``t_s`` and return its array index."""
        position = int(np.searchsorted(self.t_s, t_s))
        for name in (
            "index",
            "t_s",
            "m1_msun",
            "m2_msun",
            "sma_cm",
            "ecc",
            "chi1",
            "chi2",
            "th1",
            "th2",
        ):
            setattr(self, name, np.insert(getattr(self, name), position, 0.0))

        self.t_s[position] = t_s
        self.num += 1
        return position


class TrapStateCollector:
    def __init__(self, num: int = 0):
        self.num = num
        self.t_s = np.zeros(self.num, float)
        self.volumn_cm3 = np.zeros(self.num, float)
        self.density_cgs = np.zeros(self.num, float)
        self.bbh_num = np.zeros(self.num, float)
        self.sbh_num = np.zeros(self.num, float)

    def insert_state(self, t_s: float) -> int:
        """Insert an empty state at time ``t_s`` and return its array index."""
        position = int(np.searchsorted(self.t_s, t_s))
        for name in ("t_s", "volumn_cm3", "density_cgs", "bbh_num", "sbh_num"):
            setattr(self, name, np.insert(getattr(self, name), position, 0.0))

        self.t_s[position] = t_s
        self.num += 1
        return position


class Timeline:
    # for discrete event timeline
    # each with different length, logging the event happen time

    def __init__(self):
        # construct after migration
        self.sbh_reaching_s = np.array([], dtype=float)
        # construct after sbh reaching, using possibility, containing randomness
        self.bbh_pairing_up_s = np.array([], dtype=float)

        # construct the BSI event timeline after build all pair-up BBH and sBH timeline
        self.weak_perturbation_s = np.array([], dtype=float)
        self.strong_perturbation_s = np.array([], dtype=float)

        self.bbh_merge_s = np.array([], dtype=float)
        self.remnant_reaching_s = np.array([], dtype=float)
