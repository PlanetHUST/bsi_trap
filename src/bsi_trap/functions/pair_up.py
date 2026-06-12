import numpy as np

from bsi_trap.constant.constant import G
from bsi_trap.classes.classes import Embedded_BH, AGN_properties, DiskProfile


# important to investigate in similar works
def pair_up_timescale(
    agn: AGN_properties, disk: DiskProfile, m1: float, m2: float
) -> float:
    r_trap_cm = disk.r_trap_cm
    cs_cgs = disk.get_cs(r_trap_cm)
    h = disk.get_h(r_trap_cm)
    sigma_cgs = disk.get_sigma(r_trap_cm)
    t_pair_s = (
        3 * cs_cgs**3 * h * r_trap_cm / (4 * np.pi * G**2 * sigma_cgs * (m1 + m2))
    )
    return t_pair_s


def construct_trap_population(
    agn: AGN_properties,
    disk: DiskProfile,
    ebh: Embedded_BH,
    mask: np.ndarray,
    seed: int | None = None,
) -> None:
    """
    Fill ebh.trap_flag for BHs that reached the trap.

    trap_flag meaning:
        -1     : trap not reached
        -2     : reached trap but remains single
        0-num  : paired with BH index
    """

    rng = np.random.default_rng(seed)

    mask = np.asarray(mask, dtype=bool)

    # Global indices of BHs that reached the trap
    reached_idx = np.where(mask)[0]

    # Reset reached BHs as temporary singles
    ebh.trap_flag[reached_idx] = -2

    # Sort reached BHs by trap reaching time
    reached_idx = reached_idx[np.argsort(ebh.t_ini_s[reached_idx])]

    # BHs currently in trap and not paired yet
    waiting: list[int] = []

    for i in reached_idx:
        m_i = ebh.m_msun[i]
        t_i = ebh.t_ini_s[i]

        candidates = []

        for j in waiting:
            m_j = ebh.m_msun[j]
            t_j = ebh.t_ini_s[j]

            dt = t_i - t_j
            tp = pair_up_timescale(agn, disk, m_i, m_j)

            if dt <= tp:
                candidates.append((j, tp))

        if candidates:
            weights = np.array([1.0 / tp for _, tp in candidates])
            weights /= weights.sum()  # weight normalization

            chosen_idx = rng.choice(len(candidates), p=weights)
            j, _ = candidates[chosen_idx]

            ebh.trap_flag[i] = j
            ebh.trap_flag[j] = i

            # Remove paired BH from waiting pool
            waiting.remove(j)

        else:
            waiting.append(i)

    # Anything left in waiting already has trap_flag = -2
