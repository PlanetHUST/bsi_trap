import numpy as np

from bsi_trap.constant.constant import G, Myr, msun
from bsi_trap.classes.classes import Embedded_BH, AGN_properties, DiskProfile


# important to investigate in similar works
def pair_up_timescale(disk: DiskProfile, m1_msun: float, m2_msun: float) -> float:
    r_trap_cm = disk.r_trap_cm
    m1_g = m1_msun * msun
    m2_g = m2_msun * msun
    cs_cgs = disk.get_cs(r_trap_cm)
    h = disk.get_h(r_trap_cm)
    sigma_cgs = disk.get_sigma(r_trap_cm)
    t_pair_s = (
        3 * cs_cgs**3 * h * r_trap_cm / (4 * np.pi * G**2 * sigma_cgs * (m1_g + m2_g))
    )
    return t_pair_s


def construct_trap_population(
    agn: AGN_properties,
    disk: DiskProfile,
    ebh: Embedded_BH,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """
    Fill ebh.trap_flag for BHs that reached the trap.

    trap_flag meaning:
        -1     : trap not reached
        -2     : reached trap but remains single
        0-num  : paired with BH index
    """

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (ebh.num,):
        raise ValueError(
            f"mask must have shape ({ebh.num},), got {mask.shape}"
        )

    # Global indices of BHs that reached the trap
    reached_idx = np.where(mask)[0]

    ebh.trap_flag[~mask] = -1

    # Reset reached BHs as temporary singles
    ebh.trap_flag[reached_idx] = -2

    # Sort reached BHs by trap reaching time
    reached_idx = reached_idx[np.argsort(ebh.t_ini_s[reached_idx])]

    # BHs currently in trap and not paired yet
    waiting: list[int] = []
    agn_end_s = agn.lifetime_myr * Myr

    for i in reached_idx:
        m_msun_i = float(ebh.m_msun[i])
        t_s_i = float(ebh.t_ini_s[i])

        candidates = []

        for j in waiting:
            m_msun_j = float(ebh.m_msun[j])
            tp_s = pair_up_timescale(disk=disk, m1_msun=m_msun_i, m2_msun=m_msun_j)
            t_pair_complete_s = t_s_i + tp_s

            if t_pair_complete_s < agn_end_s:
                candidates.append((j, tp_s, t_pair_complete_s))

        if candidates:
            weights = np.array([1.0 / tp for _, tp, _ in candidates])
            weights /= weights.sum()  # weight normalization

            chosen_idx = rng.choice(len(candidates), p=weights)
            j, _, t_pair_complete_s = candidates[chosen_idx]

            ebh.trap_flag[i] = j
            ebh.trap_flag[j] = i
            ebh.t_ini_s[i] = t_pair_complete_s
            ebh.t_ini_s[j] = t_pair_complete_s

            # Remove paired BH from waiting pool
            waiting.remove(j)

        else:
            waiting.append(i)

    # Anything left in waiting already has trap_flag = -2
