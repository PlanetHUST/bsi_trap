import time

from bsi_trap.constant.stop_reason import StopReasons


def stop_for_walltime(flags, i, walltime_deadline):
    """Return True after marking a binary stopped by the generation deadline."""
    if walltime_deadline is None or time.monotonic() < walltime_deadline:
        return False

    flags.merger_flag[i] = 0
    flags.continue_ng_flag[i] = 0
    if flags.stop_reason_flag[i] == StopReasons.STOP_NONE:
        flags.stop_reason_flag[i] = StopReasons.STOP_GENERATION_WALLTIME_LIMIT
    return True


def stop_for_trap_not_reached(BBH, i, flags, t_ini_s, track_vectors):
    """Defensively stop direct solver calls that have not reached the trap."""
    flags.trap_reached_flag[i] = 0
    flags.merger_flag[i] = 0
    flags.continue_ng_flag[i] = 0
    flags.stop_reason_flag[i] = StopReasons.STOP_TRAP_NOT_REACHED
    return BBH, 0.0, t_ini_s, flags, track_vectors


def stop_for_invalid_step(flags, i):
    """Mark a binary whose integration step could not stay physical."""
    flags.merger_flag[i] = 0
    flags.continue_ng_flag[i] = 0
    flags.stop_reason_flag[i] = StopReasons.STOP_INTEGRATION_LIMIT


def _stop_for_walltime(flags, i, walltime_deadline=None):
    return stop_for_walltime(flags, i, walltime_deadline)


def _stop_for_trap_not_reached(BBH, i, flags, t_ini_s, track_vectors=None):
    if track_vectors is None:
        flags.trap_reached_flag[i] = 0
        flags.merger_flag[i] = 0
        flags.continue_ng_flag[i] = 0
        flags.stop_reason_flag[i] = StopReasons.STOP_TRAP_NOT_REACHED
        return BBH, 0.0, t_ini_s, flags
    return stop_for_trap_not_reached(BBH, i, flags, t_ini_s, track_vectors)


_stop_for_invalid_step = stop_for_invalid_step
