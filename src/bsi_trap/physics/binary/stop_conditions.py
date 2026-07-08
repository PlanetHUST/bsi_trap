import time

from bsi_trap.constant.stop_reason import StopReasons


def stop_for_walltime(flags, i, walltime_deadline=None):
    """Return True after marking a binary stopped by the generation deadline."""
    if walltime_deadline is None or time.monotonic() < walltime_deadline:
        return False

    flags.merger_flag[i] = 0
    flags.continue_ng_flag[i] = 0
    if flags.stop_reason_flag[i] == StopReasons.STOP_NONE:
        flags.stop_reason_flag[i] = StopReasons.STOP_GENERATION_WALLTIME_LIMIT
    return True


def stop_for_invalid_step(flags, i):
    """Mark a binary whose integration step could not stay physical."""
    flags.merger_flag[i] = 0
    flags.continue_ng_flag[i] = 0
    flags.stop_reason_flag[i] = StopReasons.STOP_INTEGRATION_LIMIT
