
############stop-reason flags##################
# These integer codes are persisted to HDF5 as ``stop_reason_flag``. Keep the
# numeric values stable once output files have been produced, because plotting
# and post-processing read old files by code.
class StopReasons:
    STOP_NONE = 0
    STOP_SOFT_BINARY = 1
    STOP_GLOBAL_TRAP_OVERLOAD = 2
    STOP_SINGLE_TRAP_OVERLOAD = 3
    STOP_TRAP_NOT_REACHED = 4
    STOP_INITIAL_BINARY_TOO_TIGHT = 5
    STOP_HUBBLE_LIMIT = 6
    STOP_INTEGRATION_LIMIT = 7
    STOP_REMNANT_TRAP_OVERLOAD = 8
    STOP_GW_ESCAPE = 9
    STOP_DISK_RETENTION = 10
    STOP_AGN_LIFETIME = 11
    STOP_NON_PHYSICAL_ORBIT = 12
    STOP_TIME_BUDGET = 13
    STOP_POST_AGN_ISOLATED_MERGER = 14
    STOP_GENERATION_WALLTIME_LIMIT = 15
    STOP_BSI_IONIZATION = 16
    STOP_BSI_EJECTION = 17