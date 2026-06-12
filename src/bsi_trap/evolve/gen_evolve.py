from bsi_trap.classes.classes import BBH_properties, AGN_properties, merger_flags
from bsi_trap.config.model_config import ModelConfig
from bsi_trap.evolve.gas_harden import evolve_gas
from bsi_trap.evolve.gw import evolve_gw_only

def make_generation():
    pass


def evolve_binary(BBH: BBH_properties, model_config: ModelConfig, i:int, AGN: AGN_properties, flags: merger_flags, t_ini_s: list) -> tuple:

    """Evolve a binary through its lifecycle, applying gas hardening and GW emission until merger or stop condition."""

    if model_config.gas_mode == 'with_gas':
        BBH, t_del_s, _, flags = evolve_gas(
            AGN.disk_profile, BBH, AGN, i, flags, t_ini_s
        )
    elif model_config.gas_mode == 'without_gas':
        BBH, t_del_s, _, flags = evolve_gw_only(BBH, AGN, i, flags, t_ini_s)
    else:
        raise ValueError(f"Invalid gas mode specified: {model_config.gas_mode}")


    return BBH, t_del_s, flags
