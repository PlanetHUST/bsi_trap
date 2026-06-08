# bsi_trap/storage/generation.py

from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np

from .contract import StorageContract


@dataclass(frozen=True)
class GenerationRecord:
    simulation_type: str
    sample_index: int
    generation: int
    data: Mapping[str, Any]
    generation_type: str = ""
    process_id: int = -1
    smbh_mass_msun: float = np.nan


def get_field(source: Any, name: str) -> Any:
    if source is None:
        return None

    if isinstance(source, Mapping):
        return source.get(name)

    return getattr(source, name, None)


def normalize_generation_record(
    *,
    simulation_type: str,
    sample_index: int,
    generation: int,
    generation_data: Mapping[str, Any],
    contract: StorageContract,
) -> GenerationRecord:
    agn = generation_data.get(contract.key_agn)

    smbh_mass = get_field(agn, contract.attr_smbh_mass_msun)

    return GenerationRecord(
        simulation_type=simulation_type,
        sample_index=sample_index,
        generation=generation,
        data=generation_data,
        generation_type=str(generation_data.get(contract.key_generation_type, "")),
        process_id=int(generation_data.get(contract.key_process_id, -1)),
        smbh_mass_msun=float(smbh_mass) if smbh_mass is not None else float(np.nan),
    )
