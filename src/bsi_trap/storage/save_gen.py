# bsi_trap/storage/save_gen.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from .contract import HDF5Contract
from .gen_structure import GenerationRecord, get_field, normalize_generation_record
from .hdf5_utils import write_mapping_group, write_model_config_attrs
from .schemas import generation_schema


def _write_value(
    group: h5py.Group,
    name: str,
    value: Any,
    unit: str | None = None,
) -> None:
    if value is None:
        return

    if isinstance(value, (str, bytes)) or np.isscalar(value):
        group.attrs[name] = value
        return

    arr = np.asarray(value)

    if arr.size == 0:
        return

    dataset = group.create_dataset(name, data=arr)

    if unit:
        dataset.attrs["unit"] = unit


def _write_generation_payload(
    h5f: h5py.File,
    record: GenerationRecord,
    contract: HDF5Contract,
) -> None:
    for group_spec in generation_schema(contract):
        source = record.data.get(group_spec.source_key)

        if source is None:
            continue

        group = h5f.create_group(group_spec.group_name)

        if group_spec.recursive:
            for key, value in source.items():
                if isinstance(value, Mapping):
                    write_mapping_group(group, key, value)
                else:
                    _write_value(group, key, value)
            continue

        for field in group_spec.fields:
            value = get_field(source, field.source_name)

            if value is None:
                continue

            if field.transform:
                value = field.transform(value, record.data, contract)

            if field.dtype:
                value = np.asarray(value, dtype=field.dtype)

            _write_value(
                group=group,
                name=field.output_name,
                value=value,
                unit=field.unit,
            )


def save_generation_to_file(
    *,
    simulation_type: str,
    sample_index: int,
    generation: int,
    generation_data: Mapping[str, Any],
    contract: HDF5Contract = HDF5Contract(),
) -> Path:
    record = normalize_generation_record(
        simulation_type=simulation_type,
        sample_index=sample_index,
        generation=generation,
        generation_data=generation_data,
        contract=contract,
    )

    output_path = contract.generation_file_path(
        simulation_type=record.simulation_type,
        sample_index=record.sample_index,
        generation=record.generation,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5f:
        h5f.attrs[contract.attr_simulation_type] = record.simulation_type
        h5f.attrs[contract.attr_sample_index] = record.sample_index
        h5f.attrs[contract.attr_generation] = record.generation
        h5f.attrs[contract.attr_generation_type] = record.generation_type
        h5f.attrs[contract.attr_process_id] = record.process_id
        h5f.attrs[contract.attr_smbh_mass_msun] = record.smbh_mass_msun

        write_model_config_attrs(
            h5f,
            record.data.get(contract.key_model_config),
            contract=contract,
        )

        _write_generation_payload(h5f, record, contract)

    return output_path