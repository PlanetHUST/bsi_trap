# bsi_trap/storage/combined.py

from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np

from .catalog import write_sample_catalog
from .contract import HDF5Contract


def _require_attrs(
    h5f: h5py.File,
    path: Path,
    contract: HDF5Contract,
) -> None:
    required = (
        contract.attr_sample_index,
        contract.attr_generation,
        contract.attr_simulation_type,
    )

    missing = [name for name in required if name not in h5f.attrs]

    if missing:
        raise ValueError(f"{path} is missing required HDF5 attrs: {missing}")


def _copy_model_attrs(
    source: h5py.File,
    target: h5py.Group,
    contract: HDF5Contract,
) -> None:
    for attr_key, attr_value in source.attrs.items():
        if attr_key.startswith("model_") or attr_key == contract.attr_model_config_json:
            target.attrs[attr_key] = attr_value


def _copy_agn_to_sample(
    source: h5py.File,
    sample_group: h5py.Group,
    contract: HDF5Contract,
) -> None:
    if contract.group_agn not in source:
        return

    agn_group = source[contract.group_agn]

    for key, value in agn_group.attrs.items():
        sample_group.attrs[key] = value

    for key in agn_group.keys():
        if key not in sample_group:
            sample_group.create_dataset(key, data=agn_group[key][()])


def _create_sample_group(
    *,
    samples_group: h5py.Group,
    source_file: h5py.File,
    sample_index: int,
    simulation_type: str,
    contract: HDF5Contract,
) -> h5py.Group:
    sample_name = contract.sample_group_name(sample_index)
    sample_group = samples_group.create_group(sample_name)

    sample_group.attrs[contract.attr_sample_index] = sample_index
    sample_group.attrs[contract.attr_simulation_type] = simulation_type
    sample_group.attrs[contract.attr_process_id] = int(
        source_file.attrs.get(contract.attr_process_id, -1)
    )
    sample_group.attrs[contract.attr_smbh_mass_msun] = float(
        source_file.attrs.get(contract.attr_smbh_mass_msun, np.nan)
    )

    _copy_model_attrs(source_file, sample_group, contract)
    _copy_agn_to_sample(source_file, sample_group, contract)

    return sample_group


def _copy_generation_groups(
    source_file: h5py.File,
    generation_group: h5py.Group,
    contract: HDF5Contract,
) -> None:
    for key in source_file.keys():
        if key == contract.group_agn:
            continue

        source_file.copy(key, generation_group, name=key)


def _read_generation_file_index(
    h5_file: Path,
    contract: HDF5Contract,
) -> tuple[int, int, Path]:
    with h5py.File(h5_file, "r") as h5f:
        _require_attrs(h5f, h5_file, contract)

        sample_index = int(h5f.attrs[contract.attr_sample_index])
        generation = int(h5f.attrs[contract.attr_generation])

    return sample_index, generation, h5_file


def combine_all_files(
    contract: HDF5Contract = HDF5Contract(),
    *,
    overwrite_duplicate_generations: bool = False,
) -> list[Path]:
    contract.final_output_dir.mkdir(parents=True, exist_ok=True)

    simulation_dirs = [
        path
        for path in contract.root_dir.iterdir()
        if path.is_dir() and path.name != contract.final_dir_name
    ]

    written_files: list[Path] = []

    for sim_dir in sorted(simulation_dirs):
        simulation_type = sim_dir.name
        h5_files = sorted(sim_dir.glob("*.h5"))

        if not h5_files:
            continue

        indexed_files = sorted(
            (_read_generation_file_index(path, contract) for path in h5_files),
            key=lambda item: (item[0], item[1]),
        )

        final_path = contract.combined_file_path(simulation_type)

        with h5py.File(final_path, "w") as f_out:
            f_out.attrs[contract.attr_simulation_type] = simulation_type
            f_out.attrs["combination_time"] = float(time.time())

            samples_group = f_out.create_group(contract.group_samples)

            for sample_index, generation, h5_file in indexed_files:
                with h5py.File(h5_file, "r") as f_in:
                    file_simulation_type = str(
                        f_in.attrs[contract.attr_simulation_type]
                    )

                    if file_simulation_type != simulation_type:
                        raise ValueError(
                            f"{h5_file} belongs to simulation "
                            f"{file_simulation_type!r}, expected {simulation_type!r}"
                        )

                    sample_name = contract.sample_group_name(sample_index)

                    if sample_name not in samples_group:
                        sample_group = _create_sample_group(
                            samples_group=samples_group,
                            source_file=f_in,
                            sample_index=sample_index,
                            simulation_type=simulation_type,
                            contract=contract,
                        )
                    else:
                        sample_group = samples_group[sample_name]

                    generation_name = contract.generation_group_name(generation)

                    if generation_name in sample_group:
                        if not overwrite_duplicate_generations:
                            raise ValueError(
                                f"Duplicate generation {generation_name} "
                                f"for {sample_name} from {h5_file}"
                            )

                        del sample_group[generation_name]

                    generation_group = sample_group.create_group(generation_name)
                    generation_group.attrs[contract.attr_generation] = generation
                    generation_group.attrs[contract.attr_sample_index] = sample_index
                    generation_group.attrs[contract.attr_generation_type] = str(
                        f_in.attrs.get(contract.attr_generation_type, "")
                    )

                    _copy_generation_groups(f_in, generation_group, contract)

            write_sample_catalog(
                f_out=f_out,
                samples_group=samples_group,
                contract=contract,
            )

        written_files.append(final_path)

    return written_files