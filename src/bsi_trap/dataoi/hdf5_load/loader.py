"""Public HDF5 loader API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py

from bsi_trap.storage.contract import HDF5Contract

from .array_utils import finalize_arrays
from .catalog_reader import read_catalog
from .extra_group_reader import route_extra_groups
from .generation_reader import read_generation_payload
from .hdf5_paths import require_group
from .metadata_reader import read_sample_metadata
from .result_store import empty_result
from .router import route_generation_payload


class HDF5Loader:
    """Load combined HDF5 simulation files into plotting-ready dictionaries."""

    def __init__(self, contract: HDF5Contract | None = None) -> None:
        self.contract = contract or HDF5Contract()

    def load_hdf5_data(
        self,
        simulation_type: str,
        data_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        contract = self._contract_for_data_dir(data_dir)
        file_path = contract.combined_file_path(simulation_type)

        result = empty_result(
            simulation_type=simulation_type,
            route_paths=[
                "first_gen",
                "n_gen",
                "merge_data/first_gen",
                "merge_data/n_gen",
                contract.group_trap_population,
            ],
        )

        if not file_path.exists():
            return finalize_arrays(result)

        with h5py.File(file_path, "r") as h5_file:
            result["metadata"]["sample_table"] = read_catalog(h5_file, contract)

            sample_container = require_group(h5_file, contract.group_samples)

            if sample_container is None:
                return finalize_arrays(result)

            self._read_sample_container(
                sample_container=sample_container,
                result=result,
                contract=contract,
            )

        return finalize_arrays(result)

    def _contract_for_data_dir(
        self,
        data_dir: str | Path | None,
    ) -> HDF5Contract:
        if data_dir is None:
            return self.contract

        return HDF5Contract(
            root_dir=Path(data_dir),
            final_dir_name=self.contract.final_dir_name,
        )

    def _read_sample_container(
        self,
        sample_container: h5py.Group,
        result: dict[str, Any],
        contract: HDF5Contract,
    ) -> None:
        for sample_name in sorted(sample_container.keys()):
            sample_group = sample_container[sample_name]

            if not isinstance(sample_group, h5py.Group):
                continue

            sample_metadata = read_sample_metadata(sample_group, contract)

            for generation_name in sorted(sample_group.keys()):
                generation_number = contract.parse_generation_name(generation_name)

                if generation_number is None:
                    continue

                generation_group = sample_group[generation_name]

                if not isinstance(generation_group, h5py.Group):
                    continue

                payload = read_generation_payload(generation_group, contract)

                route_generation_payload(
                    result=result,
                    payload=payload,
                    sample_metadata=sample_metadata,
                    generation_number=generation_number,
                )

                route_extra_groups(
                    result=result,
                    generation_group=generation_group,
                    sample_metadata=sample_metadata,
                    generation_number=generation_number,
                    contract=contract,
                )