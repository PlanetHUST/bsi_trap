# bsi_trap/storage/contract.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HDF5Contract:
    root_dir: Path = Path("output")
    final_dir_name: str = "final_combined"

    # generation_data keys
    key_agn: str = "AGN"
    key_bbh: str = "BBH"
    key_flags: str = "flags"
    key_timescales: str = "timescales"
    key_bbh_tracks: str = "bbh_tracks"
    key_trap_population: str = "trap_population"
    key_hierarchical_mergers: str = "hierarchical_mergers"
    key_model_config: str = "model_config"
    key_generation_type: str = "generation_type"
    key_process_id: str = "process_id"

    # HDF5 groups
    group_samples: str = "samples"
    group_catalog: str = "catalog"
    group_agn: str = "agn_properties"
    group_bbh: str = "bbh_properties"
    group_flags: str = "flags"
    group_timescales: str = "timescales"
    group_bbh_tracks: str = "bbh_tracks"
    group_trap_population: str = "trap_population"
    group_hierarchical_mergers: str = "hierarchical_mergers"

    # HDF5 attrs
    attr_sample_index: str = "sample_index"
    attr_generation: str = "generation"
    attr_simulation_type: str = "simulation_type"
    attr_generation_type: str = "generation_type"
    attr_process_id: str = "process_id"
    attr_smbh_mass_msun: str = "smbh_mass_msun"
    attr_model_config_json: str = "model_config_json"

    generation_prefix: str = "gen_"

    def simulation_dir(self, simulation_type: str) -> Path:
        return self.root_dir / simulation_type

    @property
    def final_output_dir(self) -> Path:
        return self.root_dir / self.final_dir_name

    def generation_file_name(self, sample_index: int, generation: int) -> str:
        return f"sample_{sample_index:04d}_gen_{generation:03d}.h5"

    def generation_file_path(
        self,
        simulation_type: str,
        sample_index: int,
        generation: int,
    ) -> Path:
        return self.simulation_dir(simulation_type) / self.generation_file_name(
            sample_index, generation
        )

    def combined_file_path(self, simulation_type: str) -> Path:
        return self.final_output_dir / f"{simulation_type}_combined.h5"

    def sample_group_name(self, sample_index: int) -> str:
        return f"sample_{sample_index:04d}"

    def generation_group_name(self, generation: int) -> str:
        return f"{self.generation_prefix}{generation:03d}"

    def parse_generation_name(self, name: str) -> int | None:
        if not name.startswith(self.generation_prefix):
            return None

        suffix = name.removeprefix(self.generation_prefix)

        if not suffix.isdigit():
            return None

        return int(suffix)


StorageContract = HDF5Contract
