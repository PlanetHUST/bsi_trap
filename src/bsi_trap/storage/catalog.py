# bsi_trap/storage/catalog.py

from __future__ import annotations

import h5py
import numpy as np

from .contract import HDF5Contract


def write_sample_catalog(
    *,
    f_out: h5py.File,
    samples_group: h5py.Group,
    contract: HDF5Contract,
) -> None:
    catalog = f_out.create_group(contract.group_catalog)

    sample_names = sorted(samples_group.keys())

    catalog.attrs["num_samples"] = len(sample_names)
    catalog.attrs["sample_names"] = np.asarray(sample_names, dtype="S")

    sample_indices = []
    process_ids = []
    smbh_masses = []

    for sample_name in sample_names:
        sample_group = samples_group[sample_name]

        sample_indices.append(
            int(sample_group.attrs.get(contract.attr_sample_index, -1))
        )
        process_ids.append(
            int(sample_group.attrs.get(contract.attr_process_id, -1))
        )
        smbh_masses.append(
            float(sample_group.attrs.get(contract.attr_smbh_mass_msun, np.nan))
        )

    catalog.create_dataset("sample_index", data=np.asarray(sample_indices, dtype=int))
    catalog.create_dataset("process_id", data=np.asarray(process_ids, dtype=int))
    catalog.create_dataset("smbh_mass_msun", data=np.asarray(smbh_masses, dtype=float))