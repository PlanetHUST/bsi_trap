# bsi_trap/storage/schemas.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .contract import HDF5Contract


Transform = Callable[[Any, Mapping[str, Any], HDF5Contract], Any]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    source: str | None = None
    unit: str | None = None
    dtype: type | None = None
    transform: Transform | None = None

    @property
    def output_name(self) -> str:
        return self.name

    @property
    def source_name(self) -> str:
        return self.source or self.name


@dataclass(frozen=True)
class GroupSpec:
    group_name: str
    source_key: str
    fields: tuple[FieldSpec, ...] = ()
    recursive: bool = False
    generation_level: bool = True


def fields(
    *names: str,
    dtype: type | None = None,
    unit: str | None = None,
) -> tuple[FieldSpec, ...]:
    return tuple(
        FieldSpec(
            name=name,
            dtype=dtype,
            unit=unit,
        )
        for name in names
    )


def _get_field(source: Any, name: str) -> Any:
    if source is None:
        return None

    if isinstance(source, Mapping):
        return source.get(name)

    return getattr(source, name, None)


def radius_to_rg(
    value: Any,
    context: Mapping[str, Any],
    contract: HDF5Contract,
) -> Any:
    agn = context.get(contract.key_agn)
    rg = _get_field(agn, "rg")

    if rg is None:
        return value

    rg = float(rg)

    if np.isfinite(rg) and rg > 0:
        return np.asarray(value) / rg

    return value


def generation_schema(contract: HDF5Contract) -> tuple[GroupSpec, ...]:
    return (
        GroupSpec(
            group_name=contract.group_agn,
            source_key=contract.key_agn,
            recursive=True,
            generation_level=False,
        ),
        GroupSpec(
            group_name=contract.group_bbh,
            source_key=contract.key_bbh,
            fields=(
                *fields("m1", "m2", "chi1", "chi2"),
                FieldSpec("A", unit="r_g", transform=radius_to_rg),
                FieldSpec("Aini", unit="r_g", transform=radius_to_rg),
                FieldSpec("sma_initial", unit="cm"),
                FieldSpec("sma_final", unit="cm"),
                *fields(
                    "sma",
                    "ecc_initial",
                    "ecc_final",
                    "ecc",
                    "ecc10",
                    "ecc0p1",
                    "th1",
                    "th2",
                    "idi",
                    "mmerg",
                    "mrem",
                    "chimerg",
                    "chirem",
                    "vk_cgs",
                    "generation",
                    "trapreach_time_s",
                    "bsi_timescale_s",
                    "bsi_weak_count",
                    "bsi_delta_e_total",
                    "bsi_delta_theta_total",
                    "bsi_perturbed_flag",
                ),
            ),
        ),
        GroupSpec(
            group_name=contract.group_flags,
            source_key=contract.key_flags,
            fields=fields(
                "merger_flag",
                "GWescape_flag",
                "continue_ng_flag",
                "hard",
                "bsi_event_flag",
                "trap_reached_flag",
                "stop_reason_flag",
                dtype=int,
            ),
        ),
        GroupSpec(
            group_name=contract.group_timescales,
            source_key=contract.key_timescales,
            fields=fields(
                "tform_myr",
                "trel_myr",
                "tdamp_myr",
                "tmig_myr",
                "tini_myr",
                "tdel_myr",
                "tmerg_myr",
                "t_bsi_myr",
                dtype=float,
                unit="Myr",
            ),
        ),
        GroupSpec(
            group_name=contract.group_bbh_tracks,
            source_key=contract.key_bbh_tracks,
            fields=fields(
                "binary_idx",
                "step_idx",
                "time_s",
                "sma_cm",
                "ecc",
            ),
        ),
        GroupSpec(
            group_name=contract.group_trap_population,
            source_key=contract.key_trap_population,
            recursive=True,
        ),
        GroupSpec(
            group_name=contract.group_hierarchical_mergers,
            source_key=contract.key_hierarchical_mergers,
            recursive=True,
        ),
    )


def generation_level_schema(contract: HDF5Contract) -> tuple[GroupSpec, ...]:
    return tuple(
        spec for spec in generation_schema(contract)
        if spec.generation_level
    )


def sample_level_schema(contract: HDF5Contract) -> tuple[GroupSpec, ...]:
    return tuple(
        spec for spec in generation_schema(contract)
        if not spec.generation_level
    )
