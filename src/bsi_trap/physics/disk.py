"""pAGN adapter with one explicit SI-to-CGS conversion boundary."""

from __future__ import annotations

import contextlib
import os

import numpy as np

from bsi_trap.simulation.models import DemoConfig, DiskProfile
from bsi_trap.units import (
    KG_M2_TO_G_CM2,
    KG_M3_TO_G_CM3,
    M2_KG_TO_CM2_G,
    M_TO_CM,
)

TOOMRE_Q_OUTER_BOUNDARY = 1.0
SIRKO_Q_SEARCH_MAX_RADIUS_RG = 1.0e12
SIRKO_Q_SEARCH_RADIUS_FACTOR = 10.0
SIRKO_Q_SEARCH_RESOLUTION = 600
SIRKO_Q_BOUNDARY_PADDING = 1.05


def _require_pagn():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        from pagn import Sirko, Thompson
        import pagn.constants as ct
    except ImportError as exc:
        raise ImportError(
            "pAGN is required for physical disk profiles. Install with "
            "`pip install pagn`."
        ) from exc
    return Sirko, Thompson, ct


def _sort_disk_profile_arrays(disk: object) -> None:
    """Sort one-dimensional pAGN profile arrays by increasing radius in-place."""
    radius_m = np.asarray(getattr(disk, "R"), dtype=float)
    order = np.argsort(radius_m)
    if np.all(order == np.arange(radius_m.size)):
        return

    for name in (
        "R",
        "rho",
        "T",
        "h",
        "cs",
        "Omega",
        "tauV",
        "kappa",
        "Teff4",
        "Q",
    ):
        if not hasattr(disk, name):
            continue
        values = np.asarray(getattr(disk, name))
        if values.shape == radius_m.shape:
            setattr(disk, name, values[order])


def _profile_q(disk: object, ct: object) -> np.ndarray:
    """Return dimensionless Toomre Q from pAGN SI arrays."""
    radius_m = np.asarray(getattr(disk, "R"), dtype=float)
    if hasattr(disk, "Q"):
        toomre_q = np.asarray(getattr(disk, "Q"), dtype=float)
        if toomre_q.shape == radius_m.shape:
            return toomre_q

    sound_speed_m_s = np.asarray(getattr(disk, "cs"), dtype=float)
    omega_s_inv = np.asarray(getattr(disk, "Omega"), dtype=float)
    density_kg_m3 = np.asarray(getattr(disk, "rho"), dtype=float)
    scale_height_m = np.asarray(getattr(disk, "h"), dtype=float)
    surface_density_kg_m2 = 2.0 * density_kg_m3 * scale_height_m
    return sound_speed_m_s * omega_s_inv / (
        np.pi * float(getattr(ct, "G")) * surface_density_kg_m2
    )


def _disk_radius_attr_cm(disk: object, fallback_cm: float, *names: str) -> float:
    for name in names:
        if not hasattr(disk, name):
            continue
        value = np.asarray(getattr(disk, name), dtype=float)
        if value.shape != ():
            continue
        radius_cm = float(value) * M_TO_CM
        if np.isfinite(radius_cm) and radius_cm > 0.0:
            return radius_cm
    return float(fallback_cm)


def find_toomre_q_outer_radius_cm(
    radius_cm: np.ndarray,
    toomre_q: np.ndarray,
) -> float | None:
    """Return the first outward Q=1 self-gravity boundary [cm], if present."""
    radius_values_cm = np.asarray(radius_cm, dtype=float)
    q_values = np.asarray(toomre_q, dtype=float)
    n_values = min(radius_values_cm.size, q_values.size)
    if n_values == 0:
        return None

    radius_values_cm = radius_values_cm[:n_values]
    q_values = q_values[:n_values]
    valid = (
        np.isfinite(radius_values_cm)
        & np.isfinite(q_values)
        & (radius_values_cm > 0.0)
        & (q_values > 0.0)
    )
    radius_values_cm = radius_values_cm[valid]
    q_values = q_values[valid]
    if radius_values_cm.size == 0:
        return None

    order = np.argsort(radius_values_cm)
    radius_values_cm = radius_values_cm[order]
    q_values = q_values[order]
    log_radius_cm = np.log10(radius_values_cm)
    log_q_ratio = np.log10(q_values / TOOMRE_Q_OUTER_BOUNDARY)
    boundary_tolerance = 1.0e-8

    seen_stable_disk = False
    outermost_exact_radius_cm = None
    for index, q_ratio_value in enumerate(log_q_ratio):
        if q_ratio_value > boundary_tolerance:
            seen_stable_disk = True
            continue
        if np.isclose(q_ratio_value, 0.0, rtol=0.0, atol=boundary_tolerance):
            outermost_exact_radius_cm = float(radius_values_cm[index])
            if seen_stable_disk:
                return outermost_exact_radius_cm
            continue
        if seen_stable_disk and index > 0 and q_ratio_value < -boundary_tolerance:
            previous_q_ratio = log_q_ratio[index - 1]
            if previous_q_ratio <= boundary_tolerance:
                return float(radius_values_cm[index])
            x0, x1 = log_radius_cm[index - 1], log_radius_cm[index]
            y0, y1 = previous_q_ratio, q_ratio_value
            if y1 == y0:
                return float(radius_values_cm[index])
            return float(10.0 ** (x0 - y0 * (x1 - x0) / (y1 - y0)))

    return outermost_exact_radius_cm


def _reset_sirko_solve_state(disk: object) -> None:
    setattr(disk, "i", 0)
    setattr(disk, "isf", -1)
    setattr(disk, "Mdisk", 0.0)


def _solve_sirko_disk_with_q_outer_radius(
    disk: object,
    n_resolution: int,
    ct: object,
) -> None:
    """Solve Sirko with its outer edge set by the first outward Q=1 crossing."""
    initial_outer_radius_m = float(getattr(disk, "Rmax"))
    gravitational_radius_m = (
        float(getattr(ct, "G"))
        * float(getattr(disk, "Mbh"))
        / float(getattr(ct, "c")) ** 2
    )
    max_search_radius_m = SIRKO_Q_SEARCH_MAX_RADIUS_RG * gravitational_radius_m
    search_radius_m = initial_outer_radius_m
    search_resolution = min(int(n_resolution), SIRKO_Q_SEARCH_RESOLUTION)
    q_outer_radius_cm = None

    while search_radius_m <= max_search_radius_m:
        setattr(disk, "Rmax", search_radius_m)
        _reset_sirko_solve_state(disk)
        getattr(disk, "solve_disk")(N=search_resolution)
        q_outer_radius_cm = find_toomre_q_outer_radius_cm(
            np.asarray(getattr(disk, "R"), dtype=float) * M_TO_CM,
            np.asarray(getattr(disk, "Q"), dtype=float),
        )
        if q_outer_radius_cm is not None:
            break
        search_radius_m *= SIRKO_Q_SEARCH_RADIUS_FACTOR

    if q_outer_radius_cm is None:
        setattr(disk, "Rmax", initial_outer_radius_m)
    else:
        setattr(
            disk,
            "Rmax",
            q_outer_radius_cm / M_TO_CM * SIRKO_Q_BOUNDARY_PADDING,
        )
    _reset_sirko_solve_state(disk)
    getattr(disk, "solve_disk")(N=int(n_resolution))


def _toomre_outer_radius_cm(
    radius_cm: np.ndarray,
    toomre_q: np.ndarray,
    fallback_cm: float,
) -> float:
    boundary_cm = find_toomre_q_outer_radius_cm(radius_cm, toomre_q)
    return float(fallback_cm if boundary_cm is None else boundary_cm)


def build_pagn_disk_profile(
    smbh_mass_msun: float,
    alpha: float,
    *,
    model: str = "sirko",
    n_resolution: int = 1000,
    eddington_ratio: float = 0.5,
    b: int = 0,
    accretion_rate_g_s: float | None = None,
    thompson_outer_radius_rg: float = 1.0e7,
    thompson_outer_accretion_rate_msun_per_yr: float | None = None,
    suppress_pagn_output: bool = True,
) -> DiskProfile:
    """Build one pAGN disk and convert its complete radial profile to CGS."""
    Sirko, Thompson, ct = _require_pagn()
    model_name = model.lower()
    smbh_mass_kg = float(smbh_mass_msun) * float(ct.MSun)

    with contextlib.ExitStack() as output_context:
        if suppress_pagn_output:
            devnull = output_context.enter_context(open(os.devnull, "w"))
            output_context.enter_context(contextlib.redirect_stdout(devnull))

        if model_name == "sirko":
            kwargs = {"Mbh": smbh_mass_kg, "alpha": float(alpha), "b": int(b)}
            if accretion_rate_g_s is None:
                kwargs["le"] = float(eddington_ratio)
            else:
                kwargs["Mdot"] = float(accretion_rate_g_s) * 1.0e-3
            disk = Sirko.SirkoAGN(**kwargs)
            _solve_sirko_disk_with_q_outer_radius(disk, int(n_resolution), ct)
        elif model_name == "thompson":
            gravitational_radius_m = ct.G * smbh_mass_kg / ct.c**2
            kwargs = {
                "Mbh": smbh_mass_kg,
                "Rout": float(thompson_outer_radius_rg) * gravitational_radius_m,
            }
            if thompson_outer_accretion_rate_msun_per_yr is not None:
                kwargs["Mdot_out"] = (
                    float(thompson_outer_accretion_rate_msun_per_yr)
                    * ct.MSun
                    / ct.yr
                )
            disk = Thompson.ThompsonAGN(**kwargs)
            disk.solve_disk(N=int(n_resolution))
        else:
            raise ValueError("disk model must be 'sirko' or 'thompson'")

    _sort_disk_profile_arrays(disk)
    radius_m = np.asarray(disk.R, dtype=float)
    scale_height_m = np.asarray(disk.h, dtype=float)
    density_kg_m3 = np.asarray(disk.rho, dtype=float)
    radius_cm = radius_m * M_TO_CM
    temperature_k = np.asarray(disk.T, dtype=float)
    surface_density_g_cm2 = (
        2.0 * density_kg_m3 * scale_height_m * KG_M2_TO_G_CM2
    )
    density_g_cm3 = density_kg_m3 * KG_M3_TO_G_CM3
    sound_speed_cm_s = np.asarray(disk.cs, dtype=float) * M_TO_CM
    aspect_ratio = scale_height_m / radius_m
    optical_depth = np.asarray(disk.tauV, dtype=float)
    opacity_cm2_g = np.asarray(disk.kappa, dtype=float) * M2_KG_TO_CM2_G
    toomre_q = _profile_q(disk, ct)
    omega_s_inv = np.asarray(disk.Omega, dtype=float)
    effective_temperature_fourth_k4 = np.asarray(disk.Teff4, dtype=float)

    inner_radius_cm = _disk_radius_attr_cm(
        disk,
        float(np.nanmin(radius_cm)),
        "Rin",
        "R_in",
    )
    fallback_outer_radius_cm = _disk_radius_attr_cm(
        disk,
        float(np.nanmax(radius_cm)),
        "Rout",
        "R_out",
    )
    outer_radius_cm = _toomre_outer_radius_cm(
        radius_cm,
        toomre_q,
        fallback_outer_radius_cm,
    )
    schwarzschild_radius_cm = float(np.asarray(disk.Rs)) * M_TO_CM

    return DiskProfile(
        model_name=model_name,
        radius_cm=radius_cm,
        temperature_k=temperature_k,
        surface_density_g_cm2=surface_density_g_cm2,
        density_g_cm3=density_g_cm3,
        sound_speed_cm_s=sound_speed_cm_s,
        aspect_ratio=aspect_ratio,
        optical_depth=optical_depth,
        opacity_cm2_g=opacity_cm2_g,
        toomre_q=toomre_q,
        omega_s_inv=omega_s_inv,
        effective_temperature_fourth_k4=effective_temperature_fourth_k4,
        inner_radius_cm=inner_radius_cm,
        outer_radius_cm=outer_radius_cm,
        gravitational_radius_cm=0.5 * schwarzschild_radius_cm,
        schwarzschild_radius_cm=schwarzschild_radius_cm,
    )


def build_disk_profile(config: DemoConfig) -> DiskProfile:
    """Build the disk selected by one simulation configuration."""
    return build_pagn_disk_profile(
        smbh_mass_msun=config.smbh_mass_msun,
        alpha=config.alpha,
        model=config.disk_model,
        n_resolution=config.disk_resolution,
        eddington_ratio=config.disk_eddington_ratio,
        b=config.disk_b,
        accretion_rate_g_s=config.disk_accretion_rate_g_s,
        thompson_outer_radius_rg=config.thompson_outer_radius_rg,
        thompson_outer_accretion_rate_msun_per_yr=(
            config.thompson_outer_accretion_rate_msun_per_yr
        ),
    )


__all__ = [
    "DiskProfile",
    "build_disk_profile",
    "build_pagn_disk_profile",
    "find_toomre_q_outer_radius_cm",
]
