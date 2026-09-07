"""pAGN migration-torque equations with CGS-facing outputs.

The equations are adapted from ``imri/src/disk_quantity/migration_torque.py``.
pAGN constants and formula evaluation remain SI inside ``compute_torque_profile``;
all public torque arrays are converted once to dyn cm (erg).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bsi_trap.simulation.models import DiskProfile
from bsi_trap.units import (
    KG_M2_TO_G_CM2,
    KG_M3_TO_G_CM3,
    M2_KG_TO_CM2_G,
    M_TO_CM,
    N_M_TO_DYN_CM,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

TYPE_II_KII_THRESHOLD = 11.0
GAP_OPENING_CONDITIONS = ("kii", "crida")
DEFAULT_MIN_TRAP_SEPARATION_CELLS = 10.0
STABLE_TRAP_KIND = 1
ANTI_TRAP_KIND = -1


@dataclass(frozen=True)
class MigrationParameters:
    alpha: float = 0.01
    gamma_adiabat: float = 5.0 / 3.0
    hydrogen_fraction: float = 0.7
    thermal_saturation_k: float = 20.0
    type_i_normalization_k: float = 25.0


@dataclass(frozen=True)
class TorqueProfile:
    """Migration quantities on one disk radius grid.

    ``torque_attenuation_k`` is q^2/(alpha h^5), used only in the pAGN
    Type-I/thermal formula. ``type_ii_transition_kii`` is q^2/(alpha h^2),
    whose threshold selects Type-II when the configured condition is ``kii``.
    These dimensionless quantities are intentionally not interchangeable.
    """

    migrator_mass_msun: float
    radius_cm: FloatArray
    radius_rg: FloatArray
    pressure_pa: FloatArray
    dln_surface_density_dln_radius: FloatArray
    dln_temperature_dln_radius: FloatArray
    dln_pressure_dln_radius: FloatArray
    corotation_coefficient: FloatArray
    lindblad_coefficient: FloatArray
    thermal_coefficient: FloatArray
    torque_attenuation_k: FloatArray
    gap_attenuation_factor: FloatArray
    type_i_gap_denominator: FloatArray
    type_ii_transition_kii: FloatArray
    hill_radius_over_scale_height: FloatArray
    crida_gap_parameter: FloatArray
    type_ii_mask: BoolArray
    gamma_0_cgs: FloatArray
    gamma_type_i_cgs: FloatArray
    gamma_thermal_cgs: FloatArray
    gamma_type_ii_cgs: FloatArray
    gamma_selected_cgs: FloatArray


@dataclass(frozen=True)
class TorqueTrapRadii:
    """Resolved migration attractors and repellers in both radius units."""

    stable_radius_cm: FloatArray
    anti_trap_radius_cm: FloatArray
    stable_radius_rg: FloatArray
    anti_trap_radius_rg: FloatArray


def _empty_torque_traps() -> TorqueTrapRadii:
    empty = np.array([], dtype=float)
    return TorqueTrapRadii(empty, empty, empty, empty)


def _require_pagn_bits():
    try:
        import pagn.constants as ct
        from pagn.opacities import electron_scattering_opacity
    except ImportError as exc:
        raise ImportError(
            "pAGN is required for migration torques. Install with `pip install pagn`."
        ) from exc
    return ct, electron_scattering_opacity


def _constant(constants: object, *names: str) -> float:
    for name in names:
        if hasattr(constants, name):
            return float(getattr(constants, name))
    raise AttributeError(f"pAGN.constants is missing one of: {', '.join(names)}")


def torque_n_m_to_cgs(torque_n_m: npt.ArrayLike) -> FloatArray:
    """Convert torque [N m] to [dyn cm] using 1 N m = 1e7 dyn cm."""
    return np.asarray(torque_n_m, dtype=float) * N_M_TO_DYN_CM


def safe_divide(numerator: npt.ArrayLike, denominator: npt.ArrayLike) -> FloatArray:
    numerator_values = np.asarray(numerator, dtype=float)
    denominator_values = np.asarray(denominator, dtype=float)
    num, den = np.broadcast_arrays(numerator_values, denominator_values)
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0.0)


def radial_log_slope(
    radius: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    smoothing: float = 0.005,
    spline_order: int = 3,
) -> FloatArray:
    """Return d ln(values)/d ln(radius) using the donor spline convention."""
    radius_values = np.asarray(radius, dtype=float)
    field_values = np.asarray(values, dtype=float)
    if radius_values.ndim != 1 or field_values.ndim != 1:
        raise ValueError("radius and values must be one-dimensional")
    if radius_values.shape != field_values.shape or radius_values.size < 2:
        raise ValueError("radius and values need matching shapes with at least two cells")
    if np.any(radius_values <= 0.0) or np.any(field_values <= 0.0):
        raise ValueError("radius and values must be positive for log-slope")

    try:
        from scipy.interpolate import UnivariateSpline
    except ImportError as exc:
        raise ImportError("scipy is required for torque-profile slopes") from exc

    log_radius = np.log10(radius_values)
    log_values = np.log10(field_values)
    order = min(int(spline_order), log_radius.size - 1)
    spline = UnivariateSpline(
        log_radius,
        log_values,
        k=order,
        s=float(smoothing),
        ext=0,
    )
    return np.asarray(spline.derivative()(log_radius), dtype=float)


def _gap_diagnostics(
    mass_ratio: float,
    aspect_ratio: FloatArray,
    alpha: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return KII, R_H/H, and Crida P as distinct dimensionless arrays."""
    type_ii_transition_kii = mass_ratio**2 / (alpha * aspect_ratio**2)
    hill_radius_over_scale_height = np.cbrt(mass_ratio / 3.0) / aspect_ratio
    crida_gap_parameter = (
        0.75 / hill_radius_over_scale_height
        + 50.0 * alpha * aspect_ratio**2 / mass_ratio
    )
    return (
        type_ii_transition_kii,
        hill_radius_over_scale_height,
        crida_gap_parameter,
    )


def gap_opening_mask(
    mass_ratio: float,
    aspect_ratio: npt.ArrayLike,
    alpha: float,
    condition: str,
) -> BoolArray:
    """Select Type-II cells using the configured donor criterion.

    ``kii`` uses q^2/(alpha h^2) > 11. ``crida`` uses
    P = 3/(4 R_H/H) + 50 alpha h^2/q < 1 together with R_H/H > 1,
    where q=m/M and h=H/R.
    """
    if mass_ratio <= 0.0 or alpha <= 0.0:
        raise ValueError("mass ratio and alpha must be positive")
    if condition not in GAP_OPENING_CONDITIONS:
        raise ValueError(
            f"gap-opening condition must be one of {GAP_OPENING_CONDITIONS}"
        )
    local_aspect_ratio = np.asarray(aspect_ratio, dtype=float)
    valid = np.isfinite(local_aspect_ratio) & (local_aspect_ratio > 0.0)
    kii, hill_over_height, crida = _gap_diagnostics(
        mass_ratio,
        local_aspect_ratio,
        alpha,
    )
    if condition == "kii":
        return valid & np.isfinite(kii) & (kii > TYPE_II_KII_THRESHOLD)
    return (
        valid
        & np.isfinite(crida)
        & np.isfinite(hill_over_height)
        & (crida < 1.0)
        & (hill_over_height > 1.0)
    )


def _compute_si_torques(
    disk: DiskProfile,
    smbh_mass_msun: float,
    migrator_mass_msun: float,
    smoothing: float,
    params: MigrationParameters,
) -> dict[str, FloatArray]:
    """Evaluate the donor pAGN equations after explicit CGS-to-SI adaptation."""
    ct, electron_scattering_opacity = _require_pagn_bits()
    radius_m = disk.radius_cm / M_TO_CM
    scale_height_m = disk.scale_height_cm / M_TO_CM
    density_kg_m3 = disk.density_g_cm3 / KG_M3_TO_G_CM3
    sound_speed_m_s = disk.sound_speed_cm_s / M_TO_CM
    surface_density_kg_m2 = disk.surface_density_g_cm2 / KG_M2_TO_G_CM2
    opacity_m2_kg = disk.opacity_cm2_g / M2_KG_TO_CM2_G
    temperature_k = disk.temperature_k
    omega_s_inv = disk.omega_s_inv

    k_boltzmann = _constant(ct, "Kb", "kB", "k")
    atomic_mass_kg = _constant(ct, "massU", "mU", "mp")
    gas_pressure_pa = density_kg_m3 * temperature_k * k_boltzmann / atomic_mass_kg
    radiation_pressure_pa = (
        disk.optical_depth
        * float(ct.sigmaSB)
        * disk.effective_temperature_fourth_k4
        / (2.0 * float(ct.c))
    )
    pressure_pa = gas_pressure_pa + radiation_pressure_pa

    dln_surface_density = radial_log_slope(
        radius_m,
        surface_density_kg_m2,
        smoothing=smoothing,
    )
    dln_temperature = radial_log_slope(
        radius_m,
        temperature_k,
        smoothing=smoothing,
    )
    dln_pressure = radial_log_slope(radius_m, pressure_pa, smoothing=smoothing)

    gamma_adiabat = params.gamma_adiabat
    thermal_diffusivity_m2_s = (
        16.0
        * gamma_adiabat
        * (gamma_adiabat - 1.0)
        * float(ct.sigmaSB)
        * temperature_k**4
        / (3.0 * opacity_m2_kg * density_kg_m3**2 * scale_height_m**2 * omega_s_inv**2)
    )
    critical_diffusivity_m2_s = scale_height_m**2 * omega_s_inv
    x_squared = np.sqrt(
        safe_divide(thermal_diffusivity_m2_s, 2.0 * critical_diffusivity_m2_s)
    )
    effective_gamma_factor = (x_squared + 1.0 / gamma_adiabat) / (x_squared + 1.0)
    corotation_coefficient = (
        0.46 + 0.96 * dln_surface_density - 1.8 * dln_temperature
    ) / gamma_adiabat
    lindblad_coefficient = (
        -2.34 - 0.1 * dln_surface_density + 1.5 * dln_temperature
    ) * effective_gamma_factor

    smbh_mass_kg = float(smbh_mass_msun) * float(ct.MSun)
    migrator_mass_kg = float(migrator_mass_msun) * float(ct.MSun)
    mass_ratio = migrator_mass_kg / smbh_mass_kg
    aspect_ratio = disk.aspect_ratio
    torque_attenuation_k = mass_ratio**2 / (aspect_ratio**5 * params.alpha)
    gap_attenuation_factor = np.exp(
        -torque_attenuation_k / params.thermal_saturation_k
    )
    type_i_gap_denominator = 1.0 + (
        torque_attenuation_k / params.type_i_normalization_k
    )
    gamma_0_n_m = (
        mass_ratio**2
        * radius_m**4
        * omega_s_inv**2
        * surface_density_kg_m2
        / aspect_ratio**2
    )

    corotation_radius_m = (
        np.abs(dln_pressure)
        * sound_speed_m_s**2
        / (3.0 * gamma_adiabat * radius_m * omega_s_inv**2)
    )
    electron_scattering_opacity_m2_kg = electron_scattering_opacity(
        X=params.hydrogen_fraction
    )
    eddington_luminosity_w = (
        4.0
        * np.pi
        * float(ct.G)
        * float(ct.c)
        * migrator_mass_kg
        / electron_scattering_opacity_m2_kg
    )
    critical_luminosity_w = (
        4.0
        * np.pi
        * float(ct.G)
        * migrator_mass_kg
        * density_kg_m3
        * thermal_diffusivity_m2_s
        / gamma_adiabat
    )
    thermal_length_m = np.sqrt(
        2.0 * thermal_diffusivity_m2_s / (3.0 * gamma_adiabat * omega_s_inv)
    )
    bondi_radius_m = float(ct.G) * migrator_mass_kg / sound_speed_m_s**2
    thermal_saturation_parameter = thermal_diffusivity_m2_s / (
        sound_speed_m_s * np.minimum(scale_height_m, bondi_radius_m)
    )
    luminosity_ratio = safe_divide(
        eddington_luminosity_w,
        critical_luminosity_w,
    )
    unsaturated = luminosity_ratio * (
        4.0 * thermal_saturation_parameter / (1.0 + 4.0 * thermal_saturation_parameter)
    ) - (
        2.0 * thermal_saturation_parameter / (1.0 + 2.0 * thermal_saturation_parameter)
    )
    saturated = luminosity_ratio - 1.0
    thermal_prefactor = (
        1.61
        * (gamma_adiabat - 1.0)
        * safe_divide(corotation_radius_m, gamma_adiabat * thermal_length_m)
        * (radius_m / scale_height_m)
    )
    thermal_coefficient = np.where(
        thermal_saturation_parameter < 1.0,
        thermal_prefactor * unsaturated,
        thermal_prefactor * saturated,
    )
    gamma_type_i_n_m = (
        (corotation_coefficient * gap_attenuation_factor + lindblad_coefficient)
        / type_i_gap_denominator
        * gamma_0_n_m
    )
    gamma_thermal_n_m = gamma_0_n_m * thermal_coefficient
    gamma_type_ii_n_m = (
        -params.alpha
        * migrator_mass_kg
        * aspect_ratio**2
        * radius_m**2
        * omega_s_inv**2
    )
    return {
        "pressure_pa": pressure_pa,
        "dln_surface_density": dln_surface_density,
        "dln_temperature": dln_temperature,
        "dln_pressure": dln_pressure,
        "corotation_coefficient": corotation_coefficient,
        "lindblad_coefficient": lindblad_coefficient,
        "thermal_coefficient": thermal_coefficient,
        "torque_attenuation_k": torque_attenuation_k,
        "gap_attenuation_factor": gap_attenuation_factor,
        "type_i_gap_denominator": type_i_gap_denominator,
        "gamma_0_n_m": gamma_0_n_m,
        "gamma_type_i_n_m": gamma_type_i_n_m,
        "gamma_thermal_n_m": gamma_thermal_n_m,
        "gamma_type_ii_n_m": gamma_type_ii_n_m,
    }


def compute_torque_profile(
    disk: DiskProfile,
    smbh_mass_msun: float,
    migrator_mass_msun: float,
    *,
    smoothing: float = 0.005,
    params: MigrationParameters | None = None,
    gap_opening_condition: str = "kii",
    consider_type_ii_migration: bool = True,
) -> TorqueProfile:
    """Return the selected migration torque profile in CGS."""
    if smbh_mass_msun <= 0.0 or migrator_mass_msun <= 0.0:
        raise ValueError("SMBH and migrator masses [Msun] must be positive")
    params = MigrationParameters() if params is None else params
    if params.alpha <= 0.0:
        raise ValueError("migration alpha must be positive")
    si = _compute_si_torques(
        disk,
        smbh_mass_msun,
        migrator_mass_msun,
        smoothing,
        params,
    )
    mass_ratio = migrator_mass_msun / smbh_mass_msun
    kii, hill_over_height, crida = _gap_diagnostics(
        mass_ratio,
        disk.aspect_ratio,
        params.alpha,
    )
    if consider_type_ii_migration:
        type_ii_mask = gap_opening_mask(
            mass_ratio,
            disk.aspect_ratio,
            params.alpha,
            gap_opening_condition,
        )
    else:
        type_ii_mask = np.zeros(disk.radius_cm.shape, dtype=bool)

    gamma_0_cgs = torque_n_m_to_cgs(si["gamma_0_n_m"])
    gamma_type_i_cgs = torque_n_m_to_cgs(si["gamma_type_i_n_m"])
    gamma_thermal_cgs = torque_n_m_to_cgs(si["gamma_thermal_n_m"])
    gamma_type_ii_cgs = torque_n_m_to_cgs(si["gamma_type_ii_n_m"])
    type_i_thermal_cgs = gamma_type_i_cgs + (
        1.0 - si["gap_attenuation_factor"]
    ) * gamma_thermal_cgs
    gamma_selected_cgs = np.where(
        type_ii_mask,
        gamma_type_ii_cgs,
        type_i_thermal_cgs,
    )
    radius_rg = disk.radius_cm / disk.gravitational_radius_cm
    return TorqueProfile(
        migrator_mass_msun=float(migrator_mass_msun),
        radius_cm=disk.radius_cm,
        radius_rg=radius_rg,
        pressure_pa=si["pressure_pa"],
        dln_surface_density_dln_radius=si["dln_surface_density"],
        dln_temperature_dln_radius=si["dln_temperature"],
        dln_pressure_dln_radius=si["dln_pressure"],
        corotation_coefficient=si["corotation_coefficient"],
        lindblad_coefficient=si["lindblad_coefficient"],
        thermal_coefficient=si["thermal_coefficient"],
        torque_attenuation_k=si["torque_attenuation_k"],
        gap_attenuation_factor=si["gap_attenuation_factor"],
        type_i_gap_denominator=si["type_i_gap_denominator"],
        type_ii_transition_kii=kii,
        hill_radius_over_scale_height=hill_over_height,
        crida_gap_parameter=crida,
        type_ii_mask=type_ii_mask,
        gamma_0_cgs=gamma_0_cgs,
        gamma_type_i_cgs=gamma_type_i_cgs,
        gamma_thermal_cgs=gamma_thermal_cgs,
        gamma_type_ii_cgs=gamma_type_ii_cgs,
        gamma_selected_cgs=gamma_selected_cgs,
    )


def _resolved_crossing_mask(
    root_radius_cm: FloatArray,
    crossing_kind: npt.NDArray[np.int64],
    local_width_cm: FloatArray,
    min_separation_cells: float,
) -> BoolArray:
    keep = np.ones(root_radius_cm.size, dtype=bool)
    if min_separation_cells <= 0.0 or root_radius_cm.size < 2:
        return keep

    group = [0]

    def flush_group() -> None:
        if len(group) < 2:
            return
        group_kind = crossing_kind[np.asarray(group, dtype=int)]
        if np.unique(group_kind).size < 2:
            return
        keep[group] = False

    for crossing_index in range(1, root_radius_cm.size):
        previous_index = group[-1]
        separation_cm = root_radius_cm[crossing_index] - root_radius_cm[previous_index]
        local_resolution_cm = min(
            local_width_cm[crossing_index],
            local_width_cm[previous_index],
        )
        if separation_cm < min_separation_cells * local_resolution_cm:
            group.append(crossing_index)
            continue
        flush_group()
        group = [crossing_index]
    flush_group()
    return keep


def classify_torque_traps(
    radius_cm: npt.ArrayLike,
    torque_cgs: npt.ArrayLike,
    gravitational_radius_cm: float,
    *,
    min_separation_cells: float = DEFAULT_MIN_TRAP_SEPARATION_CELLS,
) -> TorqueTrapRadii:
    """Classify interpolated torque zeros as stable (+ to -) or anti-traps."""
    radius_values_cm = np.asarray(radius_cm, dtype=float)
    torque_values_cgs = np.asarray(torque_cgs, dtype=float)
    min_separation_cells = float(min_separation_cells)
    if gravitational_radius_cm <= 0.0 or not np.isfinite(gravitational_radius_cm):
        raise ValueError("gravitational radius [cm] must be finite and positive")
    if radius_values_cm.ndim != 1 or torque_values_cgs.ndim != 1:
        raise ValueError("radius and torque arrays must be one-dimensional")
    if radius_values_cm.shape != torque_values_cgs.shape:
        raise ValueError("radius and torque arrays must have the same shape")

    finite_nonzero = (
        np.isfinite(radius_values_cm)
        & np.isfinite(torque_values_cgs)
        & (radius_values_cm > 0.0)
        & (torque_values_cgs != 0.0)
    )
    radius_values_cm = radius_values_cm[finite_nonzero]
    torque_values_cgs = torque_values_cgs[finite_nonzero]
    if (
        torque_values_cgs.size < 2
        or not np.isfinite(min_separation_cells)
        or min_separation_cells < 0.0
    ):
        return _empty_torque_traps()

    order = np.argsort(radius_values_cm)
    radius_values_cm = radius_values_cm[order]
    torque_values_cgs = torque_values_cgs[order]
    candidate_roots_cm: list[float] = []
    candidate_kinds: list[int] = []
    local_width_cm: list[float] = []
    crossing_indices = np.where(
        np.sign(torque_values_cgs[1:]) != np.sign(torque_values_cgs[:-1])
    )[0]
    for index in crossing_indices:
        inner_radius_cm = radius_values_cm[index]
        outer_radius_cm = radius_values_cm[index + 1]
        inner_torque_cgs = torque_values_cgs[index]
        outer_torque_cgs = torque_values_cgs[index + 1]
        root_cm = inner_radius_cm - inner_torque_cgs * (
            outer_radius_cm - inner_radius_cm
        ) / (outer_torque_cgs - inner_torque_cgs)
        if inner_torque_cgs > 0.0 and outer_torque_cgs < 0.0:
            kind = STABLE_TRAP_KIND
        elif inner_torque_cgs < 0.0 and outer_torque_cgs > 0.0:
            kind = ANTI_TRAP_KIND
        else:
            continue
        candidate_roots_cm.append(float(root_cm))
        candidate_kinds.append(kind)
        local_width_cm.append(float(outer_radius_cm - inner_radius_cm))

    if not candidate_roots_cm:
        return _empty_torque_traps()

    roots_cm = np.asarray(candidate_roots_cm, dtype=float)
    kinds = np.asarray(candidate_kinds, dtype=np.int64)
    widths_cm = np.asarray(local_width_cm, dtype=float)
    root_order = np.argsort(roots_cm)
    roots_cm = roots_cm[root_order]
    kinds = kinds[root_order]
    widths_cm = widths_cm[root_order]
    keep = _resolved_crossing_mask(
        roots_cm,
        kinds,
        widths_cm,
        min_separation_cells,
    )
    stable_radius_cm = roots_cm[keep & (kinds == STABLE_TRAP_KIND)]
    anti_trap_radius_cm = roots_cm[keep & (kinds == ANTI_TRAP_KIND)]
    return TorqueTrapRadii(
        stable_radius_cm=stable_radius_cm,
        anti_trap_radius_cm=anti_trap_radius_cm,
        stable_radius_rg=stable_radius_cm / gravitational_radius_cm,
        anti_trap_radius_rg=anti_trap_radius_cm / gravitational_radius_cm,
    )


__all__ = [
    "GAP_OPENING_CONDITIONS",
    "DEFAULT_MIN_TRAP_SEPARATION_CELLS",
    "MigrationParameters",
    "TYPE_II_KII_THRESHOLD",
    "TorqueProfile",
    "TorqueTrapRadii",
    "classify_torque_traps",
    "compute_torque_profile",
    "gap_opening_mask",
    "radial_log_slope",
    "torque_n_m_to_cgs",
]
