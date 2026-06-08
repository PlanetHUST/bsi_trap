import numpy as np
from typing import Optional, Tuple

from bsi_trap.classes import MigrationParameters, TorqueProfile
from bsi_trap.disk_profile.pagn_get import constant, require_pagn_bits
from bsi_trap.disk_profile.disk_numeric import sorted_disk_view, radial_log_slope, safe_divide


def total_pressure(disk):
    ct, _ = require_pagn_bits()
    k_boltzmann = constant(ct, "Kb", "kB", "k")
    mass_u = constant(ct, "massU", "mU", "mp")
    gas_pressure = disk.rho * disk.T * k_boltzmann / mass_u
    radiation_pressure = disk.tauV * ct.sigmaSB * disk.Teff4 / (2.0 * ct.c)
    return gas_pressure + radiation_pressure


def surface_density(disk):
    return 2.0 * disk.rho * disk.h


def comp_corot(dsigma_dlogr, dtemp_dlogr, gamma_adiabat):
    return (0.46 + 0.96 * dsigma_dlogr - 1.8 * dtemp_dlogr) / gamma_adiabat


def thermal_diffusivity(disk, gamma_adiabat):
    ct, _ = require_pagn_bits()
    return (
        16.0
        * gamma_adiabat
        * (gamma_adiabat - 1.0)
        * ct.sigmaSB
        * disk.T**4
        / (3.0 * disk.kappa * disk.rho**2 * disk.h**2 * disk.Omega**2)
    )


def comp_lind(dsigma_dlogr, dtemp_dlogr, disk, gamma_adiabat):
    chi = thermal_diffusivity(disk, gamma_adiabat)
    chi_c = disk.h**2 * disk.Omega
    x_sq = np.sqrt(safe_divide(chi, 2.0 * chi_c))
    f_gamma = (x_sq + 1.0 / gamma_adiabat) / (x_sq + 1.0)
    return (-2.34 - 0.1 * dsigma_dlogr + 1.5 * dtemp_dlogr) * f_gamma


def comp_kappa(q, h_over_r, alpha):
    return q**2 / (h_over_r**5 * alpha)


def comp_gamma_0(q, radius, scale_height, omega, rho):
    # return normalization torque Gamma_0 = (q/h)^2 * Sigma * r^4 * Omega^2
    sigma = 2.0 * rho * scale_height
    h_over_r = scale_height / radius
    return q**2 * radius**4 * omega**2 * sigma / (h_over_r**2)


def comp_gamma_th(dpressure_dlogr, disk, migrator_mass, params):
    # return the thermal torque coefficient
    ct, electron_scattering_opacity = require_pagn_bits()
    gamma = params.gamma_adiabat  # adibatic index

    chi = thermal_diffusivity(disk, gamma)

    x_c = (
        np.abs(dpressure_dlogr) * disk.cs**2 / (3.0 * gamma * disk.R * disk.Omega**2)
    )  # corotational radius of the embedded object

    k_es = electron_scattering_opacity(X=params.hydrogen_fraction)

    l_edd = (
        4.0 * np.pi * ct.G * ct.c * migrator_mass / k_es
    )  # Eddington luminosity of the migrator
    l_crit = (
        4.0 * np.pi * ct.G * migrator_mass * disk.rho * chi / gamma
    )  # critical luminosity for saturation of the thermal torque

    lambda_c = np.sqrt(
        2.0 * chi / (3.0 * gamma * disk.Omega)
    )  # typical length scale for thermal diffusion in the corotation region

    bondi_radius = ct.G * migrator_mass / disk.cs**2
    mu_th = chi / (disk.cs * np.minimum(disk.h, bondi_radius))

    unsaturated = safe_divide(l_edd, l_crit) * (4.0 * mu_th / (1.0 + 4.0 * mu_th)) - (
        2.0 * mu_th / (1.0 + 2.0 * mu_th)
    )
    saturated = safe_divide(l_edd, l_crit) - 1.0

    prefactor = (
        1.61 * (gamma - 1.0) * safe_divide(x_c, gamma * lambda_c) * (disk.R / disk.h)
    )

    return np.where(mu_th < 1.0, prefactor * unsaturated, prefactor * saturated)


###########Main migration torque and trap-finding functions###########


def compute_torque_profile(
    disk,
    *,
    migrator_mass_msun: float = 10.0,
    z: float = 0.0,
    smoothing: float = 0.005,
    params: Optional[MigrationParameters] = None,
) -> TorqueProfile:

    if params is None:
        params = MigrationParameters()

    ct, _ = require_pagn_bits()
    disk_view = sorted_disk_view(disk)
    migrator_mass = float(migrator_mass_msun) * ct.MSun

    sigma = surface_density(disk_view)
    pressure = total_pressure(disk_view)

    dsigma_dlogr = radial_log_slope(disk_view.R, sigma, smoothing=smoothing)
    dtemp_dlogr = radial_log_slope(disk_view.R, disk_view.T, smoothing=smoothing)
    dpressure_dlogr = radial_log_slope(disk_view.R, pressure, smoothing=smoothing)

    q = migrator_mass / disk_view.Mbh  # mass ratio of migrator to central black hole
    h_over_r = disk_view.h / disk_view.R  # aspect ratio of the disk at each radius

    dep_factor = np.exp(
        -0.5 * (float(z) / disk_view.h) ** 2
    )  # vertical dependence factor for the torque, assuming a Gaussian vertical density profile

    # compute coefficients for torques
    corot = comp_corot(dsigma_dlogr, dtemp_dlogr, params.gamma_adiabat)
    lind = comp_lind(dsigma_dlogr, dtemp_dlogr, disk_view, params.gamma_adiabat)
    kappa_gap = comp_kappa(q, h_over_r, params.alpha)
    gamma_0 = comp_gamma_0(q, disk_view.R, disk_view.h, disk_view.Omega, disk_view.rho)
    gamma_th_coeff = comp_gamma_th(dpressure_dlogr, disk_view, migrator_mass, params)

    gap_exp = np.exp(-kappa_gap / params.kappa_t)
    gamma_i = (
        (corot * gap_exp + lind) / (1.0 + kappa_gap / params.kappa_norm) * gamma_0
    )  # Lindblad torque
    gamma_th = gamma_0 * gamma_th_coeff  # thermal torque
    gamma_tot = gamma_i + gap_exp * gamma_th

    orbital_angular_momentum = migrator_mass * disk_view.Omega * disk_view.R**2

    tau_mig = np.full_like(gamma_tot, np.inf, dtype=float)
    finite_mask = (gamma_tot != 0.0) & np.isfinite(gamma_tot) & (dep_factor > 0.0)
    tau_mig[finite_mask] = orbital_angular_momentum[finite_mask] / (
        2.0 * gamma_tot[finite_mask] * dep_factor[finite_mask]
    )  # differential migration timescale, accounting for vertical dependence

    tau_e = np.abs(tau_mig) * h_over_r**2 / 0.780
    tau_i = np.abs(tau_mig) * h_over_r**2 / 0.544

    radius_cm = disk_view.R * 100.0  # convert from meters to centimeters
    radius_rs = disk_view.R / disk_view.Rs
    radius_rg = radius_rs * 2.0

    return TorqueProfile(
        radius=disk_view.R,
        radius_cm=radius_cm,
        radius_rg=radius_rg,
        radius_rs=radius_rs,
        surface_density=sigma,
        pressure=pressure,
        dsigma_dlogr=dsigma_dlogr,
        dtemp_dlogr=dtemp_dlogr,
        dpressure_dlogr=dpressure_dlogr,
        corotation_coeff=corot,
        lindblad_coeff=lind,
        thermal_coeff=gamma_th_coeff,
        gap_parameter=kappa_gap,
        gamma_0=gamma_0,
        gamma_i=gamma_i,
        gamma_th=gamma_th,
        gamma_tot=gamma_tot,
        tau_mig=tau_mig,
        tau_e=tau_e,
        tau_i=tau_i,
    )


def find_torque_zero_crossings(radius: np.ndarray, gamma_tot: np.ndarray) -> np.ndarray:
    radius = np.asarray(radius, dtype=float)
    values = np.asarray(gamma_tot, dtype=float)

    finite = np.isfinite(radius) & np.isfinite(values)
    radius = radius[finite]
    values = values[finite]

    nonzero = values != 0.0
    radius = radius[nonzero]
    values = values[nonzero]

    if values.size < 2:
        return np.array([], dtype=float)

    order = np.argsort(radius)
    radius = radius[order]
    values = values[order]

    sign_changes = np.where(np.sign(values[1:]) != np.sign(values[:-1]))[0]
    roots = []
    for index in sign_changes:
        r0, r1 = radius[index], radius[index + 1]
        y0, y1 = values[index], values[index + 1]
        roots.append(r0 - y0 * (r1 - r0) / (y1 - y0))
    return np.asarray(roots, dtype=float)


def find_stable_traps(
    torque_profile: TorqueProfile,
    *,
    require_negative_to_positive: bool = False,
) -> np.ndarray:
    r = np.asarray(torque_profile.radius_cm, dtype=float)
    g = np.asarray(torque_profile.gamma_tot, dtype=float)

    finite = np.isfinite(r) & np.isfinite(g) & (g != 0.0)
    r = r[finite]
    g = g[finite]
    if g.size < 2:
        return np.array([], dtype=float)

    order = np.argsort(r)
    r = r[order]
    g = g[order]

    stable_roots = []
    for i in np.where(np.sign(g[1:]) != np.sign(g[:-1]))[0]:
        g_left = g[i]
        g_right = g[i + 1]
        if require_negative_to_positive:
            stable = g_left < 0.0 and g_right > 0.0
        else:
            stable = g_left > 0.0 and g_right < 0.0
        if stable:
            r0, r1 = r[i], r[i + 1]
            root = r0 - g_left * (r1 - r0) / (g_right - g_left)
            stable_roots.append(root)

    return np.asarray(stable_roots, dtype=float)


def choose_outermost_stable_trap(
    trap_radii_cm: np.ndarray,
    *,
    fallback_radius_cm: float,
) -> float:
    trap_radii_cm = np.asarray(trap_radii_cm, dtype=float)
    trap_radii_cm = trap_radii_cm[np.isfinite(trap_radii_cm) & (trap_radii_cm > 0.0)]
    if trap_radii_cm.size == 0:
        return float(fallback_radius_cm)
    return float(np.max(trap_radii_cm))


def find_outermost_stable_trap(
    disk,
    *,
    migrator_mass_msun: float = 10.0,
    smoothing: float = 0.005,
    params: Optional[MigrationParameters] = None,
    fallback_radius_cm: Optional[float] = None,
) -> Tuple[float, np.ndarray, TorqueProfile]:
    torque_profile = compute_torque_profile(
        disk,
        migrator_mass_msun=migrator_mass_msun,
        smoothing=smoothing,
        params=params,
    )
    trap_radii = find_stable_traps(torque_profile)

    finite_radius = torque_profile.radius_cm[
        np.isfinite(torque_profile.radius_cm) & (torque_profile.radius_cm > 0.0)
    ]
    if finite_radius.size == 0:
        raise ValueError("torque profile has no positive finite radii")

    if fallback_radius_cm is None:
        fallback = float(np.median(finite_radius))
    else:
        fallback = float(fallback_radius_cm)
        if fallback < np.nanmin(finite_radius) or fallback > np.nanmax(finite_radius):
            fallback = float(np.median(finite_radius))

    rtrap = choose_outermost_stable_trap(
        trap_radii,
        fallback_radius_cm=fallback,
    )
    return rtrap, trap_radii, torque_profile
