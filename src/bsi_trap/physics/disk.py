"""AGN disk setup for the compact demo."""

from __future__ import annotations

import contextlib
import os

import numpy as np

from bsi_trap.simulation.models import DemoConfig, DiskProfile
from bsi_trap.units import KG_M2_TO_G_CM2, METER_TO_CM


def _load_pagn():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        from pagn import Sirko, Thompson
        import pagn.constants as ct
    except ImportError as exc:
        raise RuntimeError(
            "This demo needs pAGN for the disk profile. Install pagn or run in "
            "the project environment."
        ) from exc
    return Sirko, Thompson, ct


def build_disk_profile(config: DemoConfig) -> DiskProfile:
    """Solve one pAGN disk and select a trap radius near Q=1."""
    Sirko, Thompson, ct = _load_pagn()
    model = config.disk_model.lower()
    smbh_mass_si = config.smbh_mass_msun * ct.MSun

    with contextlib.ExitStack() as stack:
        devnull = stack.enter_context(open(os.devnull, "w"))
        stack.enter_context(contextlib.redirect_stdout(devnull))
        if model == "sirko":
            disk = Sirko.SirkoAGN(Mbh=smbh_mass_si, alpha=config.alpha, b=0, le=0.5)
        elif model == "thompson":
            rg_si = ct.G * smbh_mass_si / ct.c**2
            disk = Thompson.ThompsonAGN(Mbh=smbh_mass_si, Rout=1.0e7 * rg_si)
        else:
            raise ValueError("disk_model must be 'sirko' or 'thompson'")
        disk.solve_disk(N=config.disk_resolution)

    radius_si = np.asarray(disk.R, dtype=float)
    height_si = np.asarray(disk.h, dtype=float)
    rho_si = np.asarray(disk.rho, dtype=float)
    cs_si = np.asarray(disk.cs, dtype=float)
    omega_si = np.asarray(disk.Omega, dtype=float)
    sigma_si = 2.0 * rho_si * height_si
    toomre_q = (
        np.asarray(disk.Q, dtype=float)
        if hasattr(disk, "Q")
        else cs_si * omega_si / (np.pi * ct.G * sigma_si)
    )

    order = np.argsort(radius_si)
    radius_cm = radius_si[order] * METER_TO_CM
    scale_height_cm = height_si[order] * METER_TO_CM
    sigma_g_cm2 = sigma_si[order] * KG_M2_TO_G_CM2
    cs_cm_s = cs_si[order] * METER_TO_CM
    toomre_q = toomre_q[order]
    finite = (
        np.isfinite(radius_cm)
        & np.isfinite(scale_height_cm)
        & np.isfinite(sigma_g_cm2)
        & np.isfinite(cs_cm_s)
        & (radius_cm > 0.0)
        & (scale_height_cm > 0.0)
        & (sigma_g_cm2 > 0.0)
        & (cs_cm_s > 0.0)
    )
    if not np.any(finite):
        raise RuntimeError("pAGN returned no finite disk cells")

    radius_cm = radius_cm[finite]
    scale_height_cm = scale_height_cm[finite]
    sigma_g_cm2 = sigma_g_cm2[finite]
    cs_cm_s = cs_cm_s[finite]
    toomre_q = toomre_q[finite]
    h_over_r = scale_height_cm / radius_cm

    trap_idx = radius_cm.size // 2
    q_candidates = np.isfinite(toomre_q) & (toomre_q > 0.0)
    if np.any(q_candidates):
        candidate_idx = np.flatnonzero(q_candidates)
        trap_idx = candidate_idx[
            int(np.argmin(np.abs(np.log10(toomre_q[q_candidates]))))
        ]

    return DiskProfile(
        radius_cm=radius_cm,
        sigma_g_cm2=sigma_g_cm2,
        cs_cm_s=cs_cm_s,
        h_over_r=h_over_r,
        toomre_q=toomre_q,
        r_trap_cm=float(radius_cm[trap_idx]),
        r_in_cm=float(radius_cm[0]),
        r_out_cm=float(radius_cm[-1]),
    )
