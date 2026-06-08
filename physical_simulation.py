import os
import time
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

import agn_function as agn_f
import auxiliary as auxi
import constants as const
import general_class as agn
import initialize as initi
import make_damp
import make_generation as makegen
from migration_torque import MigrationParameters, find_outermost_stable_trap
from model_config import ModelConfig, legacy_model_config
from pagn_disk_adapter import build_pagn_disk_profile

try:
    import diagnostics as diag
except ImportError:
    diag = None

try:
    import storedata as std

    STORAGE_AVAILABLE = True
except ImportError:
    print("WARNING: storedata not found; disabling HDF5 output.")
    STORAGE_AVAILABLE = False


TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass
class SampleSaveContext:
    """Sample-level storage fields reused by generation and metadata saves."""

    sample_index: int
    smbh_mass: float
    simulation_type: str
    output_dir: str
    temp_file: str
    model_config: ModelConfig
    pid: int
    agn_properties: Dict[str, Any] | None = None

    def model_config_dict(self) -> Dict[str, Any] | None:
        return self.model_config.to_dict() if self.model_config is not None else None


def _progress_enabled() -> bool:
    """Return True when live terminal progress summaries should be printed."""
    return os.getenv("AGN_PROGRESS", "1").lower() in TRUE_VALUES


def _progress(sample_index: int, simulation_type: str, message: str) -> None:
    """Print one live, flush-enabled progress line for long parallel runs."""
    if not _progress_enabled():
        return
    print(
        f"[PID:{os.getpid()}][sample {sample_index:04d}][{simulation_type}] \n{message}",
        flush=True,
    )


def _finite_mean(values, default=np.nan) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(np.mean(finite))


def _finite_minmax(values) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan, np.nan
    return float(np.min(finite)), float(np.max(finite))


def _finite_median(values, default=np.nan) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(np.median(finite))


def _count_stop_reason(flags, reason: int) -> int:
    return int(np.sum(np.asarray(flags.stop_reason_flag, dtype=int) == int(reason)))


def _format_speed(speed_cm_s: float) -> str:
    if not np.isfinite(speed_cm_s):
        return "nan km/s"
    return f"{speed_cm_s / 1.0e5:.2f} km/s"


def _trap_hardness_velocity(AGN, h_list, cs_list, r_list) -> float:
    """Return the local velocity scale used for hard/soft classification."""
    disk_candidates = []
    if getattr(AGN, "rtrap", None) is not None:
        try:
            v_kep = makegen._kepler_speed_at_trap(AGN)
            h_trap = agn_f.h_gas(AGN.rtrap, h_list, r_list)
            cs_trap = agn_f.cs_gas(AGN.rtrap, cs_list, r_list)
            disk_candidates.extend([float(cs_trap), float(h_trap * v_kep)])
        except Exception:
            pass

    finite_disk = [abs(v) for v in disk_candidates if np.isfinite(v) and v > 0.0]
    if finite_disk:
        return float(max(finite_disk))

    return float(getattr(AGN, "sigma", np.nan))


def _hard_boundary_cm(m1, m2, maverage, velocity):
    """Semi-major axis below which a binary is hard for ``velocity``."""
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    if not np.isfinite(velocity) or velocity <= 0.0 or maverage <= 0.0:
        return np.full_like(m1, np.nan, dtype=float)
    return const.G * m1 * m2 * const.msun / (float(maverage) * velocity**2)


def _print_damping_progress(
    sample_index: int,
    simulation_type: str,
    result: Dict[str, Any],
    N1: int,
    N2: int,
) -> None:
    summary = result.get("summary", {}) if isinstance(result, dict) else {}
    original_stars = int(summary.get("numStar_original", 0))
    original_bh = int(summary.get("numBH_original", 0))
    copy_stars = int(summary.get("numStar_weighted_only", 0))
    copy_bh = int(summary.get("numBH_weighted_only", 0))
    copy_outside = int(summary.get("numOutside_weighted_only", 0))
    combined_stars = int(summary.get("numStar_combined", 0))
    combined_bh = int(summary.get("numBH_combined", 0))
    mean_mult_stars = float(summary.get("mean_multiplicity_stars", 0.0))
    mean_mult_bh = float(summary.get("mean_multiplicity_bh", 0.0))

    _progress(
        sample_index,
        simulation_type,
        "damping: "
        f"embedded BH={N1}, embedded stars={N2}; "
        f"original BH/star={original_bh}/{original_stars}; "
        f"copies BH/star/outside={copy_bh}/{copy_stars}/{copy_outside}; "
        f"combined BH/star={combined_bh}/{combined_stars}; "
        f"mean multiplicity BH/star={mean_mult_bh:.2f}/{mean_mult_stars:.2f}",
    )


def _print_generation_progress(
    sample_index: int,
    simulation_type: str,
    gen: int,
    BBH,
    flags,
    tmerg_myr,
    tdel_myr,
    tmig_s,
    AGN,
    hard_velocity,
    maverage,
) -> None:
    n_total = len(BBH.m1) if BBH is not None and hasattr(BBH, "m1") else 0
    reached_mask = np.asarray(flags.trap_reached_flag, dtype=int) == 1
    hard_mask = np.asarray(flags.hard, dtype=int) == 1
    reached = int(np.sum(reached_mask))
    mergers = int(np.sum(np.asarray(flags.merger_flag, dtype=int) == 1))
    continue_ng = int(np.sum(np.asarray(flags.continue_ng_flag, dtype=int) == 1))
    hard_total = int(np.sum(hard_mask))
    hard_reached = int(np.sum(hard_mask & reached_mask))
    trap_not_reached = _count_stop_reason(flags, const.STOP_TRAP_NOT_REACHED)
    soft = _count_stop_reason(flags, const.STOP_SOFT_BINARY)
    escaped = int(np.sum(np.asarray(flags.GWescape_flag, dtype=int) == 1))

    merger_mask = np.asarray(flags.merger_flag, dtype=int) == 1
    vk_values = np.asarray(getattr(BBH, "vk", []), dtype=float)
    mean_vk = _finite_mean(vk_values[merger_mask]) if vk_values.size else np.nan
    mean_tmig_myr = _finite_mean(np.asarray(tmig_s, dtype=float) / const.Myr)
    mean_tdel_myr = _finite_mean(tdel_myr)
    mean_tmerg_myr = _finite_mean(np.asarray(tmerg_myr, dtype=float)[merger_mask])
    tini_min_myr, tini_max_myr = _finite_minmax(
        np.asarray(getattr(BBH, "trapreach_time_s", []), dtype=float) / const.Myr
    )
    v_kep = makegen._kepler_speed_at_trap(AGN)
    a_hard = _hard_boundary_cm(BBH.m1, BBH.m2, maverage, hard_velocity)
    sma_over_ahard = np.asarray(BBH.sma, dtype=float) / a_hard
    median_sma_over_ahard = _finite_median(sma_over_ahard)

    _progress(
        sample_index,
        simulation_type,
        f"gen {gen}: N={n_total}, hard_flag={hard_total}, "
        f"trap_reached={reached}, hard_reached={hard_reached}, "
        f"trap_not_reached={trap_not_reached}, soft={soft}, mergers={mergers}, "
        f"continue_next={continue_ng}, escaped={escaped}; "
        f"median sma/a_hard={median_sma_over_ahard:.3e}, "
        f"v_hard={_format_speed(hard_velocity)}, "
        f"mean tmig={mean_tmig_myr:.3e} Myr, "
        f"tini range={tini_min_myr:.3e}-{tini_max_myr:.3e} Myr, "
        f"mean tdel={mean_tdel_myr:.3e} Myr, "
        f"mean tmerg(merged)={mean_tmerg_myr:.3e} Myr; "
        f"mean kick={_format_speed(mean_vk)}, "
        f"v_kep(trap)={_format_speed(v_kep)}, vesc={_format_speed(float(AGN.vesc))}",
    )


def _closest_index(values: np.ndarray, target: float) -> int:
    """Return the index of the grid point closest to a target radius."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("values must not be empty")
    return int(np.argmin(np.abs(values - target)))


def _build_agn_storage_dict(
    AGN,
    Mnsc: float,
    numstar_ini: int,
    dutycycle_myr: float,
    Mdot_cgs: float,
    r_list: np.ndarray,
    sigma_list: np.ndarray,
    h_list: np.ndarray,
    cs_list: np.ndarray,
    Q_list: np.ndarray,
) -> Dict[str, Any]:
    """Create a compact, plotting-friendly AGN metadata dictionary.

    These values are stored once per sample and later repeated onto event arrays
    by ``hdf5_load.py`` so every system can be traced back to its parent AGN.
    """
    trap_idx = _closest_index(r_list, AGN.rtrap)
    out_idx = _closest_index(r_list, AGN.r_out)

    return {
        "M_SMBH": float(AGN.M_SMBH),
        "rg": float(AGN.rg),
        "rtrap": float(AGN.rtrap),
        "rtrap_rg": float(AGN.rtrap / AGN.rg),
        "r_out": float(AGN.r_out),
        "r_out_rg": float(AGN.r_out / AGN.rg),
        "AGNlifetime_myr": float(dutycycle_myr),
        "AGNlifetime_s": float(dutycycle_myr * const.Myr),
        "alpha": float(getattr(AGN, "alpha", np.nan)),
        "vesc": float(AGN.vesc),
        "Mnsc": float(Mnsc),
        "numstar_finalbh": int(numstar_ini),
        "sigma_nsc_cm_s": float(getattr(AGN, "sigma", np.nan)),
        "Mdot_cgs": float(Mdot_cgs),
        "trap_count": int(getattr(AGN, "trap_count", 0)),
        "trap_policy": str(getattr(AGN, "trap_policy", "")),
        "trap_model": str(getattr(AGN, "trap_model", "")),
        "trap_migrator_mass_msun": float(
            getattr(AGN, "trap_migrator_mass_msun", np.nan)
        ),
        "trap_smoothing": float(getattr(AGN, "trap_smoothing", np.nan)),
        "trap_fallback_used": bool(getattr(AGN, "trap_fallback_used", False)),
        "trap_radii_rg_csv": ",".join(
            f"{float(r) / float(AGN.rg):.8e}"
            for r in np.asarray(getattr(AGN, "trap_radii", []), dtype=float)
            if np.isfinite(r)
        ),
        "h_trap": float(h_list[trap_idx]),
        "sigma_trap": float(sigma_list[trap_idx]),
        "cs_trap": float(cs_list[trap_idx]),
        "Q_trap": float(Q_list[trap_idx]),
        "sigma_outer": float(sigma_list[out_idx]),
        "h_outer": float(h_list[out_idx]),
    }


def _filter_next_generation_trap_return(
    current_BBH,
    current_flags,
    current_tmerg_myr: np.ndarray,
    continue_mask: np.ndarray,
    AGN,
    h_list: np.ndarray,
    sigma_list: np.ndarray,
    r_list: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return only remnants that can migrate back to the trap before AGN shutdown."""
    candidate_indices = np.where(np.asarray(continue_mask, dtype=bool))[
        0
    ]  # get indices of candidates flagged for next generation

    if candidate_indices.size == 0:
        return candidate_indices, np.array([], dtype=float), np.array([], dtype=float)

    remnant_birth_time_myr = np.asarray(current_tmerg_myr, dtype=float)[
        candidate_indices
    ]  # time of remnant birth = merger time of parent binary; this is the clock for migration back to trap

    remnant_masses = np.asarray(current_BBH.mmerg, dtype=float)[candidate_indices]
    remnant_orbits = np.asarray(current_BBH.A, dtype=float)[
        candidate_indices
    ]  # kicked radial distance, semi-analyticl with much simpler assumptions, e=0, theta=pi/2

    tmig_return_s = np.full(candidate_indices.size, np.inf, dtype=float)
    valid_inputs = (
        np.isfinite(remnant_birth_time_myr)
        & np.isfinite(remnant_masses)
        & np.isfinite(remnant_orbits)
        & (remnant_masses > 0.0)
        & (remnant_orbits > 0.0)
    )
    if np.any(valid_inputs):
        tmig_return_s[valid_inputs] = agn_f.t_migration(
            AGN,
            remnant_masses[valid_inputs],
            remnant_orbits[valid_inputs],
            h_list,
            sigma_list,
            r_list,
        )

    trap_return_time_s = remnant_birth_time_myr * const.Myr + tmig_return_s
    dutycycle_s = AGN.AGNlifetime * const.Myr
    can_pair_mask = np.isfinite(trap_return_time_s) & (
        trap_return_time_s <= dutycycle_s
    )

    rejected_indices = candidate_indices[~can_pair_mask]
    if rejected_indices.size:
        current_flags.continue_ng_flag[rejected_indices] = 0
        current_flags.stop_reason_flag[rejected_indices] = const.STOP_TRAP_NOT_REACHED

    return (
        candidate_indices[can_pair_mask],
        remnant_birth_time_myr[can_pair_mask],
        tmig_return_s[can_pair_mask],
    )


def run_physics_simulation(
    Msmbh: float,
    model_config: ModelConfig | str,
    temp_file: str,
    sample_index: int,
    sample_seed: int | None = None,
) -> Tuple[int, int, Dict[str, Any]]:
    """Run one SMBH sample through the full physics/storage channel.

    Inputs:
        Msmbh:
            SMBH mass in Msun for this sample.
        model_config:
            ``ModelConfig`` or legacy model name. Controls gas mode, disk model,
            radial distribution, damping-copy behavior, trap policy, generation
            limit, and storage metadata.
        temp_file:
            Worker-side pickle path for compact completion/error metadata.
        sample_index:
            Shared SMBH-grid index, preserved across model comparisons.
        sample_seed:
            Optional deterministic RNG seed for this model/sample task.

    Main outputs:
        ``(total_mergers, num_generations, metadata)``.

    Storage side effect:
        Each generation is written immediately to
        ``output/<model_name>/sample_XXXX_gen_YYY_*.h5`` with AGN and model
        metadata. Later, ``storedata.combine_all_files`` builds combined HDF5.

    Unit rule:
        Internal dynamics use seconds where possible. Stored timescale datasets
        use Myr, except explicitly suffixed second fields such as
        ``trapreach_time_s``.
    """
    if isinstance(model_config, str):
        model_config = legacy_model_config(model_config)
    simulation_type = model_config.model_name
    if sample_seed is not None:
        np.random.seed(int(sample_seed))

    logger = None
    try:
        pid = os.getpid()
        if diag is not None and diag.debug_enabled():
            logger = diag.setup_sample_logger(simulation_type, sample_index)
            logger.info(
                "Starting sample=%s type=%s Msmbh=%.6e pid=%s",
                sample_index,
                simulation_type,
                Msmbh,
                pid,
            )
        _progress(
            sample_index,
            simulation_type,
            f"start: Msmbh={Msmbh:.3e} Msun",
        )

        gf = agn.general_flags()
        output_dir = os.path.join(gf.out_met, simulation_type)
        save_context = SampleSaveContext(
            sample_index=int(sample_index),
            smbh_mass=float(Msmbh),
            simulation_type=simulation_type,
            output_dir=output_dir,
            temp_file=temp_file,
            model_config=model_config,
            pid=pid,
        )

        Mnsc = agn_f.m_NSC(Msmbh)
        numstar_ini = agn_f.num_Star(Msmbh)

        # ------------------------------------------------------------------
        # Build the AGN environment for this SMBH sample
        # ------------------------------------------------------------------

        AGN = agn.AGN_properties()
        dutycycle_myr = float(model_config.agn_lifetime_myr)
        AGN.init_AGN(Msmbh, dutycycle_myr)

        mdot = 0.1  # Eddington ratio; this is a placeholder and can be randomized or parameterized as needed
        Mdot_cgs = Msmbh * mdot * 1.44e18
        pagn_n_resolution = int(os.getenv("PAGN_DISK_N_RESOLUTION", "10000"))
        profile = build_pagn_disk_profile(
            Msmbh,
            model=model_config.disk_model,
            n_resolution=pagn_n_resolution,
            eddington_ratio=mdot,
            alpha=float(AGN.alpha),
            b=0,
        )

        params = MigrationParameters(alpha=float(AGN.alpha))
        if model_config.trap_mode == "fixed":
            rtrap = float(model_config.fixed_trap_rg) * AGN.rg
            trap_radii = np.asarray([rtrap], dtype=float)
            torque_profile = None
        elif model_config.trap_mode in {"torque_outermost", "torque_stable"}:
            rtrap, trap_radii, torque_profile = find_outermost_stable_trap(
                profile.disk,
                migrator_mass_msun=10.0,
                smoothing=0.005,
                params=params,
                fallback_radius_cm=1.0e3 * AGN.rg,
            )
        else:
            raise ValueError(f"Unknown trap_mode: {model_config.trap_mode}")

        r_list = profile.r_list
        Tc_list = profile.Tc_list
        Q_list = profile.Q_list
        sigma_list = profile.sigma_list
        tau_list = profile.tau_list
        h_list = profile.h_list
        cs_list = profile.cs_list
        rho_list = profile.rho_list

        AGN.r_out = profile.r_out
        AGN.rtrap = rtrap
        AGN.trap_radii = trap_radii
        AGN.trap_count = int(len(trap_radii))
        AGN.trap_policy = model_config.trap_mode
        AGN.trap_model = profile.model_name
        AGN.trap_migrator_mass_msun = 10.0
        AGN.trap_smoothing = 0.005
        AGN.trap_fallback_used = bool(len(trap_radii) == 0)
        AGN.torque_profile = torque_profile
        if diag is not None and diag.debug_enabled():
            diag.dump_disk_diagnostics(
                simulation_type,
                sample_index,
                AGN,
                profile,
                r_list,
                Tc_list,
                Q_list,
                sigma_list,
                tau_list,
                h_list,
                cs_list,
                rho_list,
                Mdot_cgs=Mdot_cgs,
                dutycycle_myr=dutycycle_myr,
                logger=logger,
            )
            if torque_profile is not None:
                diag.dump_torque_diagnostics(
                    simulation_type,
                    sample_index,
                    AGN,
                    torque_profile,
                    trap_radii,
                    rtrap,
                    logger=logger,
                )
        if AGN.trap_fallback_used:
            print(
                f"WARNING: no stable torque trap found for Msmbh={Msmbh:.3e} Msun; "
                f"using fallback rtrap={AGN.rtrap / AGN.rg:.3e} rg"
            )
        AGN.vesc, M_tot = agn_f.vesc(Mnsc, AGN)
        hard_velocity = _trap_hardness_velocity(AGN, h_list, cs_list, r_list)
        AGN.hard_velocity = hard_velocity

        agn_properties = _build_agn_storage_dict(
            AGN=AGN,
            Mnsc=Mnsc,
            numstar_ini=numstar_ini,
            dutycycle_myr=dutycycle_myr,
            Mdot_cgs=Mdot_cgs,
            r_list=np.asarray(r_list),
            sigma_list=np.asarray(sigma_list),
            h_list=np.asarray(h_list),
            cs_list=np.asarray(cs_list),
            Q_list=np.asarray(Q_list),
        )
        save_context.agn_properties = agn_properties
        _progress(
            sample_index,
            simulation_type,
            "AGN/disk: "
            f"numstar_ini={numstar_ini:.3e}, "
            f"rtrap={rtrap / AGN.rg:.3e} rg, "
            f"rout={AGN.r_out / AGN.rg:.3e} rg, "
            f"trap_count={AGN.trap_count}, "
            f"v_hard={_format_speed(hard_velocity)}, "
            f"vesc={_format_speed(float(AGN.vesc))}",
        )

        # ------------------------------------------------------------------
        # Seed the stellar/BH reservoir and run damping into the disk
        # ------------------------------------------------------------------

        Stars, AGN, mBHaverage, max1g, tform = initi.initialize_stellar(
            numstar_ini, AGN, gf.in_dir, gf
        )

        result = make_damp.apply_damping_channel(
            model_config,
            Stars,
            AGN,
            tform,
            agn.damp_flag(numstar_ini),
            h_list,
            sigma_list,
            r_list,
        )

        BBH, Stars_disk, combined_data = extract_damping_results(
            result, mode=result.get("selected_mode", "combined")
        )
        if diag is not None and diag.debug_enabled():
            diag.dump_damping_diagnostics(
                simulation_type,
                sample_index,
                result,
                combined_data,
                BBH,
                Stars_disk,
                dutycycle_myr=dutycycle_myr,
                logger=logger,
            )
        star_tform_myr, star_tdamp_myr, star_trel_myr = extract_timing_data(
            combined_data, "stars"
        )
        bh_tform_myr, bh_tdamp_myr, bh_trel_myr = extract_timing_data(
            combined_data, "bh"
        )

        mask = (
            star_tform_myr < dutycycle_myr
        )  # stars with tform beyond AGN lifetime are omitted
        N1 = len(BBH.m1) if BBH is not None and hasattr(BBH, "m1") else 0
        N2 = (
            len(Stars_disk.m[mask])
            if (
                Stars_disk is not None
                and hasattr(Stars_disk, "m")
                and len(star_tform_myr) > 0
            )
            else 0
        )

        _print_damping_progress(
            sample_index,
            simulation_type,
            result,
            N1,
            N2,
        )

        if N1 + N2 <= 0:
            return _handle_empty_case(save_context)

        # ------------------------------------------------------------------
        # Build and evolve the first generation
        # ------------------------------------------------------------------

        BBH_1g, tmig_s_1g, initial_valid_systems = create_and_populate_bbh_properties(
            N1,
            N2,
            BBH,
            Stars_disk,
            mask,
            (star_tform_myr, star_tdamp_myr, star_trel_myr),
            AGN,
            gf,
            model_config,
            h_list,
            sigma_list,
            r_list,
            return_valid_mask=True,
        )

        if diag is not None and diag.debug_enabled():
            diag.dump_initial_population_diagnostics(
                simulation_type,
                sample_index,
                1,
                BBH_1g,
                tmig_s_1g,
                AGN=AGN,
                logger=logger,
            )
            diag.dump_migration_diagnostics(
                simulation_type,
                sample_index,
                1,
                AGN,
                BBH_1g,
                tmig_s_1g,
                h_list,
                sigma_list,
                r_list,
                logger=logger,
            )

        _progress(
            sample_index,
            simulation_type,
            f"first-gen initialized: N={len(BBH_1g.m1)}; "
            f"mean migration={_finite_mean(tmig_s_1g) / const.Myr:.3e} Myr\n",
        )

        total_mergers = 0
        num_generations = 0

        if len(BBH_1g.m1) > 0:
            raw_tform_1g_myr = (
                np.concatenate([bh_tform_myr, star_tform_myr[mask]])
                if (len(bh_tform_myr) > 0 or len(star_tform_myr[mask]) > 0)
                else np.array([])
            )
            raw_tdamp_1g_myr = (
                np.concatenate([bh_tdamp_myr, star_tdamp_myr[mask]])
                if (len(bh_tdamp_myr) > 0 or len(star_tdamp_myr[mask]) > 0)
                else np.array([])
            )
            raw_trel_1g_myr = (
                np.concatenate([bh_trel_myr, star_trel_myr[mask]])
                if (len(bh_trel_myr) > 0 or len(star_trel_myr[mask]) > 0)
                else np.array([])
            )
            tform_1g_myr = _filter_initial_timescale(
                raw_tform_1g_myr, initial_valid_systems, len(BBH_1g.m1)
            )
            tdamp_1g_myr = _filter_initial_timescale(
                raw_tdamp_1g_myr, initial_valid_systems, len(BBH_1g.m1)
            )
            trel_1g_myr = _filter_initial_timescale(
                raw_trel_1g_myr, initial_valid_systems, len(BBH_1g.m1)
            )

            tdamp_1g_s = np.asarray(tdamp_1g_myr, dtype=float) * const.Myr

            # compute trapreach time
            BBH_1g.trapreach_time_s = np.asarray(tmig_s_1g, dtype=float) + tdamp_1g_s

            flags_1g = agn.binary_flags(len(BBH_1g.m1))
            # Set orbital properties
            BBH_1g.sma = auxi.semimajor(BBH_1g.m1, BBH_1g.m2, AGN)
            BBH_1g.ecc = auxi.ecc(BBH_1g.m1)

            flags_1g.hard = auxi.check_hard(
                BBH_1g.m1,
                BBH_1g.m2,
                BBH_1g.sma,
                mBHaverage,
                hard_velocity,
            )

            # Set orbital inclinations
            th1 = np.random.rand(len(BBH_1g.m1))
            th2 = np.random.rand(len(BBH_1g.m1))

            if model_config.gas_mode == "without_gas":
                BBH_1g.th1 = auxi.set_theta(th1)
                BBH_1g.th2 = auxi.set_theta(th2)
            elif model_config.gas_mode == "with_gas":
                BBH_1g.th1 = auxi.set_theta_gas(th1)
                BBH_1g.th2 = auxi.set_theta_gas(th2)
            else:
                raise ValueError(f"Unknown gas_mode: {model_config.gas_mode}")

            BBH_1g, flags_1g, tmerg_1g_myr, tdel_1g_myr = makegen.make_generation(
                BBH_1g,
                AGN,
                flags_1g,
                model_config,
                1,
                r_list,
                sigma_list,
                h_list,
                cs_list,
            )

            timescales_1g = {
                "tform_myr": tform_1g_myr,
                "tdamp_myr": tdamp_1g_myr,
                "trel_myr": trel_1g_myr,
                "tmig_myr": np.asarray(tmig_s_1g, dtype=float) / const.Myr,
                "tini_myr": np.asarray(BBH_1g.trapreach_time_s, dtype=float)
                / const.Myr,
                "tmerg_myr": tmerg_1g_myr,
                "tdel_myr": tdel_1g_myr,
                "t_bsi_myr": np.asarray(
                    getattr(BBH_1g, "bsi_timescale_s", []), dtype=float
                )
                / const.Myr,
            }

            _print_generation_progress(
                sample_index,
                simulation_type,
                1,
                BBH_1g,
                flags_1g,
                tmerg_1g_myr,
                tdel_1g_myr,
                tmig_s_1g,
                AGN,
                hard_velocity,
                mBHaverage,
            )

            # Diagnostics before storage
            if diag is not None and diag.debug_enabled():
                diag.dump_generation_diagnostics(
                    simulation_type,
                    sample_index,
                    1,  # first generation
                    BBH_1g,
                    flags_1g,
                    timescales_1g,
                    logger=logger,
                )

            generation_data = _create_generation_data(
                generation=1,
                BBH=BBH_1g,
                flags=flags_1g,
                M_tot=M_tot,
                generation_type="first_gen",
                timescales=timescales_1g,
                trap_population=getattr(AGN, "trap_population_latest", None),
                model_config=model_config,
            )

            num_generations += 1

            _save_generation_immediate(save_context, generation_data)
            total_mergers += int(np.sum(flags_1g.merger_flag))

            _progress(
                sample_index,
                simulation_type,
                f"saved gen 1: N={len(generation_data['BBH'].m1)}, "
                f"mergers={int(np.sum(flags_1g.merger_flag))}",
            )

            current_BBH = BBH_1g
            current_flags = flags_1g
            current_tmerg_myr = np.asarray(tmerg_1g_myr, dtype=float)
            gen = 2  # next generation index

            maxgen = len(
                BBH_1g.m1
            )  # upper limit on generation, BH number limited to number of 1g BHs

            # ------------------------------------------------------------------
            # Recycle merger remnants into higher generations while allowed
            # ------------------------------------------------------------------
            while (
                getattr(
                    current_flags, "continue_ng_flag", np.array([])
                ).size  # continue flag
                and np.any(current_flags.continue_ng_flag == 1)
                and gen <= maxgen
            ):
                continue_mask = (
                    current_flags.continue_ng_flag == 1
                )  # get first continue mask

                if not np.any(continue_mask):
                    _progress(
                        sample_index,
                        simulation_type,
                        f"stop hierarchy: no systems continue after gen {gen - 1}",
                    )
                    break

                next_gen_candidates = int(np.sum(continue_mask))
                (
                    merger_indices,
                    remnant_birth_time_myr,
                    tmig_s_ng,
                ) = _filter_next_generation_trap_return(
                    current_BBH,
                    current_flags,
                    current_tmerg_myr,
                    continue_mask,
                    AGN,
                    h_list,
                    sigma_list,
                    r_list,
                )

                if remnant_birth_time_myr.size == 0:
                    _progress(
                        sample_index,
                        simulation_type,
                        f"gen {gen}: rejected all {next_gen_candidates} "
                        "candidate remnants before pairing; trap not reached in time",
                    )
                    break

                next_gen_count = remnant_birth_time_myr.size

                rejected_before_pairing = next_gen_candidates - next_gen_count
                if rejected_before_pairing:
                    _progress(
                        sample_index,
                        simulation_type,
                        f"gen {gen}: candidates={next_gen_candidates}, "
                        f"paired={next_gen_count}, "
                        f"rejected_before_pairing={rejected_before_pairing}",
                    )
                new_BBH = agn.BBH_properties(next_gen_count)

                # Carry remnant mass/spin/orbital location into the next
                # generation before reinitializing the secondary component.
                for j, src_idx in enumerate(merger_indices):
                    new_BBH.m1[j] = current_BBH.mmerg[src_idx]
                    new_BBH.chi1[j] = current_BBH.chimerg[src_idx]
                    new_BBH.A[j] = current_BBH.A[src_idx]

                new_BBH = initi.initialize_nthgen(new_BBH, AGN, max1g, gf, model_config)

                _progress(
                    sample_index,
                    simulation_type,
                    f"nth-gen initialized: gen={gen}, N={next_gen_count}, "
                    f"mean return migration={_finite_mean(tmig_s_ng) / const.Myr:.3e} Myr\n",
                )

                if diag is not None and diag.debug_enabled():
                    diag.dump_migration_diagnostics(
                        simulation_type,
                        sample_index,
                        gen,
                        AGN,
                        new_BBH,
                        tmig_s_ng,
                        h_list,
                        sigma_list,
                        r_list,
                        logger=logger,
                    )

                new_BBH.trapreach_time_s = np.asarray(
                    remnant_birth_time_myr, dtype=float
                ) * const.Myr + np.asarray(tmig_s_ng, dtype=float)

                new_flags = agn.binary_flags(next_gen_count)
                new_flags.hard = auxi.check_hard(
                    new_BBH.m1,
                    new_BBH.m2,
                    new_BBH.sma,
                    AGN.maverage,
                    hard_velocity,
                )

                new_BBH, new_flags, tmerg_ng_myr, tdel_ng_myr = makegen.make_generation(
                    new_BBH,
                    AGN,
                    new_flags,
                    model_config,
                    gen,
                    r_list,
                    sigma_list,
                    h_list,
                    cs_list,
                )

                timescales_ng = {
                    "tmig_myr": np.asarray(tmig_s_ng, dtype=float) / const.Myr,
                    "tini_myr": np.asarray(new_BBH.trapreach_time_s, dtype=float)
                    / const.Myr,
                    "tmerg_myr": tmerg_ng_myr,
                    "tdel_myr": tdel_ng_myr,
                    "t_bsi_myr": np.asarray(
                        getattr(new_BBH, "bsi_timescale_s", []), dtype=float
                    )
                    / const.Myr,
                }

                _print_generation_progress(
                    sample_index,
                    simulation_type,
                    gen,
                    new_BBH,
                    new_flags,
                    tmerg_ng_myr,
                    tdel_ng_myr,
                    tmig_s_ng,
                    AGN,
                    hard_velocity,
                    AGN.maverage,
                )

                if diag is not None and diag.debug_enabled():
                    diag.dump_generation_diagnostics(
                        simulation_type,
                        sample_index,
                        gen,
                        new_BBH,
                        new_flags,
                        timescales_ng,
                        logger=logger,
                    )

                generation_data = _create_generation_data(
                    generation=gen,
                    BBH=new_BBH,
                    flags=new_flags,
                    M_tot=M_tot,
                    # Ngen=Ngen_ng,
                    generation_type="n_gen",
                    timescales=timescales_ng,
                    trap_population=getattr(AGN, "trap_population_latest", None),
                    model_config=model_config,
                )
                num_generations += 1

                _save_generation_immediate(save_context, generation_data)
                total_mergers += int(np.sum(new_flags.merger_flag))

                _progress(
                    sample_index,
                    simulation_type,
                    f"saved gen {gen}: N={len(generation_data['BBH'].m1)}, "
                    f"mergers={int(np.sum(new_flags.merger_flag))}, "
                    f"mean migration={_finite_mean(tmig_s_ng) / const.Myr:.3e} Myr",
                )
                current_BBH = new_BBH

                current_flags = new_flags
                current_tmerg_myr = np.asarray(tmerg_ng_myr, dtype=float)
                gen += 1

        metadata = _build_sample_metadata(
            save_context,
            total_mergers=total_mergers,
            num_generations=num_generations,
            completed=True,
        )
        _save_sample_manifest(save_context, num_generations, total_mergers)
        _write_temp_metadata(save_context, metadata)
        _progress(
            sample_index,
            simulation_type,
            f"done: total_mergers={total_mergers}, " f"generations={num_generations}",
        )
        return total_mergers, num_generations, metadata

    except Exception as exc:
        if logger is not None:
            logger.exception("Sample failed")
        print(f"[PID:{os.getpid()}] ERROR in sample {sample_index}: {exc}", flush=True)
        metadata = {
            "sample_index": sample_index,
            "smbh_mass_msun": float(Msmbh),
            "simulation_type": simulation_type,
            "model_config": model_config.to_dict(),
            "completed": False,
            "error": str(exc),
        }
        _write_temp_metadata(temp_file, metadata)
        return 0, 0, metadata


def _create_generation_data(
    generation: int,
    BBH,
    flags,
    M_tot,
    generation_type: str,
    timescales: Dict[str, Any],
    trap_population: Dict[str, Any] | None = None,
    model_config: ModelConfig | None = None,
) -> Dict[str, Any]:
    """Build the minimal generation payload expected by the storage layer."""
    return {
        "generation": int(generation),
        "BBH": BBH,
        "flags": flags,
        "M_tot": M_tot,
        # "Ngen": Ngen if Ngen is not None else np.array([]),
        "generation_type": generation_type,
        "timescales": timescales,
        "trap_population": trap_population,
        "model_config": model_config,
    }


def _save_generation_immediate(
    save_context: SampleSaveContext, generation_data: Dict[str, Any]
):
    """Persist one generation immediately as an HDF5 fragment."""
    if not STORAGE_AVAILABLE:
        return None
    gen_data_with_agn = dict(generation_data)
    gen_data_with_agn["AGN"] = save_context.agn_properties or {}
    gen_data_with_agn["smbh_mass"] = float(save_context.smbh_mass)
    gen_data_with_agn["simulation_type"] = save_context.simulation_type
    return std.save_generation_to_file(
        sample_index=save_context.sample_index,
        generation_num=int(generation_data["generation"]),
        generation_data=gen_data_with_agn,
        output_dir=save_context.output_dir,
    )


def _save_sample_manifest(
    save_context: SampleSaveContext,
    num_generations: int,
    num_mergers: int,
):
    """Write a small JSON summary for one completed sample."""
    manifest = {
        "sample_index": int(save_context.sample_index),
        "simulation_type": save_context.simulation_type,
        "smbh_mass_msun": float(save_context.smbh_mass),
        "num_generations": int(num_generations),
        "num_mergers": int(num_mergers),
        "model_config": save_context.model_config_dict(),
        "process_id": int(save_context.pid),
        "completion_time": time.time(),
        "agn_properties": save_context.agn_properties or {},
        "status": "completed",
    }
    os.makedirs(save_context.output_dir, exist_ok=True)
    manifest_file = os.path.join(
        save_context.output_dir,
        f"manifest_sample_{save_context.sample_index:04d}.json",
    )
    with open(manifest_file, "w", encoding="utf-8") as handle:
        import json

        json.dump(manifest, handle, indent=2, default=str)


def _build_sample_metadata(
    save_context: SampleSaveContext,
    *,
    total_mergers: int,
    num_generations: int,
    completed: bool,
    **extra,
) -> Dict[str, Any]:
    metadata = {
        "sample_index": int(save_context.sample_index),
        "smbh_mass_msun": float(save_context.smbh_mass),
        "simulation_type": save_context.simulation_type,
        "model_config": save_context.model_config_dict(),
        "total_mergers": int(total_mergers),
        "num_generations": int(num_generations),
        "completed": bool(completed),
    }
    if save_context.agn_properties is not None:
        metadata["agn_properties"] = save_context.agn_properties
    metadata.update(extra)
    return metadata


def _handle_empty_case(save_context: SampleSaveContext):
    """Handle samples that produce no initial in-disk BH population."""
    metadata = _build_sample_metadata(
        save_context,
        total_mergers=0,
        num_generations=0,
        completed=True,
        empty=True,
    )
    _save_sample_manifest(save_context, num_generations=0, num_mergers=0)
    _write_temp_metadata(save_context, metadata)
    return 0, 0, metadata


def _write_temp_metadata(
    save_context_or_path: SampleSaveContext | str, metadata: Dict[str, Any]
) -> None:
    """Serialize worker metadata for the parent process/debugging tools."""
    temp_file = (
        save_context_or_path.temp_file
        if isinstance(save_context_or_path, SampleSaveContext)
        else save_context_or_path
    )
    if not temp_file:
        return
    (
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        if os.path.dirname(temp_file)
        else None
    )
    with open(temp_file, "wb") as handle:
        pickle.dump(metadata, handle)


# ---------------------------------------------------------------------
# Existing helper API placeholders:
# Keep signatures stable with the project codebase. If these helpers already
# exist elsewhere in your local version, you can keep your richer versions.
# ---------------------------------------------------------------------


def extract_damping_results(result, mode="combined"):
    """
    Normalize the make_damp.make_damping(...) return payload to the historical
    3-tuple expected by the simulation driver:

        (BBH_obj, Stars_disk_obj, combined_data_dict)

    The refactored make_damp layer returns a dictionary with keys like
    'BBH_combined', 'Stars_disk_combined', and 'combined_data'. Returning the
    raw dict here causes Python to unpack dict keys instead of objects, which
    triggers:

        ValueError: too many values to unpack (expected 3)
    """
    if result is None:
        return None, None, {}

    if not isinstance(result, dict):
        raise TypeError(
            f"extract_damping_results expected dict from make_damp.make_damping, got {type(result).__name__}"
        )

    mode = str(mode)
    bbh_key = f"BBH_{mode}"
    stars_key = f"Stars_disk_{mode}"
    data_key = f"{mode}_data"

    # Primary path for the current make_damp output schema
    if bbh_key in result or stars_key in result or data_key in result:
        return (
            result.get(bbh_key),
            result.get(stars_key),
            result.get(data_key, {}),
        )

    # Compatibility fallback for older payload names
    if mode == "combined":
        return (
            result.get("BBH_combined", result.get("BBH_weighted")),
            result.get("Stars_disk_combined", result.get("Stars_disk_weighted")),
            result.get("combined_data", result.get("weighted_data", {})),
        )

    raise KeyError(
        f"Could not extract damping results for mode='{mode}'. Available keys: {sorted(result.keys())}"
    )


def extract_timing_data(combined_data, population_key: str):
    """Extract formation, damping, and relaxation times for one population."""
    if population_key not in combined_data:
        return np.array([]), np.array([]), np.array([])
    payload = combined_data[population_key]
    return (
        np.asarray(payload.get("tform", []), dtype=float),
        np.asarray(payload.get("tdamp", []), dtype=float),
        np.asarray(payload.get("trel", []), dtype=float),
    )


def _filter_initial_timescale(values, valid_systems, target_len):
    """Keep first-generation timescales aligned with the compacted BBH rows."""
    values = np.asarray(values, dtype=float)
    valid_systems = np.asarray(valid_systems, dtype=bool)

    if values.size == valid_systems.size:
        return values[valid_systems]
    if values.size == target_len:
        return values

    aligned = np.zeros(target_len, dtype=float)
    n_copy = min(values.size, target_len)
    if n_copy > 0:
        aligned[:n_copy] = values[:n_copy]
    return aligned


def _selected_stellar_indices(Stars_disk, mask, n_requested):
    """Return stellar-origin rows available for first-generation BBH seeding."""
    if n_requested <= 0 or Stars_disk is None or not hasattr(Stars_disk, "mBH"):
        return np.array([], dtype=int)

    n_stars = len(Stars_disk.mBH)
    if mask is None or len(mask) == 0:
        return np.arange(min(n_requested, n_stars), dtype=int)

    mask_array = np.asarray(mask)
    if mask_array.dtype == bool:
        candidate_indices = np.where(mask_array)[0]
    else:
        candidate_indices = mask_array.astype(int, copy=False)

    candidate_indices = candidate_indices[
        (candidate_indices >= 0) & (candidate_indices < n_stars)
    ]
    return candidate_indices[:n_requested].astype(int, copy=False)


def _stellar_timescale_seconds(timescale_myr, Stars_disk, attr_name, indices):
    """Read a stellar timescale array and convert it to seconds."""
    if indices.size == 0:
        return None

    source = np.asarray(timescale_myr, dtype=float)
    max_index = int(np.max(indices))
    if source.size > 0 and max_index < source.size:
        return source[indices] * const.Myr

    fallback = np.asarray(getattr(Stars_disk, attr_name, []), dtype=float)
    if fallback.size > 0 and max_index < fallback.size:
        return fallback[indices] * const.Myr

    return None


def _finalize_population(BBH_obj, tmig_s, valid_systems, return_valid_mask=False):
    """Optionally compact a population while preserving the old two-value API."""
    tmig_s = np.asarray(tmig_s, dtype=float)
    valid_systems = np.asarray(valid_systems, dtype=bool)
    total_systems = valid_systems.size

    if total_systems == 0:
        result = (agn.BBH_properties(0), np.array([], dtype=float))
        return (*result, valid_systems) if return_valid_mask else result

    if np.all(valid_systems):
        result = (BBH_obj, tmig_s)
        return (*result, valid_systems) if return_valid_mask else result

    n_valid = int(np.sum(valid_systems))
    if n_valid == 0:
        result = (agn.BBH_properties(0), np.array([], dtype=float))
        return (*result, valid_systems) if return_valid_mask else result

    filtered = agn.BBH_properties(n_valid)
    for attr in dir(BBH_obj):
        if attr.startswith("_"):
            continue
        value = getattr(BBH_obj, attr)
        if callable(value):
            continue
        if isinstance(value, np.ndarray) and len(value) == total_systems:
            setattr(filtered, attr, value[valid_systems])

    result = (filtered, tmig_s[valid_systems])
    return (*result, valid_systems) if return_valid_mask else result


def _legacy_fractional_migration_radius(AGN, start_radius, elapsed_s, tmig_s):
    """Fallback radius estimate for the local, non-integrated migration clock."""
    if elapsed_s <= 0.0 or not np.isfinite(tmig_s) or tmig_s <= 0.0:
        return float(start_radius)

    trap_r = getattr(AGN, "rtrap", None)
    if trap_r is None or not np.isfinite(trap_r) or trap_r <= 0.0:
        return float(start_radius)

    factor = np.clip(1.0 - elapsed_s / tmig_s, 0.0, 1.0)
    return float((start_radius - trap_r) * factor + trap_r)


def _trap_target_radius(trap_r, torque_radius, side, use_resolved_trap):
    if not use_resolved_trap:
        return float(trap_r)

    if side == "outer":
        candidates = torque_radius[torque_radius > trap_r]
        return float(candidates[0]) if candidates.size > 0 else float(trap_r)

    candidates = torque_radius[torque_radius < trap_r]
    return float(candidates[-1]) if candidates.size > 0 else float(trap_r)


def _build_torque_migration_curve(AGN, torque_radius, gamma_tot, query_radii, side):
    """Build one cumulative reference-mass migration-time curve for a trap side."""
    trap_r = getattr(AGN, "rtrap", None)
    if trap_r is None or not np.isfinite(trap_r) or trap_r <= 0.0:
        return None

    query_radii = np.asarray(query_radii, dtype=float)
    if query_radii.size == 0:
        return None

    use_resolved_trap = not bool(getattr(AGN, "trap_fallback_used", False))
    target_r = _trap_target_radius(trap_r, torque_radius, side, use_resolved_trap)

    if side == "outer":
        away_r = float(np.max(query_radii))
        if away_r <= target_r:
            return {
                "target": target_r,
                "radius": np.array([target_r]),
                "cum": np.array([0.0]),
            }
        interior = torque_radius[(torque_radius > target_r) & (torque_radius < away_r)]
        queried = query_radii[(query_radii > target_r) & (query_radii <= away_r)]
        path_r = np.unique(np.concatenate(([target_r], interior, queried)))
    else:
        away_r = float(np.min(query_radii))
        if away_r >= target_r:
            return {
                "target": target_r,
                "radius": np.array([target_r]),
                "cum": np.array([0.0]),
            }
        interior = torque_radius[(torque_radius > away_r) & (torque_radius < target_r)]
        queried = query_radii[(query_radii >= away_r) & (query_radii < target_r)]
        path_r = np.unique(np.concatenate((queried, interior, [target_r])))[::-1]

    if path_r.size < 2:
        return None

    path_gamma = np.interp(path_r, torque_radius, gamma_tot)
    path_sign = np.sign(path_gamma)
    finite = np.isfinite(path_gamma) & (path_sign != 0.0)
    if not finite[0]:
        return None

    sign0 = path_sign[0]
    usable = finite & (path_sign == sign0)
    if not np.all(usable):
        first_bad = int(np.where(~usable)[0][0])
        path_r = path_r[:first_bad]
        path_gamma = path_gamma[:first_bad]

    if path_r.size < 2:
        return None

    reference_mass_msun = float(getattr(AGN, "trap_migrator_mass_msun", 10.0))
    if reference_mass_msun <= 0.0:
        reference_mass_msun = 10.0

    gamma_cgs = path_gamma * 1.0e7
    abs_gamma = np.abs(gamma_cgs)
    if np.any(abs_gamma <= 0.0) or not np.all(np.isfinite(abs_gamma)):
        return None

    M_SMBH = AGN.M_SMBH * const.msun
    mbh_ref = reference_mass_msun * const.msun
    omega = np.sqrt(const.G * M_SMBH / path_r**3.0)
    integrand = mbh_ref * path_r * omega / (2.0 * abs_gamma)
    if not np.all(np.isfinite(integrand)):
        return None

    segment_dt = 0.5 * (integrand[:-1] + integrand[1:]) * np.abs(np.diff(path_r))
    cumulative = np.concatenate(([0.0], np.cumsum(segment_dt)))
    return {
        "target": target_r,
        "radius": path_r,
        "cum": cumulative,
        "reference_mass_msun": reference_mass_msun,
    }


def _time_from_torque_curve(curve, radii):
    path_r = curve["radius"]
    cumulative = curve["cum"]
    if path_r[0] < path_r[-1]:
        return np.interp(radii, path_r, cumulative)
    return np.interp(radii, path_r[::-1], cumulative[::-1])


def _radius_from_torque_curve_time(curve, reference_time_s):
    return np.interp(reference_time_s, curve["cum"], curve["radius"])


def _migration_times_and_radii_after_elapsed(
    AGN,
    masses_msun,
    start_radii,
    elapsed_s,
    h_list,
    sigma_list,
    r_list,
):
    """Return migration times and elapsed-migration radii in one pass."""
    masses_msun = np.asarray(masses_msun, dtype=float)
    start_radii = np.asarray(start_radii, dtype=float)
    elapsed_s = np.asarray(elapsed_s, dtype=float)

    tmig = np.full(start_radii.size, np.inf, dtype=np.float64)
    radius_after_elapsed = start_radii.astype(np.float64).copy()

    valid = (masses_msun > 0.0) & (start_radii > 0.0)
    if not np.any(valid):
        return tmig, radius_after_elapsed

    torque_arrays = agn_f._sorted_torque_profile_arrays(AGN)
    trap_r = getattr(AGN, "rtrap", None)
    if (
        torque_arrays is None
        or trap_r is None
        or not np.isfinite(trap_r)
        or trap_r <= 0.0
    ):
        # return default migration parameters
        tmig[valid] = agn_f.t_migration(
            AGN, masses_msun[valid], start_radii[valid], h_list, sigma_list, r_list
        )
        for idx in np.where(valid)[0]:
            radius_after_elapsed[idx] = _legacy_fractional_migration_radius(
                AGN, start_radii[idx], elapsed_s[idx], tmig[idx]
            )
        return tmig, radius_after_elapsed

    # calculate migration times and elapsed-migration radii using the resolved torque profile
    torque_radius, gamma_tot = torque_arrays
    side_masks = {
        "outer": valid & (start_radii > trap_r),
        "inner": valid & (start_radii < trap_r),
    }
    at_trap = valid & np.isclose(
        start_radii,
        trap_r,
        rtol=0.0,
        atol=np.finfo(float).eps * np.maximum(start_radii, trap_r),
    )
    tmig[at_trap] = 0.0

    for side, side_mask in side_masks.items():
        if not np.any(side_mask):
            continue

        curve = _build_torque_migration_curve(
            AGN, torque_radius, gamma_tot, start_radii[side_mask], side
        )

        if curve is None:
            tmig[side_mask] = agn_f.t_migration(
                AGN,
                masses_msun[side_mask],
                start_radii[side_mask],
                h_list,
                sigma_list,
                r_list,
            )
            for idx in np.where(side_mask)[0]:
                radius_after_elapsed[idx] = _legacy_fractional_migration_radius(
                    AGN, start_radii[idx], elapsed_s[idx], tmig[idx]
                )
            continue

        side_idx = np.where(side_mask)[0]
        target_r = curve["target"]
        if side == "outer":
            at_target = start_radii[side_idx] <= target_r
            within_curve = start_radii[side_idx] <= np.max(curve["radius"])
        else:
            at_target = start_radii[side_idx] >= target_r
            within_curve = start_radii[side_idx] >= np.min(curve["radius"])

        target_idx = side_idx[at_target]
        tmig[target_idx] = 0.0
        radius_after_elapsed[target_idx] = target_r

        active_idx = side_idx[~at_target & within_curve]
        if active_idx.size == 0:
            continue

        reference_tmig = _time_from_torque_curve(curve, start_radii[active_idx])
        mass_scale = curve["reference_mass_msun"] / masses_msun[active_idx]
        tmig[active_idx] = reference_tmig * mass_scale

        reaches_target = elapsed_s[active_idx] >= tmig[active_idx]
        radius_after_elapsed[active_idx[reaches_target]] = target_r

        partial_idx = active_idx[~reaches_target & (elapsed_s[active_idx] > 0.0)]
        if partial_idx.size > 0:
            remaining_reference_time = (
                (tmig[partial_idx] - elapsed_s[partial_idx])
                * masses_msun[partial_idx]
                / curve["reference_mass_msun"]
            )
            radius_after_elapsed[partial_idx] = _radius_from_torque_curve_time(
                curve, remaining_reference_time
            )

    return tmig, radius_after_elapsed


def create_and_populate_bbh_properties(
    N1,
    N2,
    BBH,
    Stars_disk,
    mask,
    star_timescales_myr,
    AGN,
    gf,
    model_config: ModelConfig | str,
    h_list,
    sigma_list,
    r_list,
    return_valid_mask=False,
):
    """Construct the first-generation BBH population for one scenario.

    The first generation can contain both BHs already in the disk and stellar
    progenitors that damp into the disk before the AGN duty cycle ends.
    """
    total_systems = N1 + N2
    if total_systems == 0:
        return _finalize_population(
            agn.BBH_properties(0),
            np.array([], dtype=float),
            np.array([], dtype=bool),
            return_valid_mask,
        )

    if isinstance(model_config, str):
        model_config = legacy_model_config(model_config)

    radial_distribution = model_config.radial_distribution
    if radial_distribution == "damp":
        BBH_1g, tmig_s, valid_systems = populate_damping_type(
            N1,
            N2,
            BBH,
            Stars_disk,
            mask,
            star_timescales_myr,
            AGN,
            gf,
            h_list,
            sigma_list,
            r_list,
        )
    elif radial_distribution == "linear":
        BBH_1g, tmig_s, valid_systems = populate_random_type(
            N1,
            N2,
            BBH,
            Stars_disk,
            mask,
            star_timescales_myr,
            AGN,
            gf,
            "linear",
            h_list,
            sigma_list,
            r_list,
        )
    elif radial_distribution == "log":
        BBH_1g, tmig_s, valid_systems = populate_random_type(
            N1,
            N2,
            BBH,
            Stars_disk,
            mask,
            star_timescales_myr,
            AGN,
            gf,
            "log",
            h_list,
            sigma_list,
            r_list,
        )
    else:
        raise ValueError(f"Unknown radial_distribution: {radial_distribution}")

    return _finalize_population(BBH_1g, tmig_s, valid_systems, return_valid_mask)


def populate_damping_type(
    N1, N2, BBH, Stars_disk, mask, star_timescales, AGN, gf, h_list, sigma_list, r_list
):
    """Populate first-generation systems using the damping-informed placement."""
    total = N1 + N2
    BBH_1g = agn.BBH_properties(total)
    tmig = np.zeros(total, dtype=np.float64)
    valid_systems = np.zeros(total, dtype=bool)

    BBH_1g.chi1 = np.array(
        [auxi.spin(gf.spinrms) for _ in range(total)], dtype=np.float64
    )
    BBH_1g.chi2 = np.array(
        [auxi.spin(gf.spinrms) for _ in range(total)], dtype=np.float64
    )
    BBH_1g.idi = np.arange(total)

    if N1 > 0 and BBH is not None:
        if hasattr(BBH, "A") and len(BBH.A) >= N1:
            BBH_1g.A[:N1] = BBH.A[:N1]
            BBH_1g.Aini[:N1] = BBH.A[:N1]

        if hasattr(BBH, "m1") and len(BBH.m1) >= N1:
            BBH_1g.m1[:N1] = BBH.m1[:N1]

        BBH_1g.m2[:N1] = np.array(
            [auxi.secondary(m, BBH_1g.m2min) for m in BBH_1g.m1[:N1]],
            dtype=np.float64,
        )

        tmig[:N1], BBH_1g.Aini[:N1] = _migration_times_and_radii_after_elapsed(
            AGN,
            BBH_1g.m1[:N1],
            BBH_1g.A[:N1],
            np.zeros(N1, dtype=np.float64),
            h_list,
            sigma_list,
            r_list,
        )
        valid_systems[:N1] = True

    if N2 > 0 and Stars_disk is not None and hasattr(Stars_disk, "mBH"):
        start_idx = N1
        valid_indices = _selected_stellar_indices(Stars_disk, mask, N2)
        n_stellar = valid_indices.size

        if n_stellar > 0:
            system_slice = slice(start_idx, start_idx + n_stellar)
            star_tform_myr, star_tdamp_myr, _ = star_timescales
            tform_s = _stellar_timescale_seconds(
                star_tform_myr, Stars_disk, "tform", valid_indices
            )
            tdamp_s = _stellar_timescale_seconds(
                star_tdamp_myr, Stars_disk, "tdamp", valid_indices
            )

            if hasattr(Stars_disk, "a") and len(Stars_disk.a) > 0:
                BBH_1g.A[system_slice] = Stars_disk.a[valid_indices]
            BBH_1g.m1[system_slice] = Stars_disk.mBH[valid_indices]
            BBH_1g.m2[system_slice] = np.array(
                [auxi.secondary(m, BBH_1g.m2min) for m in BBH_1g.m1[system_slice]],
                dtype=np.float64,
            )

            if tform_s is not None and tdamp_s is not None:
                elapsed_to_bh = tform_s - tdamp_s
            else:
                elapsed_to_bh = np.zeros(n_stellar, dtype=np.float64)

            tmig_stellar, aini_stellar = _migration_times_and_radii_after_elapsed(
                AGN,
                BBH_1g.m1[system_slice],
                BBH_1g.A[system_slice],
                elapsed_to_bh,
                h_list,
                sigma_list,
                r_list,
            )

            valid_stellar = ~(
                np.isfinite(tmig_stellar) & (elapsed_to_bh > tmig_stellar)
            )
            if np.any(~valid_stellar):
                print(
                    "Warning: Omitting "
                    f"{np.sum(~valid_stellar)} systems that reach the trap before BH formation"
                )

            tmig[system_slice] = tmig_stellar
            valid_systems[system_slice] = valid_stellar
            BBH_1g.Aini[system_slice] = aini_stellar

    return BBH_1g, tmig, valid_systems


def populate_random_type(
    N1,
    N2,
    BBH,
    Stars_disk,
    mask,
    star_timescales,
    AGN,
    gf,
    dist_type,
    h_list,
    sigma_list,
    r_list,
):
    """Populate first-generation systems using linear/log radial sampling."""
    total = N1 + N2

    BBH_1g = agn.BBH_properties(total)
    tmig = np.zeros(total, dtype=np.float64)
    valid_systems = np.zeros(total, dtype=bool)

    if dist_type == "linear":
        BBH_1g.A = agn_f.semimajor_out_random_linear(total, AGN, 1, 6)
    else:
        BBH_1g.A = agn_f.semimajor_out_random_log(total, AGN, 1, 6)

    if N1 > 0 and BBH is not None and hasattr(BBH, "m1"):
        if len(BBH.m1) >= N1:
            BBH_1g.m1[:N1] = BBH.m1[:N1]
            BBH_1g.m2[:N1] = np.array(
                [auxi.secondary(m, BBH_1g.m2min) for m in BBH_1g.m1[:N1]],
                dtype=np.float64,
            )
            tmig[:N1], BBH_1g.Aini[:N1] = _migration_times_and_radii_after_elapsed(
                AGN,
                BBH_1g.m1[:N1],
                BBH_1g.A[:N1],
                np.zeros(N1, dtype=np.float64),
                h_list,
                sigma_list,
                r_list,
            )
            valid_systems[:N1] = True

    if N2 > 0 and Stars_disk is not None and hasattr(Stars_disk, "mBH"):
        start_idx = N1
        valid_indices = _selected_stellar_indices(Stars_disk, mask, N2)
        n_stellar = valid_indices.size

        if n_stellar > 0:
            system_slice = slice(start_idx, start_idx + n_stellar)
            star_tform_myr, star_tdamp_myr, _ = star_timescales
            tform_s = _stellar_timescale_seconds(
                star_tform_myr, Stars_disk, "tform", valid_indices
            )
            tdamp_s = _stellar_timescale_seconds(
                star_tdamp_myr, Stars_disk, "tdamp", valid_indices
            )

            BBH_1g.m1[system_slice] = Stars_disk.mBH[valid_indices]
            BBH_1g.m2[system_slice] = np.array(
                [auxi.secondary(m, BBH_1g.m2min) for m in BBH_1g.m1[system_slice]],
                dtype=np.float64,
            )

            if tform_s is not None and tdamp_s is not None:
                elapsed_to_bh = tform_s - tdamp_s
            else:
                elapsed_to_bh = np.zeros(n_stellar, dtype=np.float64)

            tmig_stellar, aini_stellar = _migration_times_and_radii_after_elapsed(
                AGN,
                BBH_1g.m1[system_slice],
                BBH_1g.A[system_slice],
                elapsed_to_bh,
                h_list,
                sigma_list,
                r_list,
            )

            valid_stellar = ~(
                np.isfinite(tmig_stellar) & (elapsed_to_bh > tmig_stellar)
            )
            if np.any(~valid_stellar):
                print(
                    "Warning: Omitting "
                    f"{np.sum(~valid_stellar)} random-radial systems that reach "
                    "the trap before BH formation"
                )

            tmig[system_slice] = tmig_stellar
            valid_systems[system_slice] = valid_stellar
            BBH_1g.Aini[system_slice] = aini_stellar

    BBH_1g.chi1 = np.array(
        [auxi.spin(gf.spinrms) for _ in range(total)], dtype=np.float64
    )
    BBH_1g.chi2 = np.array(
        [auxi.spin(gf.spinrms) for _ in range(total)], dtype=np.float64
    )
    BBH_1g.idi = np.arange(total)

    return BBH_1g, tmig, valid_systems
