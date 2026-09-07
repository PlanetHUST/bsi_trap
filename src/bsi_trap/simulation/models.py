"""Typed configuration and state records for the BSI-trap simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


def _readonly_array(values: npt.ArrayLike, dtype: npt.DTypeLike) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class DemoConfig:
    """Scientific and runtime choices for one realization."""

    smbh_mass_msun: float = 1.0e6
    agn_lifetime_myr: float = 10.0
    population_size: int = 500
    seed: int = 42
    disk_model: str = "sirko"
    disk_resolution: int = 256
    alpha: float = 0.01
    disk_eddington_ratio: float = 0.5
    disk_b: int = 0
    disk_accretion_rate_g_s: float | None = None
    thompson_outer_radius_rg: float = 1.0e7
    thompson_outer_accretion_rate_msun_per_yr: float | None = None
    catalog_rows: int | None = 50_000
    migration_arrival_tolerance_cells: float = 0.25
    torque_smoothing: float = 0.005
    gap_opening_condition: str = "kii"
    consider_type_ii_migration: bool = True
    trap_min_separation_cells: float = 10.0
    pair_delay_scale: float = 1.0
    pairing_region_scale_heights: float = 1.0
    initial_bbh_sma_au: float = 0.01
    initial_bbh_eccentricity: float = 0.3
    gw_relative_tolerance: float = 1.0e-8
    gw_max_step_fraction: float = 0.01
    enable_bsi: bool = True
    bsi_cross_section_sma: float = 3.0
    bsi_hardening_fraction: float = 0.15
    bsi_eccentricity_pump: float = 0.2


@dataclass(frozen=True)
class DiskProfile:
    """AGN disk arrays in CGS, independent of migration physics."""

    model_name: str
    radius_cm: FloatArray
    temperature_k: FloatArray
    surface_density_g_cm2: FloatArray
    density_g_cm3: FloatArray
    sound_speed_cm_s: FloatArray
    aspect_ratio: FloatArray
    optical_depth: FloatArray
    opacity_cm2_g: FloatArray
    toomre_q: FloatArray
    omega_s_inv: FloatArray
    effective_temperature_fourth_k4: FloatArray
    inner_radius_cm: float
    outer_radius_cm: float
    gravitational_radius_cm: float
    schwarzschild_radius_cm: float

    def __post_init__(self) -> None:
        arrays = (
            self.radius_cm,
            self.temperature_k,
            self.surface_density_g_cm2,
            self.density_g_cm3,
            self.sound_speed_cm_s,
            self.aspect_ratio,
            self.optical_depth,
            self.opacity_cm2_g,
            self.toomre_q,
            self.omega_s_inv,
            self.effective_temperature_fourth_k4,
        )
        if any(values.ndim != 1 for values in arrays):
            raise ValueError("disk profile arrays must be one-dimensional")
        if len({values.shape for values in arrays}) != 1:
            raise ValueError("disk profile arrays must have the same shape")
        if self.radius_cm.size < 2:
            raise ValueError("disk profile needs at least two radial cells")
        if not np.all(np.isfinite(self.radius_cm)) or not np.all(
            np.diff(self.radius_cm) > 0.0
        ):
            raise ValueError("disk radius [cm] must be finite and strictly increasing")
        positive_arrays = (
            self.radius_cm,
            self.temperature_k,
            self.surface_density_g_cm2,
            self.density_g_cm3,
            self.sound_speed_cm_s,
            self.aspect_ratio,
            self.opacity_cm2_g,
            self.toomre_q,
            self.omega_s_inv,
            self.effective_temperature_fourth_k4,
        )
        if not all(np.all(np.isfinite(values) & (values > 0.0)) for values in positive_arrays):
            raise ValueError("physical disk arrays must be finite and positive")
        if not np.all(np.isfinite(self.optical_depth) & (self.optical_depth >= 0.0)):
            raise ValueError("disk optical depth must be finite and nonnegative")
        if not (
            0.0 < self.inner_radius_cm <= self.outer_radius_cm
            and self.gravitational_radius_cm > 0.0
            and np.isclose(
                self.schwarzschild_radius_cm,
                2.0 * self.gravitational_radius_cm,
                rtol=1.0e-12,
            )
        ):
            raise ValueError("disk scalar radii are inconsistent")

    @property
    def scale_height_cm(self) -> FloatArray:
        return self.aspect_ratio * self.radius_cm

    def closest_index(self, radius_cm: float) -> int:
        return int(np.argmin(np.abs(self.radius_cm - radius_cm)))

    # Compatibility names used by the original scaffold.
    @property
    def sigma_g_cm2(self) -> FloatArray:
        return self.surface_density_g_cm2

    @property
    def cs_cm_s(self) -> FloatArray:
        return self.sound_speed_cm_s

    @property
    def h_over_r(self) -> FloatArray:
        return self.aspect_ratio

    @property
    def r_in_cm(self) -> float:
        return self.inner_radius_cm

    @property
    def r_out_cm(self) -> float:
        return self.outer_radius_cm


@dataclass(frozen=True)
class EmbeddedPopulation:
    """Immutable initial BH population with permanent physical IDs."""

    bh_id: IntArray
    mass_msun: FloatArray
    initial_radius_cm: FloatArray

    def __post_init__(self) -> None:
        bh_id = _readonly_array(self.bh_id, np.int64)
        mass_msun = _readonly_array(self.mass_msun, np.float64)
        initial_radius_cm = _readonly_array(self.initial_radius_cm, np.float64)
        if bh_id.ndim != 1 or mass_msun.ndim != 1 or initial_radius_cm.ndim != 1:
            raise ValueError("population arrays must be one-dimensional")
        if not (bh_id.shape == mass_msun.shape == initial_radius_cm.shape):
            raise ValueError("population arrays must have the same shape")
        if not np.array_equal(bh_id, np.arange(bh_id.size, dtype=np.int64)):
            raise ValueError("bh_id must be the permanent array index 0..N-1")
        if not np.all(np.isfinite(mass_msun) & (mass_msun > 0.0)):
            raise ValueError("BH masses [Msun] must be finite and positive")
        if not np.all(np.isfinite(initial_radius_cm) & (initial_radius_cm > 0.0)):
            raise ValueError("initial radii [cm] must be finite and positive")
        object.__setattr__(self, "bh_id", bh_id)
        object.__setattr__(self, "mass_msun", mass_msun)
        object.__setattr__(self, "initial_radius_cm", initial_radius_cm)

    @property
    def size(self) -> int:
        return int(self.bh_id.size)


@dataclass(frozen=True)
class ArrivalTimeline:
    """Per-BH arrival information; rows retain permanent BH identity."""

    bh_id: IntArray
    arrival_time_s: FloatArray
    migration_elapsed_time_s: FloatArray
    mass_msun: FloatArray
    reached_mask: BoolArray
    order: IntArray
    final_radius_cm: FloatArray
    trap_radius_cm: FloatArray
    stop_reason: npt.NDArray[np.str_]

    def __post_init__(self) -> None:
        bh_id = _readonly_array(self.bh_id, np.int64)
        arrival_time_s = _readonly_array(self.arrival_time_s, np.float64)
        migration_elapsed_time_s = _readonly_array(
            self.migration_elapsed_time_s,
            np.float64,
        )
        mass_msun = _readonly_array(self.mass_msun, np.float64)
        reached_mask = _readonly_array(self.reached_mask, np.bool_)
        order = _readonly_array(self.order, np.int64)
        final_radius_cm = _readonly_array(self.final_radius_cm, np.float64)
        trap_radius_cm = _readonly_array(self.trap_radius_cm, np.float64)
        stop_reason = _readonly_array(self.stop_reason, np.str_)
        row_arrays = (
            bh_id,
            arrival_time_s,
            migration_elapsed_time_s,
            mass_msun,
            reached_mask,
            final_radius_cm,
            trap_radius_cm,
            stop_reason,
        )
        if any(values.ndim != 1 for values in (*row_arrays, order)):
            raise ValueError("arrival arrays must be one-dimensional")
        if len({values.shape for values in row_arrays}) != 1:
            raise ValueError("per-BH arrival arrays must have the same shape")
        if not np.array_equal(bh_id, np.arange(bh_id.size, dtype=np.int64)):
            raise ValueError("arrival rows must retain permanent BH IDs")
        if np.any(order < 0) or np.any(order >= bh_id.size):
            raise ValueError("arrival order contains an invalid BH ID")
        if not np.all(reached_mask[order]):
            raise ValueError("arrival order may contain only reached BHs")
        if not np.all(np.isfinite(arrival_time_s[reached_mask])):
            raise ValueError("reached BHs need finite arrival times [s]")
        if not np.all(
            np.isfinite(migration_elapsed_time_s)
            & (migration_elapsed_time_s >= 0.0)
        ):
            raise ValueError("migration elapsed times [s] must be finite and nonnegative")
        if np.any(np.isfinite(arrival_time_s[~reached_mask])):
            raise ValueError("unreached BHs must have infinite arrival times")
        for name, values in (
            ("bh_id", bh_id),
            ("arrival_time_s", arrival_time_s),
            ("migration_elapsed_time_s", migration_elapsed_time_s),
            ("mass_msun", mass_msun),
            ("reached_mask", reached_mask),
            ("order", order),
            ("final_radius_cm", final_radius_cm),
            ("trap_radius_cm", trap_radius_cm),
            ("stop_reason", stop_reason),
        ):
            object.__setattr__(self, name, values)

    @property
    def arrival_s(self) -> FloatArray:
        return self.arrival_time_s

    @property
    def reached_count(self) -> int:
        return int(np.count_nonzero(self.reached_mask))


@dataclass
class BinaryPopulation:
    """Complete state of one binary; component fields are physical BH IDs."""

    binary_id: int
    bh1_id: int
    bh2_id: int
    m1_msun: float
    m2_msun: float
    formation_time_s: float
    formation_radius_cm: float
    sma_cm: float
    eccentricity: float
    status: str = "active"
    last_evolution_time_s: float | None = None
    evolution_revision: int = 0
    bsi_encounter_count: int = 0

    def __post_init__(self) -> None:
        if self.binary_id < 0 or self.bh1_id < 0 or self.bh2_id < 0:
            raise ValueError("binary and component IDs must be nonnegative")
        if self.bh1_id == self.bh2_id:
            raise ValueError("a binary cannot contain the same BH twice")
        if self.m1_msun <= 0.0 or self.m2_msun <= 0.0:
            raise ValueError("binary component masses [Msun] must be positive")
        if not np.isfinite(self.formation_time_s) or self.formation_time_s < 0.0:
            raise ValueError("binary formation time [s] must be finite and nonnegative")
        if not np.isfinite(self.formation_radius_cm) or self.formation_radius_cm <= 0.0:
            raise ValueError("binary formation radius [cm] must be finite and positive")
        if not np.isfinite(self.sma_cm) or self.sma_cm <= 0.0:
            raise ValueError("binary semi-major axis [cm] must be finite and positive")
        if not np.isfinite(self.eccentricity) or not 0.0 <= self.eccentricity < 1.0:
            raise ValueError("binary eccentricity must be in [0, 1)")
        if self.last_evolution_time_s is None:
            self.last_evolution_time_s = self.formation_time_s
        if (
            not np.isfinite(self.last_evolution_time_s)
            or self.last_evolution_time_s < self.formation_time_s
        ):
            raise ValueError("binary evolution time cannot precede formation")

    @property
    def first_idx(self) -> int:
        return self.bh1_id

    @property
    def second_idx(self) -> int:
        return self.bh2_id

    @property
    def ecc(self) -> float:
        return self.eccentricity


@dataclass(frozen=True)
class StateEventRecord:
    time_s: float
    event_type: str
    bh_ids: tuple[int, ...] = ()
    binary_id: int | None = None


@dataclass(frozen=True)
class PairingRecord:
    binary_id: int
    bh1_id: int
    bh2_id: int
    later_arrival_time_s: float
    pair_delay_s: float
    formation_time_s: float
    formation_radius_cm: float


@dataclass(frozen=True)
class MergerRecord:
    binary_id: int
    bh1_id: int
    bh2_id: int
    merger_time_s: float
    integration_stop_radius_cm: float
    final_mass_msun: float
    final_spin: float
    remnant_model_name: str


@dataclass(frozen=True)
class BSIRecord:
    binary_id: int
    single_bh_id: int
    event_time_s: float
    encounter_rate_s_inv: float
    sma_before_cm: float
    sma_after_cm: float
    eccentricity_before: float
    eccentricity_after: float
    prescription_name: str


@dataclass
class SimulationState:
    """Authoritative clock and non-compacting physical-object state."""

    current_time_s: float
    active_single_mask: BoolArray
    pairing_pending_mask: BoolArray
    paired_mask: BoolArray
    merged_mask: BoolArray
    rng: np.random.Generator = field(repr=False)
    binaries: list[BinaryPopulation] = field(default_factory=list)
    event_history: list[StateEventRecord] = field(default_factory=list)
    pairing_records: list[PairingRecord] = field(default_factory=list)
    merger_records: list[MergerRecord] = field(default_factory=list)
    bsi_records: list[BSIRecord] = field(default_factory=list)

    @classmethod
    def from_population(
        cls,
        population: EmbeddedPopulation,
        rng: np.random.Generator,
    ) -> SimulationState:
        size = population.size
        return cls(
            current_time_s=0.0,
            active_single_mask=np.zeros(size, dtype=bool),
            pairing_pending_mask=np.zeros(size, dtype=bool),
            paired_mask=np.zeros(size, dtype=bool),
            merged_mask=np.zeros(size, dtype=bool),
            rng=rng,
        )

    def advance_time(self, next_time_s: float) -> None:
        if not np.isfinite(next_time_s) or next_time_s < self.current_time_s:
            raise ValueError("simulation event time cannot decrease or be nonfinite")
        self.current_time_s = float(next_time_s)

    def mark_arrived(self, bh_id: int) -> None:
        self._require_ids((bh_id,))
        if self.paired_mask[bh_id] or self.merged_mask[bh_id]:
            raise ValueError("paired or merged BH cannot become an active single")
        self.active_single_mask[bh_id] = True

    def mark_paired(self, bh1_id: int, bh2_id: int) -> None:
        self._require_ids((bh1_id, bh2_id))
        if bh1_id == bh2_id:
            raise ValueError("a BH cannot pair with itself")
        ids = np.asarray((bh1_id, bh2_id), dtype=int)
        if not np.all(self.active_single_mask[ids]):
            raise ValueError("both BHs must be active singles before pairing")
        if np.any(self.paired_mask[ids] | self.merged_mask[ids]):
            raise ValueError("paired or merged BH cannot be paired again")
        self.active_single_mask[ids] = False
        self.paired_mask[ids] = True
        self.validate_masks()

    def reserve_pair(self, bh1_id: int, bh2_id: int) -> None:
        self._require_ids((bh1_id, bh2_id))
        if bh1_id == bh2_id:
            raise ValueError("a BH cannot pair with itself")
        ids = np.asarray((bh1_id, bh2_id), dtype=int)
        if not np.all(self.active_single_mask[ids]):
            raise ValueError("both BHs must be active singles before reservation")
        self.active_single_mask[ids] = False
        self.pairing_pending_mask[ids] = True
        self.validate_masks()

    def form_reserved_pair(self, bh1_id: int, bh2_id: int) -> None:
        self._require_ids((bh1_id, bh2_id))
        ids = np.asarray((bh1_id, bh2_id), dtype=int)
        if not np.all(self.pairing_pending_mask[ids]):
            raise ValueError("both BHs must be reserved for this formation event")
        self.pairing_pending_mask[ids] = False
        self.paired_mask[ids] = True
        self.validate_masks()

    def mark_merged(self, bh1_id: int, bh2_id: int) -> None:
        self._require_ids((bh1_id, bh2_id))
        ids = np.asarray((bh1_id, bh2_id), dtype=int)
        if not np.all(self.paired_mask[ids]):
            raise ValueError("both BHs must be paired before merger")
        self.paired_mask[ids] = False
        self.merged_mask[ids] = True
        self.validate_masks()

    def release_pending_pairs(self) -> None:
        self.active_single_mask[self.pairing_pending_mask] = True
        self.pairing_pending_mask[:] = False
        self.validate_masks()

    def validate_masks(self) -> None:
        shape = self.active_single_mask.shape
        if (
            self.pairing_pending_mask.shape != shape
            or self.paired_mask.shape != shape
            or self.merged_mask.shape != shape
        ):
            raise ValueError("state masks must have the same shape")
        masks = (
            self.active_single_mask,
            self.pairing_pending_mask,
            self.paired_mask,
            self.merged_mask,
        )
        occupancy_count = sum(mask.astype(np.int8) for mask in masks)
        overlap = occupancy_count > 1
        if np.any(overlap):
            raise ValueError("BH state masks must be mutually exclusive")

    def _require_ids(self, bh_ids: tuple[int, ...]) -> None:
        population_size = self.active_single_mask.size
        if any(bh_id < 0 or bh_id >= population_size for bh_id in bh_ids):
            raise IndexError("BH ID is outside the immutable population")


BinaryState = BinaryPopulation


@dataclass(frozen=True)
class DemoResult:
    """Current auditable simulation result surface."""

    config: DemoConfig
    disk: DiskProfile
    population: EmbeddedPopulation
    arrivals: ArrivalTimeline
    state: SimulationState

    @property
    def single_count(self) -> int:
        return int(np.count_nonzero(self.state.active_single_mask))

    @property
    def seed(self) -> int:
        return self.config.seed


SimulationResult = DemoResult


__all__ = [
    "ArrivalTimeline",
    "BSIRecord",
    "BinaryPopulation",
    "BinaryState",
    "DemoConfig",
    "DemoResult",
    "DiskProfile",
    "EmbeddedPopulation",
    "MergerRecord",
    "PairingRecord",
    "SimulationState",
    "SimulationResult",
    "StateEventRecord",
]
