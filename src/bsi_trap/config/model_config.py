from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal


@dataclass(frozen=True)
class ModelConfig:
    """All model-channel choices for one named simulation scenario."""

    model_name: str

    gas_mode: Literal["with_gas", "without_gas"] = "with_gas"
    disk_model: Literal["sirko", "thompson"] = "sirko"
    bsi_enabled: bool = False

    agn_lifetime_myr: float = 10.0
    alpha: float = 0.01
    disk_resolution: int = 256

    bsi_eta: float = 6.0
    pair_up_eta: float = 10.0

    # Initial trap geometry in units of the local gas scale height. These are
    # explicit model parameters so the pairing density is never hard-coded.
    trap_width_scale_heights: float = 1.0
    compact_object_scale_height_fraction: float = 1.0

    master_seed: int | None = 20260611

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def simulation_type(self) -> str:
        return self.model_name
