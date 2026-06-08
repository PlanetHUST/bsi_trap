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
    disk_resolution: int = 256
    migrator_mass_msun: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def name(self) -> str:
        """Compatibility alias used by the shrink-plan examples."""
        return self.model_name

    @property
    def simulation_type(self) -> str:
        """Legacy output label used throughout the current pipeline."""
        return self.model_name
