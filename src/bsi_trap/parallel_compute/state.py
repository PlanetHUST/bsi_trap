from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SimulationConfig:
    num_samples: int
    max_workers: int
    output_dir: Path
    temp_dir: Path
    base_seed: int = 42
    timeout: float | None = None
    models: list[Any] = field(default_factory=list)

    def validate(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive.")
        if not self.models:
            raise ValueError("At least one model config is required.")


@dataclass(frozen=True)
class SimulationTask:
    model_config: Any
    model_index: int
    sample_index: int
    smbh_mass: float


@dataclass(frozen=True)
class SimulationResult:
    success: bool
    model_index: int
    sample_index: int
    smbh_mass: float
    output_file: str | None = None
    generation_files: tuple[str, ...] | None = None
    manifest_file: str | None = None
    metadata_file: str | None = None
    execution_time: float | None = None
    merger_count: int | None = None
    evolved_binary_count: int | None = None
    generation_count: int | None = None
    error: str | None = None
