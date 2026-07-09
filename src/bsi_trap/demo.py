"""Plain Python demo for the compact BSI-trap workflow."""

from __future__ import annotations

from pathlib import Path

from bsi_trap.dataoi import DEFAULT_CATALOG, format_summary
from bsi_trap.simulation import DemoConfig, DemoResult, run_simulation

DEMO_CONFIG = DemoConfig()
SUMMARY_PREVIEW = 5


def run_demo(
    config: DemoConfig = DEMO_CONFIG,
    *,
    catalog_path: Path = DEFAULT_CATALOG,
) -> DemoResult:
    return run_simulation(config, catalog_path=catalog_path)


def demo_summary(
    config: DemoConfig = DEMO_CONFIG,
    *,
    catalog_path: Path = DEFAULT_CATALOG,
    preview: int = SUMMARY_PREVIEW,
) -> str:
    result = run_demo(config, catalog_path=catalog_path)
    return format_summary(result, preview=preview)


def main() -> None:
    print(demo_summary())


if __name__ == "__main__":
    main()
