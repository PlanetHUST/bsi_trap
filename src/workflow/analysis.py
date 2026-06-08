import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_simulation_summary(summary_path: Path) -> dict[str, Any]:
    """
    Load simulation_summary.json produced by the parallel simulation manager.
    """

    if not summary_path.exists():
        raise FileNotFoundError(f"Simulation summary not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_analysis_workflow(
    *,
    summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """
    High-level analysis workflow.

    Keep statistical analysis here, not inside CLI or parallel execution.
    """

    logger.info("Starting analysis workflow")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_simulation_summary(summary_path)
    results = summary.get("results", [])

    if not isinstance(results, list):
        raise ValueError("Invalid simulation summary: 'results' must be a list.")

    total = len(results)
    success = sum(1 for r in results if r.get("success") is True)
    failed = total - success

    analysis = {
        "total": total,
        "success": success,
        "failed": failed,
        "success_fraction": success / total if total > 0 else 0.0,
    }

    analysis_path = output_dir / "analysis_summary.json"

    with analysis_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    logger.info("Analysis workflow finished: %s", analysis_path)

    return analysis
