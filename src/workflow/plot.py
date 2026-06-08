import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def run_plotting_workflow(
    *,
    summary_path: Path,
    output_dir: Path,
) -> Path:
    """
    High-level plotting workflow.

    This creates a simple success/failure bar plot from simulation_summary.json.
    """

    if not summary_path.exists():
        raise FileNotFoundError(f"Simulation summary not found: {summary_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    success = int(summary.get("num_success", 0))
    failed = int(summary.get("num_failed", 0))

    fig, ax = plt.subplots()
    ax.bar(["success", "failed"], [success, failed])
    ax.set_ylabel("Number of simulations")
    ax.set_title("Simulation success/failure count")

    output_path = output_dir / "simulation_success_failure.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Plot saved to %s", output_path)

    return output_path
