import json
import logging
from pathlib import Path

import bsi_trap.plots.style
from bsi_trap.plots.disk_profile_interace import (
    plot_disk_parameter_profiles,
    plot_disk_torque_profiles,
)

logger = logging.getLogger(__name__)


def run_plotting_workflow(
    *,
    summary_path: Path,
    output_dir: Path,
) -> Path:
    pass
