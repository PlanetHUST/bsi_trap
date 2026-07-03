from workflow.parallel_manager import run_simulation_workflow
from workflow.analysis_run import run_analysis_workflow


def run_plotting_workflow(*args, **kwargs):
    """Load plotting dependencies only when the plotting workflow is requested."""
    from workflow.plot_run import run_plotting_workflow as _run_plotting_workflow

    return _run_plotting_workflow(*args, **kwargs)

__all__ = [
    "run_simulation_workflow",
    "run_analysis_workflow",
    "run_plotting_workflow",
]
