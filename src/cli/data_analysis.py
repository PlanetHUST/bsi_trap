import argparse
import logging
from pathlib import Path

from workflow.analysis import run_analysis_workflow

logger = logging.getLogger(__name__)


def add_analysis_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "analyze",
        help="Analyze simulation results.",
    )

    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("output/simulation_summary.json"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/analysis"),
    )

    parser.set_defaults(func=run_analysis_command)


def run_analysis_command(args: argparse.Namespace) -> None:
    logger.info("Running analysis command")

    analysis = run_analysis_workflow(
        summary_path=args.summary_path,
        output_dir=args.output_dir,
    )

    logger.info("Analysis result: %s", analysis)
