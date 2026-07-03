import argparse
import logging
from pathlib import Path

from cli.common import add_log_level_argument, configure_logging_from_args
from workflow.analysis_run import run_analysis_workflow

logger = logging.getLogger(__name__)


def add_analyze_arguments(parser: argparse.ArgumentParser) -> None:
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


def add_analyze_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "analyze",
        help="Analyze simulation results.",
    )
    add_analyze_arguments(parser)
    parser.set_defaults(func=run_analyze_command)


def run_analyze_command(args: argparse.Namespace) -> None:
    logger.info("Running analysis command")

    analysis = run_analysis_workflow(
        summary_path=args.summary_path,
        output_dir=args.output_dir,
    )

    logger.info("Analysis result: %s", analysis)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze BSI-trap simulation results.")
    add_log_level_argument(parser)
    add_analyze_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging_from_args(args)
    run_analyze_command(args)


if __name__ == "__main__":
    main()
