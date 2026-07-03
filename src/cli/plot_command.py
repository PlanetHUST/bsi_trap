from __future__ import annotations

import argparse
from pathlib import Path

from cli.common import add_log_level_argument, configure_logging_from_args
from workflow import run_plotting_workflow


def add_plot_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("output/simulation_summary.json"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/plots"),
    )


def add_plot_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "plot",
        help="Plot simulation results.",
    )
    add_plot_arguments(parser)
    parser.set_defaults(func=run_plot_command)


def run_plot_command(args: argparse.Namespace) -> None:
    run_plotting_workflow(
        summary_path=args.summary_path,
        output_dir=args.output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate BSI-trap figures from simulation outputs."
    )
    add_log_level_argument(parser)
    add_plot_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging_from_args(args)
    run_plot_command(args)


if __name__ == "__main__":
    main()
