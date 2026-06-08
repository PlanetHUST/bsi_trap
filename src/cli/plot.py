from __future__ import annotations

import argparse
from pathlib import Path

from workflow.plot import run_plotting_workflow


def add_plot_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "plot",
        help="Plot simulation results.",
    )

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

    parser.set_defaults(func=run_plot_command)


def run_plot_command(args: argparse.Namespace) -> None:
    run_plotting_workflow(
        summary_path=args.summary_path,
        output_dir=args.output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BSI-trap figures from combined HDF5 outputs."
    )
    parser.add_argument("--summary-path", default="output/simulation_summary.json")
    parser.add_argument("--output-dir", default="output/plots")

    args = parser.parse_args()

    run_plotting_workflow(
        summary_path=Path(args.summary_path),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
