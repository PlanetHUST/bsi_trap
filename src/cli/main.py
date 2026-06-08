import argparse
import logging

from cli.simulation import add_simulation_parser
from cli.data_analysis import add_analysis_parser
from cli.plot import add_plot_parser
from bsi_trap.parallel_compute import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bsi-trap",
        description="BSI Trap simulation command-line interface.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_simulation_parser(subparsers)
    add_analysis_parser(subparsers)
    add_plot_parser(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(getattr(logging, args.log_level))

    args.func(args)


if __name__ == "__main__":
    main()
