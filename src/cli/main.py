import argparse

from cli.analyze_command import add_analyze_parser
from cli.common import add_log_level_argument, configure_logging_from_args
from cli.plot_command import add_plot_parser
from cli.simulate_command import add_simulate_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bsi-trap",
        description="BSI Trap simulation command-line interface.",
    )

    add_log_level_argument(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_simulate_parser(subparsers)
    add_analyze_parser(subparsers)
    add_plot_parser(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging_from_args(args)

    args.func(args)


if __name__ == "__main__":
    main()
