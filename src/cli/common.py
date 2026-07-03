from __future__ import annotations

import argparse
import logging

from bsi_trap.parallel_compute import configure_logging

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def add_log_level_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-level", default="INFO", choices=LOG_LEVELS)


def configure_logging_from_args(args: argparse.Namespace) -> None:
    configure_logging(getattr(logging, args.log_level))
