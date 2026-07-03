#!/usr/bin/env python
"""Run the plot command from a source checkout."""

from __future__ import annotations

from _bootstrap import add_src_to_path


def main() -> None:
    add_src_to_path()

    from cli.plot_command import main as plot_main

    plot_main()


if __name__ == "__main__":
    main()
