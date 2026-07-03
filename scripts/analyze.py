#!/usr/bin/env python
"""Run the analyze command from a source checkout."""

from __future__ import annotations

from _bootstrap import add_src_to_path


def main() -> None:
    add_src_to_path()

    from cli.analyze_command import main as analyze_main

    analyze_main()


if __name__ == "__main__":
    main()
