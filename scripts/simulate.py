#!/usr/bin/env python
"""Run the simulate command from a source checkout."""

from __future__ import annotations

from _bootstrap import add_src_to_path


def main() -> None:
    add_src_to_path()

    from cli.simulate_command import main as simulate_main

    simulate_main()


if __name__ == "__main__":
    main()
