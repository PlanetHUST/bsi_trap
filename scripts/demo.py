#!/usr/bin/env python
"""Run the compact BSI-trap demo from a source checkout."""

from __future__ import annotations

from _bootstrap import add_src_to_path


def main() -> None:
    add_src_to_path()

    from bsi_trap.demo import main as demo_main

    demo_main()


if __name__ == "__main__":
    main()
