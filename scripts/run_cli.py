#!/usr/bin/env python
"""Run the full bsi-trap CLI from a source checkout."""

from __future__ import annotations

from _bootstrap import add_src_to_path


def main() -> None:
    add_src_to_path()

    from cli.main import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
