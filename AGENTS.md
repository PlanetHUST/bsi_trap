# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.11 scientific-simulation repository using a `src/` layout. Core code lives in `src/bsi_trap/`: `physics/` contains disk, population, pairing, and binary-evolution calculations; `simulation/` defines dataclasses and orchestration; `dataoi/` loads catalogs and formats results; and `parallel_compute/` provides desktop multiprocessing. Run-facing helpers live in `scripts/`, while physical-model notes and the current workflow description live in `docs/`. Large local catalogs belong in `input/`; generated artifacts belong in `output/`, `report/`, or `temp/`. These paths are intentionally ignored by Git.

## Build, Test, and Development Commands

The repository has no packaging manifest or build step. Work from the repository root in an environment containing NumPy and pAGN.

- `python scripts/demo.py` runs the source-checkout demo using `input/input_data.txt`.
- `python -m compileall -q src scripts` checks all committed Python files for syntax errors.
- `pyright` runs the configured basic type check over `src/` and `scripts/`.
- `PYTHONPATH=src python -c "import bsi_trap"` smoke-tests package imports directly.

Treat failures as real findings; do not weaken `pyrightconfig.json` to hide them.

## Coding Style & Naming Conventions

Use four-space indentation, `from __future__ import annotations`, type hints on public interfaces, and short module docstrings. Follow standard Python naming: `snake_case` for functions and variables, `PascalCase` for dataclasses, and `UPPER_CASE` for constants. Scientific quantities should carry units in their names (`arrival_s`, `mass_msun`, `radius_cm`) and use shared constants from `bsi_trap.units`. Keep physics stages explicit and preserve seeded NumPy RNG behavior.

## Testing Guidelines

No test suite or coverage threshold is currently committed. Add focused pytest tests under `tests/`, named `test_<module>.py`, with functions named `test_<behavior>()`; run them with `python -m pytest`. Cover configuration validation, array shapes and masks, deterministic seeded results, physical bounds, and unit conversions. Use small synthetic catalogs or fixtures instead of the large ignored input file.

## Commit & Pull Request Guidelines

Recent history uses terse, inconsistent subjects, so new commits should improve on it: use an imperative, specific line such as `Fix trap-arrival index selection`. Keep commits scoped. Pull requests should describe the changed physical assumption or workflow stage, list validation commands and results, call out unit/schema changes, and link relevant issues. Include before/after figures when plots or scientific outputs change; never commit generated data or local environment files.
