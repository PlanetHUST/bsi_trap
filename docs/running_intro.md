# Running Introduction

This file describes how to run the current `bsi_trap` checkout, what the run
pipeline does today, and where the main caveats are.  It is intentionally tied
to the current code state rather than to the final scientific model.

## 1. Import And Environment Rule

This repository uses a `src/` layout.  In this checkout there is no packaging
metadata file such as `pyproject.toml`, so the safest local command pattern is:

```bash
env PYTHONPATH=src python -m cli.main --help
```

Do not import modules as `src.bsi_trap...`.  The package name is `bsi_trap`;
`src/` only tells Python where to find the package.

For plotting commands in restricted or headless environments, use a writable
Matplotlib cache:

```bash
env MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src python -m cli.main plot --help
```

For day-to-day local use, the `scripts/` wrappers add `src/` to `sys.path`
for you:

```bash
python scripts/run_cli.py --help
python scripts/simulate.py --help
python scripts/analyze.py --help
python scripts/plot.py --help
python scripts/trap_population.py --help
```

The wrappers are convenience entry points for a source checkout.  They only add
`src/` to Python's import path and call the matching `src/cli/*_command.py`
entry point.  They do not assemble workflow objects or inject CLI subcommands.
The package imports still use `bsi_trap...`, not `src.bsi_trap...`.

## 2. Main CLI Surface

The current top-level CLI is:

```bash
env PYTHONPATH=src python -m cli.main --help
```

It exposes three subcommands:

```text
simulate    run parallel sample simulations
analyze     summarize simulation_summary.json
plot        generate compact timeline diagnostics
```

The simulation command is defined in:

```text
src/cli/simulate_command.py
src/workflow/sample_run.py
src/workflow/parallel_manager.py
```

The analysis and plotting commands are defined in:

```text
src/cli/analyze_command.py
src/cli/plot_command.py
src/workflow/analysis_run.py
src/workflow/plot_run.py
```

The naming convention is:

```text
src/cli/*_command.py      terminal-facing argparse adapters
src/workflow/*_run.py     project workflows connected to package modules
scripts/*.py              source-checkout wrappers and diagnostics
```

## 3. Minimal Smoke Run

The simulation command needs a text file containing one SMBH mass per sample.
For a one-sample smoke run:

```bash
printf "1.0e6\n" > /tmp/bsi_trap_smbh_grid.txt
```

Then run:

```bash
python scripts/simulate.py \
  --num-samples 1 \
  --max-workers 1 \
  --base-seed 42 \
  --smbh-grid /tmp/bsi_trap_smbh_grid.txt \
  --output-dir output \
  --temp-dir temp \
  --model-name disk_trap_single_bbh \
  --gas-mode without_gas \
  --disk-model sirko \
  --agn-lifetime-myr 10.0 \
  --disk-resolution 256 \
  --pair-up-eta 2.5 \
  --trap-width-scale-heights 1.0
```

Use `--max-workers 1` while debugging.  Parallel execution is already wired,
but single-worker runs give clearer errors and deterministic progress logs.

To suppress progress prints:

```bash
AGN_PROGRESS=0 python scripts/simulate.py ...
```

## 4. Outputs

A successful simulation writes:

```text
output/<model-name>/sample_0000_gen_001.h5
output/<model-name>/sample_0000_gen_001.manifest.json
output/final_combined/<model-name>_combined.h5
output/simulation_summary.json
temp/model_0_sample_0.json
```

The per-generation HDF5 file is the main physical output.  It contains groups
for:

```text
agn_properties
bbh_properties
flags
timescales
trap_population
```

The combined HDF5 file groups successful samples under:

```text
samples/sample_0000/gen_001
```

The JSON files are run metadata.  They are useful for checking whether a sample
finished, which output file was produced, how many mergers were counted, and
what error was raised for failed samples.

Binary evolution is recorded explicitly.  In the HDF5 file, inspect:

```text
bbh_properties/sma_initial
bbh_properties/sma_final
bbh_properties/ecc_initial
bbh_properties/ecc_final
timescales/tdel_myr
timescales/tmerg_myr
flags/merger_flag
flags/stop_reason_flag
```

The sample manifest also records:

```text
evolved_binary_count
binary_evolution/evolved_binary_count
binary_evolution/merger_count
binary_evolution/mean_delay_myr
```

## 5. Analysis And Plotting

After a simulation run:

```bash
python scripts/analyze.py \
  --summary-path output/simulation_summary.json \
  --output-dir output/analysis
```

This writes:

```text
output/analysis/analysis_summary.json
```

To make the compact diagnostic figures:

```bash
MPLCONFIGDIR=/tmp/matplotlib python scripts/plot.py \
  --summary-path output/simulation_summary.json \
  --output-dir output/plots
```

Typical plot outputs include:

```text
bbh_sbh_population_timeline.png
bbh_parameter_timeline.png
plot_manifest.json
```

The population timeline tracks active trap single BHs and BBHs.  Single BHs
increase when objects reach the trap and decrease when pairs form; BBHs
increase at BBH formation and decrease when simulated mergers occur.

For a trap-population-only diagnostic without HDF5 output:

```bash
python scripts/trap_population.py \
  --smbh-mass 1.0e6 \
  --seed 42 \
  --gas-mode without_gas \
  --disk-model sirko \
  --disk-resolution 256
```

## 6. Validation Commands

Use these checks after workflow or physics-contract edits:

```bash
env PYTHONPATH=src python -m compileall -q src tests
```

```bash
env MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src python -W error -m unittest discover -s tests -v
```

The warning-gated test command is important because several earlier problems
were import-side effects or plotting-cache warnings rather than syntax errors.

## 7. What The Code Can Do Now

The current workflow can:

- Build an AGN disk environment from a `sirko` or `thompson` pAGN disk profile.
- Load the stellar-remnant input population from `input/input_data.txt`.
- Sample an embedded BH population and assign formation times and radii.
- Resolve mass-dependent migration targets and trap-arrival times.
- Build a first-generation trap population with finite local pair-up.
- Store the pairing timeline: later arrival, encounter time, pair-up delay, and
  BBH formation time.
- Initialize first-generation BBH orbital and spin parameters.
- Evolve BBHs with either gas-plus-GW or GW-only evolution.
- Recycle retained merger remnants into later hierarchical generations up to
  `ModelConfig.max_generations`.
- Schedule reduced BSI events from the available trap-single reservoir and
  apply the weak BSI update when the BBH evolution reaches the scheduled time.
- Write per-generation HDF5 files, sample manifests, worker metadata, combined
  HDF5 files, and a simulation summary.
- Run basic analysis and smoke-run plotting from `simulation_summary.json`.

The current scientific unit of work is therefore:

```text
one SMBH mass sample
  -> one AGN disk
  -> one initial trap population
  -> generation loop:
       pair retained remnant singles
       evolve BBHs
       compute final remnant mass/spin/kick and retention
  -> HDF5 + summary/diagnostic outputs
```

## 8. What The Code Cannot Do Yet

The current workflow does not yet implement the full AGN trap merger model.
Important missing pieces are:

- No time-dependent mixed single-BH/BBH trap reservoir event loop.
- No continuously updated BSI event loop coupled to a time-dependent trap
  reservoir; BSI scheduling is reduced to the current generation-level single
  reservoir.
- No physical BSI exchange outcome that replaces a BBH component with an
  incoming single.
- No prompt-merger BSI channel.
- Hierarchical generations recycle retained merger remnants only; they do not
  yet mix with a continuously evolving background single-BH reservoir.
- No self-consistent update of compact-object scale height after vertical
  heating.
- The CLI exposes the main current `ModelConfig` fields, including gas mode,
  BSI enablement, pair-up scale, trap geometry, and maximum generation count.
  Future model fields still need to be added explicitly when they become part
  of the production workflow.
- No production-grade statistical analysis.  Current plots are smoke
  diagnostics.

The main design boundary to keep in mind is:

```text
physics modules mutate populations and binary parameters
timeline classes record when events happened
storage writes both the final state and event ledgers
```

## 9. Current Caveats

The practical caveats are:

- `PYTHONPATH=src` is required unless packaging metadata is added later.
- `input/` is ignored by Git, but the run depends on `input/input_data.txt`.
- `docs/` is ignored by Git in this checkout; this file is at repository root
  so it is visible to normal Git commands.
- The default CLI creates one `ModelConfig`; multi-model scans are available at
  the workflow layer but are not exposed as a rich CLI yet.
- Simulation failures are captured per sample in `temp/model_*_sample_*.json`
  and summarized in `output/simulation_summary.json`; a failed sample does not
  necessarily mean the whole command crashes.
- `scripts/trap_population.py` runs the initialization workflow only; it does
  not evolve BBHs or write HDF5 output.
- The input population is large.  Use small `--num-samples`, one worker, and a
  low disk resolution while debugging.

## 10. Useful Entry Points For Development

For code changes, these are the most useful files to start from:

```text
src/workflow/sample_run.py
src/cli/simulate_command.py
src/bsi_trap/initialization/environment.py
src/bsi_trap/initialization/trap_population.py
src/bsi_trap/evolve/pairing.py
src/bsi_trap/evolve/gen_evolve.py
src/bsi_trap/evolve/bsi.py
src/bsi_trap/storage/payload.py
src/workflow/analysis_run.py
src/workflow/plot_run.py
```

For behavior checks, start with:

```text
tests/test_trap_population_initialization.py
tests/test_evolution_timeline.py
tests/test_workflow.py
```
