# Current Workflow Status

This note summarizes the project logic as it stands now.  It is meant to be a
status document: what the current workflow actually does, why this structure is
useful, what tasks remain, and which caveats matter when interpreting outputs.

## Project Logic Now

The current project is a reduced AGN migration-trap BBH workflow.  One sample
is organized as:

```text
SMBH mass + ModelConfig + seed
    -> AGN disk and torque profile
    -> embedded BH reservoir
    -> mass-dependent migration toward trap targets
    -> migration-only trap-arrival population
    -> chronological pairing/remnant event scheduler
    -> one committed BBH population per accepted pairing event
    -> BBH orbital/spin initialization
    -> BBH gas/GW evolution with optional weak BSI perturbations
    -> merger/remnant/kick/retention processing
    -> HDF5 event records and compact timeline plots
```

The current code now has a top-level event loop for trap pairing and
hierarchical remnant recycling.  It is still not a fully coupled many-body or
active-BBH scheduler: once a BBH is committed, `make_generation()` evolves that
committed population, and BSI is scheduled from the live single reservoir
available at that boundary.  The staged contract is:

- initialization builds migration and trap-arrival state;
- the hierarchical scheduler repeatedly asks pairing for the next accepted pair;
- retained merger remnants are returned to the live single reservoir;
- BBH evolution evolves each committed binary from its formation time;
- BSI is a scheduled weak perturbation applied inside the binary evolution;
- timelines record event times and outcomes;
- storage writes the final state plus event ledgers.

The main runtime boundary is:

```text
src/workflow/sample_run.py::run_physics_sample()
```

The main physical data boundary is:

```text
TrapPopulation
    embedded_bh
    binaries
    singles
    pairing_timeline
```

The main evolution output boundary is:

```text
EvolutionTimeline
    start_s
    delay_s
    end_s
    merger_s
    merged
    stop_reason
    bsi_timeline
```

The timeline data contracts and ledger-building helpers live under
`src/bsi_trap/timeline/`; the evolution modules update physical state first and
then hand that state to the timeline layer for packaging.

Scenario-level choices live in `ModelConfig`, while BBH integration policy lives
in `BBHEvolverParams`.  The latter now owns the reduced-model numerical knobs
that used to be scattered through the gas/GW evolution code:

- initial timestep fractions for GW-only and gas-assisted integration;
- the merger-radius factor and dimensionless merger-boundary tolerance;
- the maximum integration-loop cap;
- accepted-track decimation for `bbh_tracks`;
- the two GW frequencies used for `ecc0p1` and `ecc10` diagnostics.

Changing those values should therefore happen through the evolver parameter
object rather than by editing `gw.py`, `gas_harden.py`, or `gen_evolve.py`.

## Event Ordering

The initialization timeline should be read as:

```text
BH formation in disk
    -> migration to a trap target
    -> reached trap single reservoir
    -> local encounter with another eligible single
    -> gas-assisted pair-up delay
    -> BBH formation
```

The useful timing relation is:

```text
t_BBH = t_encounter + t_pair
```

where `t_encounter` is the absolute encounter event time stored by the pairing
ledger.  Conceptually, the encounter is sampled after the active reservoir
boundary set by the later arrival and the current trap clock.  The encounter
time is therefore not optional bookkeeping: the pair cannot begin gas-assisted
completion before the two objects have become an eligible local pair.

After BBH formation, the binary evolves through the current gas/GW/BSI routines
until it merges or reaches a stop condition.  If a merger occurs, the remnant
processor computes remnant mass, spin, recoil, retention, escape, and whether
the retained remnant can return to the live single reservoir.  The scheduler
then recomputes the next pairing event from that updated reservoir.

## Current Advantages

The main advantage of the current structure is separation of responsibilities.
The project no longer mixes physical updates, event bookkeeping, and HDF5
payload layout in the same object.

The strongest parts of the current workflow are:

- `TrapContext` is geometry-only; migration and pairing policy live elsewhere.
- Pairing keeps a real event clock with arrival, encounter, pair delay, and BBH
  formation time.
- The local pair-up model uses persistent sampled trap radii and local
  eligibility rather than a fully mixed one-zone reservoir.
- The hierarchy layer owns the chronological pairing/remnant feedback loop
  instead of relying on a fixed generation cap.
- BBH evolution stores accepted trajectory states in `bbh_tracks`, so plots can
  show real evolution paths instead of only initial/final points.
- BBH evolution policy is centralized in `BBHEvolverParams`, reducing hidden
  constants in the gas/GW hardening routines.
- BSI now has a simple and explicit responsibility split: the evolution module
  mutates the binary, while `bsi_trap.timeline` records the applied event time.
- HDF5 output is schema-driven through `storage/payload.py` and
  `storage/schemas.py`.
- Plotting is now focused on high-density diagnostics:
  `bbh_sbh_population_timeline.png` and `bbh_parameter_timeline.png`.

This keeps the current code close to a population-level trap event loop without
forcing storage, plotting, or binary integration to own scheduler policy.

## Current Tasks

The next tasks should be ordered by physical importance and by how much they
change the workflow contract.

1. Build a shared active-BBH/BSI event queue.

   Pairing and retained-remnant recycling now happen on one chronological
   reservoir.  The next framework step is to put active BBHs, smooth evolution
   boundaries, scheduled BSI, and merger processing on one shared clock:

   ```text
   inject newly reached singles
       -> pair eligible singles/remnants
       -> evolve active BBHs to the next event boundary
       -> apply scheduled weak BSI
       -> process mergers and retained/escaped remnants
       -> return retained singles/remnants to the reservoir
       -> record population snapshots until AGN duty-cycle end
   ```

2. Improve the BSI physics kernel.

   The current BSI implementation is weak perturbation only.  It changes
   eccentricity, semimajor axis, and spin tilts, but it does not model
   ionization, ejection, exchange, or a calibrated few-body outcome
   distribution.

3. Calibrate pair-up and BSI rates.

   The current local model is useful as a reduced phase-averaged prescription,
   but the rate normalization and outcome distributions still need physical
   calibration against the intended literature model or a controlled scattering
   calculation.

4. Extend diagnostics around population balance.

   The compact plot tracks active single BHs and BBHs.  Future diagnostics
   should separate migration-origin singles, retained remnants, escaped
   remnants, committed BBHs, active BBHs, and stop reasons across the
   chronological event loop.

## Caveats

The current workflow is scientifically useful as a structured reduced model,
but it should not be read as the final AGN trap population simulation.

Important caveats:

- Pairing and retained-remnant feedback use one live single reservoir, but
  active BBHs and BSI events are not yet evolved in the same shared event queue.
- Pair-up is local and event-driven, but still phase-averaged.  It is not a
  resolved N-body calculation.
- BSI is weak-only.  Ionization, ejection, and exchange are intentionally not
  represented in the current reduced evolution.
- `t_bsi_myr` in HDF5 means an applied BSI event time.  A scheduled but
  unapplied BSI delay remains in `bbh_properties/bsi_timescale_s`.
- Timeline plots are clipped at the AGN duty cycle.  Events beyond that
  boundary are treated as physically irrelevant for the current run.
- BBH trajectory plots reflect stored accepted integrator states.  Sparse tracks
  usually mean sparse accepted states, not necessarily a plotting failure.
- Current compact plots are diagnostics, not final population-inference plots.
- Warning-free tests are part of the project contract because previous issues
  involved import side effects and plotting cache warnings, not only physics
  equations.

## Validation Contract

Use these commands after workflow, storage, evolution, or plotting edits:

```bash
env PYTHONPATH=src python -m compileall -q src tests
```

```bash
env MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  python -W error -m unittest discover -s tests -v
```

For quick plotting checks after a simulation:

```bash
env MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  python -m cli.main plot \
  --summary-path output/simulation_summary.json \
  --output-dir output/plots
```

Expected compact plot outputs are:

```text
bbh_sbh_population_timeline.png
bbh_parameter_timeline.png
plot_manifest.json
```
