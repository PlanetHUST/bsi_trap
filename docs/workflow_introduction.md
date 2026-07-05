# BSI Trap Project Workflow Introduction

This document is the consolidated workflow introduction for the current
`bsi_trap` project.  It uses `docs/structure_outline.md` as the main project
spine and folds in the implemented initialization, evolution, storage, and
diagnostic workflow described across the other local notes.

The current codebase is best understood as an AGN migration-trap BBH event
workflow.  It builds an AGN disk and embedded black-hole reservoir, computes
trap-arrival times, lets a chronological scheduler commit pairings, evolves each
committed BBH through gas/GW/BSI effects, computes merger-remnant
mass/spin/kick and retention, and recycles retained remnants into the live
single-BH reservoir until no further accepted pair forms before the AGN duty
cycle ends.

## Physical Scenario

The modeled channel is:

```text
SMBH + AGN disk
    -> embedded BH reservoir from an NSC/catalog population
    -> migration toward mass-dependent trap targets
    -> trap-arrival single-BH reservoir
    -> finite pair-up at/near the trap
    -> committed BBHs plus unpaired trap singles/remnants
    -> BBH hardening, GW evolution, and scheduled BSI effects
    -> merger/non-merger state and remnant feedback recorded to HDF5
```

The physical reason for not treating the trap as an instantaneous BBH factory
is that finite disk thickness and local encounter geometry can suppress the
single-single capture rate.  The trap can therefore contain both single BHs and
BBHs, making binary-single interactions dynamically relevant.

The current pair-up and BSI treatment is reduced and local.  It is not yet a
resolved many-body scattering calculation, but the code now keeps the main
state responsibilities separate:

- pair-up creates the next committed BBH and records its formation time;
- retained remnants return to the live single reservoir when retained;
- BBH evolution changes orbital parameters for committed binaries;
- BSI changes binary parameters through a reduced weak perturbation;
- timelines record when discrete events happened.

## Code-Level Workflow

The active sample-level entry point is:

```text
src/workflow/sample_run.py::run_physics_sample()
```

Its high-level sequence is:

```text
ModelConfig + SMBH mass + sample seed
        |
        v
make_environment()
        |
        v
make_trap_population(pair_initial=False)
        |
        v
evolve_hierarchical_generations()
        |
        v
construct_pairing_up_timeline(max_events=1)
        |
        v
BBH_params_initialize() + make_generation()
        |
        v
generation_data() + save_generation_to_file()
        |
        v
sample manifest + analysis/plot workflows
```

The workflow layer should remain orchestration only.  Physical contracts,
population containers, storage schema, and plotting summaries live under
`src/bsi_trap/`.

## Main Modules and Ownership

| Layer | Code | Responsibility |
| --- | --- | --- |
| CLI | `src/cli/main.py`, `src/cli/simulate_command.py`, `src/cli/analyze_command.py`, `src/cli/plot_command.py` | Parse terminal options and call workflow entry points. |
| Workflow orchestration | `src/workflow/sample_run.py` | Build one physics sample without owning HDF5 schema details. |
| Parallel execution | `src/workflow/parallel_manager.py`, `src/bsi_trap/parallel_compute/` | Run samples and write worker metadata. |
| Environment | `src/bsi_trap/initialization/environment.py` | Build AGN properties, pAGN disk profile, and shared reference trap. |
| Embedded population | `src/bsi_trap/initialization/embedded_bh.py` | Sample embedded BH masses, formation times, and initial radii. |
| Trap population | `src/bsi_trap/initialization/trap_population.py` | Prepare embedded-BH migration and trap-arrival state; optional `pair_initial=True` still builds standalone initial pairing diagnostics. |
| Pairing physics | `src/bsi_trap/evolve/pairing.py`, `src/bsi_trap/functions/pairing_physics.py` | Build event-driven pair-up timelines; the scheduler can request only the next accepted event with `max_events=1`. |
| Hierarchical event scheduler | `src/bsi_trap/hierarchical/generations.py` | Own the chronological pairing/remnant recycling loop until no later pair forms inside the AGN duty cycle. |
| BBH parameters | `src/bsi_trap/initialization/bbh_parameters.py` | Assign BBH semimajor axes, eccentricities, spins, and tilts. |
| BBH evolution | `src/bsi_trap/evolve/gen_evolve.py`, `binary_evolution.py`, `gas_harden.py`, `gw.py` | Evolve committed BBHs through gas/GW dynamics and BSI effects while sharing orbit bookkeeping. |
| BSI effects | `src/bsi_trap/evolve/bsi.py` | Apply scheduled weak BSI perturbations and construct the BSI event ledger. |
| Timelines | `src/bsi_trap/evolve/timeline.py` | Store event/output timelines; not the place for physical updates. |
| Storage | `src/bsi_trap/storage/` | Own HDF5 contract, payload construction, combination, and summaries. |
| Analysis/plots | `src/workflow/analysis_run.py`, `src/workflow/plot_run.py` | Summarize run metadata and generate compact timeline diagnostics. |

## Configuration and Random Streams

`ModelConfig` defines the scenario:

- gas mode: `with_gas` or `without_gas`;
- disk model: `sirko` or `thompson`;
- AGN lifetime, alpha, and disk resolution;
- pair-up and BSI switches/parameters;
- local trap geometry parameters.

`make_rng_streams()` splits the sample seed into independent streams:

```text
rng.nsc       embedded population sampling
rng.pair_up   pairing radii and pair selection
rng.dynamics  BBH orbital sampling
rng.spin      spin magnitude and spin tilt sampling
```

This separation matters because changing one stochastic stage should not
silently change the others for the same sample seed.

## Environment Construction

`make_environment()` builds the AGN and disk context:

1. Create `AGN_properties` from SMBH mass and AGN lifetime.
2. Build a pAGN disk profile.
3. Read the filtered remnant catalog.
4. Find the catalog-median reference trap and anti-trap.
5. Create a CGS `DiskProfile`.
6. Attach the disk, pAGN disk object, and torque profile to `AGN_properties`.

Important disk quantities are surface density `Sigma`, sound speed `c_s`,
aspect ratio `H/R`, and the trap radius.  The disk scale height is

```text
H = c_s / Omega
```

and the Toomre stability parameter is

```text
Q = c_s Omega / (G Sigma).
```

The reference trap defines the local reaction zone used by the current reduced
trap model, but individual BH migration targets are still mass-dependent.

## Embedded BH Reservoir

`embedded_BH_initialize()` samples a reservoir of disk-embedded BHs from the
filtered remnant catalog.  Each object receives:

- `m_msun`: BH mass;
- `t_form_s`: formation time;
- `R_cm`: initial disk radius.

The reservoir is the source population.  At this stage, no object has yet been
classified as paired, single in the trap, or failed to reach the trap.

## Migration and Trap Admission

`resolve_mass_dependent_trap_migration()` calculates stable targets and
migration times.  For each embedded BH:

```text
t_reach_s = t_form_s + t_mig_s
```

An object is admitted to the trap channel only if it has a stable target and
arrives before the AGN lifetime:

```text
stable_mask and (t_reach_s < AGN_lifetime)
```

Objects without a stable target, invalid migration route, or timely arrival
remain in the embedded reservoir and are not admitted to the live trap-reservoir
events.

## Local Trap Geometry

`TrapContext` is intentionally geometry-only.  It does not decide migration
targets and does not encode pAGN-specific policy.

At the shared reference trap, the local annulus is:

```text
H_gas   = (H/R)_gas R_trap
Delta_R = trap_width_scale_heights H_gas
H_BH    = compact_object_scale_height_fraction H_gas
V_trap  = 4 pi R_trap Delta_R H_BH
```

Each reached BH receives a persistent local pairing radius:

```text
r_pair = r_trap_target + Normal(0, trap_radial_dispersion_scale_heights H_gas)
```

clipped to the configured trap annulus.  This keeps local geometry explicit
instead of treating all reached BHs as a fully mixed one-zone bath.

## Pair-Up and Committed BBHs

The authoritative pair-up function is:

```text
src/bsi_trap/evolve/pairing.py::construct_pairing_up_timeline()
```

Reached BHs are processed in `t_reach_s` order.  Arrived unpaired BHs enter a
waiting reservoir.  Candidate pairs are filtered by local radial overlap and a
strong-gravity gate.  The current phase-averaged conjunction rate is based on
the difference in local orbital frequencies:

```text
lambda_ij = |Omega_1 - Omega_2| / (2 pi)
```

After a conjunction, gas-assisted pair completion is represented by:

```text
rho_g  = Sigma / (2 H_gas)
t_pair = 3 c_s^3 / (4 pi G^2 rho_g (m1 + m2))
t_BBH  = t_encounter + t_pair
```

A pair is accepted only if the resulting BBH formation time is inside the AGN
lifetime.  Both embedded-BH members then store reciprocal `trap_flag` indices
and the same `t_pair_s`.

In the main sample workflow, `make_trap_population(pair_initial=False)` stops
after migration and trap admission.  `evolve_hierarchical_generations()` then
owns the live reservoir.  At each scheduler step it asks
`construct_pairing_up_timeline(..., max_events=1)` for the next accepted pair,
initializes only that committed BBH population, evolves it, and returns retained
merger remnants to the live singles before requesting the following pair.

The conversion step creates the population for the current committed event:

- `BBH_properties` from reciprocal embedded-BH pairs;
- `Single_BH_properties` from reached, unpaired trap objects.

The conservation condition is:

```text
number_reached = 2 * number_BBH + number_trap_singles
```

## Initial BBH Parameters

`BBH_params_initialize()` fills the BBH internal parameters after pair
formation:

- `sma_cm`: semimajor axis, bounded by local trap/Hill constraints;
- `ecc`: eccentricity, sampled from a thermal distribution;
- `chi1`, `chi2`: spin magnitudes;
- `th1`, `th2`: spin tilts, with gas-dependent alignment policy.

These are the initial conditions for the BBH evolution stage.

## BBH Evolution and Timelines

`make_generation()` evolves one committed BBH population.  For each BBH:

```text
evolve_single_binary()
    -> evolve_gas()      if gas_mode == "with_gas"
    -> evolve_gw_only()  if gas_mode == "without_gas"
```

The smooth evolution updates orbital state through gas hardening and/or Peters
GW evolution.  The shared orbit loop in `binary_evolution.py` owns the mutable
bookkeeping that both `gas_harden.py` and `gw.py` need: stop conditions, BSI
step boundaries, frequency probes, accepted-track recording, and final state
writeback to `BBH_properties`.

The hardening routines should not carry scenario policy as anonymous literals.
`BBHEvolverParams` owns the current integration and diagnostic policy:

- `gw_initial_step_fraction` and `gas_initial_step_fraction`;
- `merger_radius_factor` and `merger_boundary_fraction`;
- `max_evolution_loops`;
- `track_max_smooth_points`;
- `eccentricity_probe_low_hz` and `eccentricity_probe_high_hz`.

The gas and GW paths use the same merger-boundary tolerance, loop cap, probe
frequencies, and track-decimation policy.  This keeps the reduced model easier
to tune without changing the physics modules directly.

`construct_bbh_evolution_timeline()` then builds `EvolutionTimeline` from the
evolved BBH object:

- start time;
- delay time;
- end time;
- merger time for merged binaries;
- stop reason;
- embedded `BSI_Timeline`.

After BBH evolution, `build_hierarchical_mergers()` computes merger-remnant
mass, spin, GW recoil, disk retention, trap-overload status, and the retained
single-BH seeds that can feed a later hierarchical generation.

`evolve_hierarchical_generations()` owns the outer event loop.  It starts from
the migration-arrival singles, commits one accepted pair at a time, evolves that
committed BBH population, mixes retained remnants back into the live singles,
and stops when fewer than two live singles remain or no further pair forms
inside the AGN duty cycle.  Each committed event keeps its own pairing timeline,
evolution timeline, and remnant ledger.

The important design rule is that a timeline records event times and outcomes;
it does not own the physical update.

## BSI Responsibility Split

BSI is now represented by two responsibilities:

1. `apply_scheduled_bsi_step()` changes the binary during evolution.
2. `construct_bsi_timeline()` records when the event happened.

The scheduled BSI delay is stored on the BBH as `bsi_timescale_s`.  When an
integration step crosses that delay and `model_config.bsi_enabled` is true,
the smooth gas/GW state is first advanced to the scheduled time, then the BSI
effect is applied to that current orbital state.

For weak BSI events, the reduced model:

- increases eccentricity;
- hardens the binary by reducing semimajor axis;
- perturbs spin-tilt angles;
- records `bsi_weak_count`, `bsi_delta_e_total`, and
  `bsi_delta_theta_total`.

`BSI_Timeline` records the absolute time of each applied weak BSI event:

```text
t_BSI = t_ini_s + bsi_timescale_s
```

In HDF5, `bbh_properties/bsi_timescale_s` remains the scheduled delay.  The
`timescales/t_bsi_myr` field is an applied-event time: it is finite only when
the BSI flag shows that the scheduled perturbation actually occurred.

Ionization, ejection, and exchange are not represented in the current reduced
BSI evolution.  The in-memory flags and HDF5 flag payload expose only the
applied BSI event marker, so `apply_scheduled_bsi_step()` never stops a binary
because of BSI.

## Storage Contract

The workflow does not hard-code HDF5 layout.  Storage responsibilities are:

- `storage/payload.py`: converts in-memory state into HDF5-ready payloads;
- `storage/schemas.py`: defines groups and fields;
- `storage/save_gen.py`: writes a single generation file;
- `storage/combined.py`: combines per-sample HDF5 files;
- `storage/summary.py`: reads compact generation summaries for lightweight reports.

The main HDF5 groups are:

```text
agn_properties
bbh_properties
flags
timescales
trap_population
```

The trap population group preserves embedded-BH migration/pairing history,
including `t_form_s`, `t_mig_s`, `t_reach_s`, `t_pair_s`, `r_pair_cm`, and the
pairing timeline arrays.

## Analysis and Visualization

`run_analysis_workflow()` reads `simulation_summary.json` and writes aggregate
run metadata, including success counts, merger counts, output files, manifests,
and worker metadata.

`run_plotting_workflow()` generates compact timeline diagnostics from readable
generation HDF5 files:

- `bbh_sbh_population_timeline.png`: active trap single-BH and BBH counts,
  including BBH removal at merger;
- `bbh_parameter_timeline.png`: BBH orbital evolution tracks in the
  energy/angular-momentum analogue plane.

The diagnostic initialization test can also generate a more detailed
trap-population timeline figure set:

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  python tests/test_trap_population_initialization.py \
  --diagnostic \
  --embedded-bh-num 1000 \
  --pair-up-eta 2.5 \
  --seed 12345 \
  --output-dir report/trap_population_timeline
```

## Validation Path

Run from the repository root:

```bash
PYTHONPATH=src python -m compileall -q src tests
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  python -W error -m unittest discover -s tests -v
```

The focused test coverage currently checks:

- trap-population conservation;
- migration and arrival-time state;
- omission of objects without stable mass-dependent traps;
- local pair eligibility and reproducibility;
- BBH orbital/spin initialization;
- BBH evolution timeline construction;
- BSI effect application before timeline recording;
- HDF5 generation payload and workflow timeline plots.

## Current Limits and Next Extensions

Implemented now:

- AGN/disk environment construction;
- embedded BH sampling;
- mass-dependent migration targets;
- migration-only trap-arrival population construction;
- event-driven local pair-up;
- chronological pairing/remnant recycling through
  `evolve_hierarchical_generations()`;
- committed-BBH and trap-single initialization;
- gas/GW evolution;
- scheduled reduced BSI effects;
- per-generation HDF5 storage;
- analysis and compact timeline visualization.

Still incomplete:

- a single shared event queue for active BBHs, scheduled BSI events, and
  pair/remnant reservoir updates;
- physical BSI exchange, ejection, ionization, or prompt-merger channels;
- production-grade scientific calibration of the reduced pair-up and BSI
  prescriptions.

The next major framework step should be the shared active-BBH event loop:

```text
inject newly arrived singles
    -> pair available singles
    -> evolve active BBHs
    -> apply scheduled weak BSI from the live single reservoir
    -> process mergers and remnants
    -> return retained remnants/singles to the trap
    -> record population snapshots until AGN end
```

That extension should reuse the current contracts instead of rebuilding them:
`TrapPopulation` for the initialized state, `BBH_properties` and
`Single_BH_properties` for mutable populations, `EvolutionTimeline` and
`BSI_Timeline` for event ledgers, and `storage/payload.py` for output
structure.
