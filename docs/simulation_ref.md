# BSI-trap simulation reference

## Scientific scope

The production path follows stellar-mass black holes embedded in a pAGN disk,
computes mass-dependent migration torques and accessible stable torque traps,
forms binaries from co-located active singles, and evolves those binaries with
Peters gravitational radiation plus optional discrete phenomenological
binary-single interactions (BSIs).

Gas hardening is currently exactly zero. The code does not invent a gas term
until a validated prescription is selected.

## Units and radii

pAGN solves in SI. `physics/disk.py` converts its complete profile once to the
simulation's CGS convention:

- mass: g, with explicitly named `mass_msun` values where useful;
- distance: cm;
- time: s;
- speed: cm/s;
- density: g/cm3;
- surface density: g/cm2;
- torque: dyn cm (erg).

The gravitational radius is `r_g = GM/c^2`; the Schwarzschild radius is
`r_s = 2r_g`. Toomre `Q=1` defines the outer self-gravity boundary only. It is
never a migration-trap criterion.

## Production stages

```text
DemoConfig
  -> pAGN disk solve and SI-to-CGS DiskProfile
  -> immutable EmbeddedPopulation (bh_id == original array index)
  -> one torque profile per distinct migrator mass
  -> stable/anti-trap sign-change classification
  -> signed one-dimensional migration and explicit stop reason
  -> BH_ARRIVAL events
  -> phenomenological pairing delay and BINARY_FORMATION events
  -> Peters GW evolution
  -> competing discrete BSI or BINARY_MERGER events
  -> AGN_END and auditable SimulationResult
```

`SimulationState.current_time_s` is the sole clock. Queued events cannot move
it backward. Population arrays never shrink; active, pending-pair, paired, and
merged masks are disjoint. A pending-pair mask is an explicit intermediate
state between selection and formation.

## Migration torque and traps

`physics/migration_torque.py` adapts the equations in the IMRI donor. Formula
evaluation remains SI internally, then all public torque arrays are converted
to CGS using `1 N m = 1e7 dyn cm`. The selected torque is Type-I plus attenuated
thermal torque unless the configured `kii` or `crida` selector chooses Type-II.
The distinct gap diagnostics are listed in `gap_opening_conventions.md`.

For radius increasing outward:

- positive to negative torque is a stable attractor;
- negative to positive torque is an anti-trap.

The migration solver uses `dR/dt = 2 Gamma/(m R Omega)`. It follows the local
sign into the correct basin and stops at a stable trap tolerance, a disk edge,
the AGN lifetime, zero/invalid torque, or another explicit reason. Arrival at a
stable root means entering a fraction of the local grid spacing; exact equality
is not required because the idealized time can diverge at zero torque.

## Pairing, GW evolution, and BSI

Pairing is explicitly phenomenological: two active singles assigned to the same
local trap region receive a gas-damping delay. Candidate positions are mapped
through the active-ID array before becoming physical component IDs. The Hill
radius convention is

`R_H = R [(m1+m2)/(3 M_SMBH)]^(1/3)`.

Peters `da/dt` and `de/dt` use CGS inputs. `6G(m1+m2)/c^2` is only the stated
integration stop; it is not a resolved strong-field merger model. The retained
remnant prescription (`5%` mass loss plus a simple spin polynomial) is labelled
`toy_5pct_mass_loss_spin_polynomial` in every merger record.

BSIs are Poisson events based on a gravitationally focused cross-section and
the current active-single mask. They do not enter the GW derivatives. The
fractional hardening/eccentricity update is explicitly named phenomenological,
and every event records its single BH ID, rate, and before/after binary state.

## Reproducibility and outputs

One `numpy.random.Generator` is created from `DemoConfig.seed` and owned by the
simulation state. Fixed configuration, catalog, and seed reproduce population
sampling, pair choices, event history, and BSI sampling.

`DemoResult` and its alias `SimulationResult` expose configuration, seed, disk,
initial population, migration outcomes, trap assignments, event history,
binaries, pairing records, BSI records, merger records, stop reasons, and final
masks. `dataoi.write_result_json` emits strict JSON: unavailable arrival/trap
values become `null`, never nonstandard `Infinity` or `NaN` tokens.

## Validation entry points

- `python -m pytest` runs fast mathematical, state, pAGN integration, and IMRI
  donor-parity tests.
- `python scripts/scientific_diagnostics.py` creates disk, torque/trap, and
  migration diagnostic figures from production APIs.
- `python scripts/reference_regression.py` compares two SMBH masses and two
  migrator masses against the read-only IMRI donor.
- `python scripts/demo.py` runs the configured end-to-end realization.
