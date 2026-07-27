# Simulation Reference — BSI Trap

## Overview

This project simulates the **Binary-Single Interaction (BSI)** dynamics of stellar-mass black hole (BH) binaries embedded in an **Active Galactic Nucleus (AGN) accretion disk**. The core idea: a migration "trap" forms in the disk where BHs accumulate, pair into binaries, and evolve through gas hardening, gravitational-wave (GW) emission, and weak binary-single perturbations until merger.

---

## Physical Pipeline

### 1. Disk Profile (`physics/disk.py`)

- Solves an AGN accretion disk using the **pAGN** library (Sirko or Thompson models).
- Computes radial profiles of: surface density $\Sigma$, sound speed cₛ, scale height H, Toomre Q.
- **Trap radius** is selected near $\Gamma_{\rm tot}=0$ — the location where migration reverses and BHs accumulate.

### 2. Embedded Population (`physics/population.py`)

- Samples BH masses from an external remnant-mass catalog (`input/input_data.txt`).
- Places BHs at random radii between r_in and r_out with a power-law distribution.
- Each BH is independent at this stage; damping population effects are ignored.

### 3. Migration & Arrivals (`physics/population.py → construct_arrivals`)

- Each embedded BH migrates inward due to disk torques.
- **Arrival time** at the trap is mass-dependent: more massive BHs migrate faster.
- BHs that arrive within the AGN duty cycle are marked as "reached" and sorted by arrival time.
- Simple migration timescale: $t_{\rm mig} = M_{\bullet}^2h^2/(m\Sigma r^2 \Omega)$.


### 4. Pair-Up into Binaries (`physics/pairing.py`)

- Once BHs reach the trap, they can **pair up** into binaries.
- **Pair-up timescale** (Qian et al. APJ 2024):
  $$
  t_{\rm pair} = 3 c_s^3 / (4\pi·G^2·\rho·(m_1 + m_2))
  $$
  where $\rho$ is the midplane gas density at the trap.
- Pairing is stochastic: random selection from arrived singles, with a Hill-radius-weighted acceptance criterion.
- no longer useing an **instant pair-up** assumption — the secondary's origin needs to be tracked explicitly.

### 5. Binary Evolution (`physics/evolution.py`)

Each active BBH evolves through three channels:

| Channel              | Effect                                                                          | Status           |
| -------------------- | ------------------------------------------------------------------------------- | ---------------- |
| **Gas hardening**    | Disk torques shrink the orbit                                                   | Stub (returns 0) |
| **GW emission**      | Peters formula hardening + circularization                                      | Stub (returns 0) |
| **BSI perturbation** | Weak encounter with a third single BH hardens the binary and pumps eccentricity | Implemented      |

- **BSI encounter rate** depends on trap volume, cross-section (∝ SMA), and relative velocity (Hill-scaled).
- **BSI outcome**: `a → a × (1 - hardening_fraction)`, `e → e + pump × (1 - e)`.
- **Merger**: when binary reaches coalescence, remnant mass ≈ 0.95 × (m₁ + m₂), spin from fit.

---

## Simulation Loop (`simulation/run.py`)

```
run_simulation(config):
    1. validate config
    2. build_disk_profile(config)        → DiskProfile
    3. sample_embedded_population(...)    → EmbeddedPopulation
    4. construct_arrivals(...)            → ArrivalTimeline
    5. bbh_list = BinaryCollect()
    6. while current_time < AGN_lifetime:
         pairing_kernal(...)              → forms BBHs, appends to bbh_list
         evolve_kernal(...)               → evolves active BBHs (stub)
```

### Event-Driven Architecture (planned)

The design document (`structure_outline.md`) describes a future event-driven loop:

```
inject new arrivals → pair singles → evolve binaries → process BSI → process mergers
→ handle kicks/ejections → re-enter remnants → snapshot
```

Smooth evolution (gas/GW) is continuous; discrete events (BSI, merger, kick) cause state jumps.

---

## Data Model (`simulation/models.py`)

| Class                | Purpose                                                                                |
| -------------------- | -------------------------------------------------------------------------------------- |
| `DemoConfig`         | All simulation parameters (SMBH mass, AGN lifetime, population size, BSI params, etc.) |
| `DiskProfile`        | Radial arrays + trap properties (σ, cₛ, H/R, Q at trap)                                |
| `EmbeddedPopulation` | Mass and initial radius arrays for all embedded BHs                                    |
| `ArrivalTimeline`    | Arrival times, masses, reach masks, and ordering for trap injection                    |
| `PairingTimeline`    | Pair formation records (indices, delays, formation times)                              |
| `BinaryPopulation`   | Single BBH record (component indices, masses, SMA, eccentricity)                       |
| `BinaryCollect`      | Container managing a list of active BBHs                                               |
| `EvolutionTimeline`  | Per-binary event history (gas/GW delays, BSI events, merger times, channels)           |
| `DemoResult`         | Final aggregated result with all stages                                                |

---

## Key Equations

### Toomre Q
$$Q = \frac{c_s \Omega}{\pi G \Sigma}$$

### Migration rate
$$\frac{dR}{dt} = \frac{2\Gamma_{\rm tot}(R)}{m_\star}\sqrt{\frac{R}{GM_\bullet}}$$

### Pair-up timescale
$$t_{\rm pair} = \frac{3c_s^3 h R}{4\pi G^2 \Sigma_g m_1}$$

### BSI encounter rate
$$\Gamma = \sum_i \frac{\pi p_{\rm max}^2 \left(1 + \frac{2G(M_{\rm bin}+m_i)}{p_{\rm max} v_\infty^2}\right) v_\infty}{V_{\rm trap}}$$

### Merger remnant
$$M_{\rm rem} \approx 0.95(m_1 + m_2), \quad \chi_{\rm final} = \mathrm{clip}(3.464\eta - 2.029\eta^2, 0, 0.99)$$
where $\eta = m_1m_2/(m_1+m_2)^2$

---

## Current Status & Gaps

### Implemented
- Disk profile solving via pAGN (Sirko/Thompson)
- Population sampling from catalog
- Mass-dependent migration & arrival timeline
- Stochastic pair-up with Hill-radius weighting
- BSI perturbation (single-step hardening + eccentricity pump)
- Merger remnant estimation

### Stubs (need implementation)
- `gw_acc()` — GW-driven da/dt, de/dt (Peters 1964)
- `gas_acc()` — gas-driven da/dt, de/dt
- `apply_state_change()` — apply smooth evolution to BBHs
- `evolve_kernal()` — main evolution loop integrating all channels
- Re-entry / kick / ejection handling
- Multi-generation tagging (remnants can pair again)

### Known Limitations
- No vertical structure in pairing (2D disk assumption)
- No population feedback (damping, trap overload)

---

## Configuration Parameters (`DemoConfig`)

| Parameter                  | Default | Meaning                             |
| -------------------------- | ------- | ----------------------------------- |
| `smbh_mass_msun`           | 1e6     | Central SMBH mass [M☉]              |
| `agn_lifetime_myr`         | 10.0    | AGN disk lifetime [Myr]             |
| `population_size`          | 500     | Number of embedded BHs              |
| `disk_model`               | "sirko" | pAGN disk model (sirko/thompson)    |
| `alpha`                    | 0.01    | Disk viscosity parameter            |
| `migration_time_myr`       | 0.7     | Characteristic migration timescale  |
| `initial_bbh_sma_au`       | 0.01    | Initial binary semi-major axis [AU] |
| `initial_bbh_eccentricity` | 0.3     | Initial binary eccentricity         |
| `bsi_cross_section_sma`    | 3.0     | BSI cross-section multiplier × SMA  |
| `bsi_hardening_fraction`   | 0.15    | Fractional SMA reduction per BSI    |
| `bsi_eccentricity_pump`    | 0.2     | Eccentricity increase per BSI       |

---

## File Map

```
src/bsi_trap/
├── units.py                    # CGS constants (G, c, Msun, AU, Myr, ...)
├── constant/
│   ├── constant.py             # Duplicate CGS constants + StopReasons
│   └── stop_reason.py          # Stop condition enum
├── dataoi/
│   ├── catalog.py              # Remnant mass catalog loader
│   └── summary.py              # Result formatting
├── physics/
│   ├── disk.py                 # pAGN disk solver → DiskProfile
│   ├── population.py           # Population sampling + arrival timeline
│   ├── pairing.py              # Pair-up logic + timescale
│   └── evolution.py            # BSI, merger, GW/gas stubs
├── simulation/
│   ├── models.py               # All dataclasses (Config, Profile, Timeline, etc.)
│   └── run.py                  # Simulation runner + validation
└── demo.py                     # Entry point: run_demo(), demo_summary()
```

---

## References

- Vaccaro Paola, A&A 2024, 2026 — Migration traps, torques
- Wang Yihan, APJ 2024 — Embedded BH populations
- Fan, APJ 2024 — AGN disk BH demographics
- Gangardt, MNRAS 2024 — Disk structure, vertical equilibrium
- Qian et al., APJ 2024 — Pair-up timescale (dynamical friction)
- Peters 1964 — GW inspiral formula
- pAGN library — Sirko & Thompson disk models
