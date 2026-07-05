# Physical Scenario
## Initialization of disk embedded BHs
### population characteristics
ignore the maybe impact of damping population of embedded BHs

ref: Vaccaro Paola A&A 2024, 2026; Wang Yihan APJ 2024; Fan APJ 2024. 
### test for independence

## Disk profile and torques
### solve disk profile
using pAGN modula in python

Toomre parameter:
$$
Q=\frac{c_s \Omega}{G\Sigma}
$$

stable disk profile, with temperature $T$, surface density $\Sigma$, sound speed $c_s$

vertical equillibrum $\longrightarrow$ disk critical height $H=c_s/\Omega$

vertical density $\rho(z)=\rho(0)\exp(-(z/H)^2)$

ref: Gangardt MNRAS 2024

### torques
$$\Gamma_{\rm tot}=\Gamma_{\rm type,i} + \Gamma_{\rm th} + \Gamma_{\rm type,ii}$$
which define the migration properties of all embedded stars/BHs, via
$$\frac{dR}{dt}=\frac{2\Gamma_{\rm tot}(R)}{m_\star}\sqrt{\frac{R}{GM_{\bullet}}}$$

mass dependence and trap width

ref: Vaccaro Paolo A&A 2026

### test for different AGN disk lifetimes

## Pair-up forming binaries
### instant pair-up assumption and weakness
first the pair-up events often occur in/near the trap, so in the first order, assuming instant pair-up in the trap, without physical considering where the secondary BH comes is qualitative right. When considersing the influence of BSI, the binary and single population in the trap needs in quantity, then it's need adding more pair-up details

### micro-dynamic consideration
assume only bbh pair-up happen in the trap, consider arrange binary pair up in the trap reach time list, one flag for embedded BH to list it as pair-up with others, or just single BH in the trap

setting one pair-up rate based on model config to consturct the bbh and sbh population in the trap

based on (Qian et al. APJ 2024) pair-up timescale as the dynamical damping process between the two stellar objects:
$$t_{\rm pair}=\frac{3c_s^3hR}{4\pi G^2\Sigma_{\rm g}*m_1}$$

### considering the vertical structure
not for 2d disk assumption, but also consider the vertical structure

## binary evolution in the trap

## description of BSI
### interaction regimes
assume hyperbolic third BH trajectory relative to the binary, with velocity $v_{\infty}$

conserved total energy, now using the sma description for the bind energy, viral formula, and change with energy eqs to change of sma

$$E_{\rm tot}=\frac{1}{2}(m_1+m_2)v_b^2-\frac{Gm_1m_2}{a}+\frac{1}{2}m_3v_{\infty}^2$$

- weak perturbation: view binary as point mass in the third BH
$$|\mathbf{r_b}-\mathbf{r_3}|=\frac{2G(m_1+m_2)}{v_3^2-v_{\infty}^2}$$

## hierarchical mergers
large density of BHs in the trap, after calculate the binary and single population, the binary evolution can be independent

---

# Code Workflow

## Task list

* [ ] Initialization trap population (contain pair-up dynamics);
* [ ] Evolution trap binary to merge;
* [ ] Data storage to hdf5 files;
* [ ] Data analysis and visualization;

## General consideration

- generate alternative simulating channels $(A, B, C,\dots)$;

    model_config.py, sets considering variables.
    1. damp, linear, log
    2. gas hardening, no gas hardening
    3. torque derived migration, first order simplified migration
    4. pair up $\dots$
    5. binary-single interaction $\dots$

- parallel computatation for results (MC simulation for each sampling).

    >physical_simulation/
    - initialize.py
    - hierachical_merger.py
    - binary_evolve.py
    - storage.py
    - classes.py
    - agn_function.py

    >physical_simulation/disk_profile/
    - pagn_adapter.py
    - migration_torque.py 


- hierarchical merger data storage, rather than add them all for statistics, view smbh mass dependence

    physical quantities relevant to observation: 
    1. absolute merge time during the simulaion $t_{\rm merg}$
    2. masses, $m_1, m_2, m_{\rm rem}$
    3. spins modula and theta
    4. eccentricity at one certain frequency

    others are derived quantities; neat storage .hdf5 file logic, channel-sample-data

- data analysis for plot

- simulation workflow
    > run_physics_sample()

        1. build AGN/disk
        2. build embedded BH population
        3. convert embedded BHs into BBHs + single BHs at trap
        4. initialize BBH orbital/spin parameters
        5. evolve BBHs with gas/GW and scheduled weak BSI
        6. construct pairing/evolution/BSI/remnant ledgers
        7. build retained-remnant next-generation inputs
        8. save generation HDF5 payloads and sample metadata

---

_timeline_ logs the discrete evlotion time line for each *pair-up binary*

the most important part in the evolution workflow 

|            | discrete process                     | continue process |
| ---------- | ------------------------------------ | ---------------- |
| for binary | scheduled BSI perturbation; merge; pair-up formation | gas/GW evolution |
| for single | to pair-up                           | reservoir waiting |

## It's important to make one demo evolution part

---

## Naive workflow

```python
def evolve_population_with_timeline(...):
    t = 0.0
    timeline.record_population_start(...)
    while t < agn.lifetime_s:
        active = get_active_bbh_indices(bbh)
        dt = choose_population_timestep(bbh, sbh, t, agn)
        for i in active:
            before = timeline.snapshot_bbh(bbh, i, t)
            evolve_one_bbh_segment(...)
            after_smooth = timeline.snapshot_bbh(bbh, i, t + dt)
            timeline.record_smooth_segment(...)
            if should_do_bsi(bbh, sbh, i, dt, rng_bsi):
                before_bsi = timeline.snapshot_bbh(bbh, i, t + dt)
                outcome = apply_bsi_event(...)
                after_bsi = timeline.snapshot_bbh(bbh, i, t + dt)
                timeline.record_bsi_event(...)
            check_final_state_and_record(...)
        t += dt
    timeline.record_population_end(...)
    return bbh, sbh, timeline
```

Current sample workflow:
```python
def run_physics_sample(...):
    environment = make_environment(...)
    population = make_trap_population(...)
    evolution = make_generation(...)
    payload = generation_data(...)
    save_generation_to_file(payload, ...)
    return sample_manifest
```

Configuration boundary:

- `ModelConfig` defines scenario switches and population-level model choices;
- `BBHEvolverParams` defines reduced BBH integration policy such as timestep
  seed fractions, merger-boundary tolerance, loop cap, track decimation, and
  diagnostic GW probe frequencies.

### Detail evolution workflow

many embedded BHs reach trap at different times
    ↓
trap population changes with time
    ↓
singles pair into BBHs
    ↓
BBHs harden / interact / merge / get kicked
    ↓
remnants or kicked binaries may leave, return, or re-enter
    ↓
trap population continues until AGN duty-cycle end

initialize all embedded BHs
compute their trap-reaching times
sort by reaching time

for time from 0 to AGN duty-cycle end:
    1. inject newly reached BHs into trap
    2. inject re-entered binaries/remnants if their return time has arrived
    3. update active binaries
    4. process BSI events
    5. process mergers
    6. process kicks/ejections/reentries
    7. pair available singles/remnants into new BBHs
    8. record trap population snapshot

embedded BH reservoir
        ↓ t_reach
trap singles/remnants reservoir
        ↓ pair-up
active BBH reservoir
        ↓ gas/GW/BSI
merger/ejection/reentry channels
        ↓
back to trap or out of system

based on event-driven scenario:

- smooth binary parameters evolution: gw and gas-hardening;
- state jump: scheduled weak bsi perturbation;
- merge to remanants, binary -> single high generation BH;
- kick out of the trap and re-enter: gw kick, single re-enter;

that's the detail trap population evolution

### Pair-up dynammics

considering cross section and trap volumn, maybe one way to extend the instant pair-up assumption and the extre-small t_pair 
