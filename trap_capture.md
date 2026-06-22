# Trap capture and initial pairing

## Scope

This document describes the first-generation sBH pairing model used by
`construct_pairing_up_timeline()` in `src/bsi_trap/evolve/pairing.py`. It
constructs initial BBH formation times after embedded BHs have migrated to
stable disk traps. It is an event-driven, phase-averaged approximation, not a
continuous N-body or hydrodynamic calculation.

## State at trap arrival

Migration supplies each embedded BH with a target trap radius
`r_trap_target_cm` and an arrival time `t_reach_s`. For every BH that reaches
before the AGN lifetime ends, pairing initialization samples and stores one
local radius, `r_pair_cm`:

\[
r_i = r_{\mathrm{target},i} + \delta r_i.
\]

`delta r_i` is a normal draw with standard deviation
`trap_radial_dispersion_scale_heights * H_gas`, clipped to half the configured
trap width. The default radial dispersion is `0.25 H_gas`; the default full
trap width is `1.0 H_gas`. The radius is sampled once, at initialization, and
is held fixed while the first-generation pairing timeline is built. This is a
quasi-static approximation: it is appropriate only when residual radial drift
is slow compared with encounter times.

## Local pair eligibility

For each pair of arrived, unpaired sBHs, the code evaluates the mean radius
`r_bar`, radial separation, and the Hill-scale encounter distance:

\[
b_{ij}=\min\left[\eta_{\rm pair} r_{\rm bar}
\left(\frac{m_i+m_j}{3M_{\rm SMBH}}\right)^{1/3},
\frac{\Delta R_{\rm trap}}{2}\right].
\]

The pair is local only if `abs(r_i - r_j) < b_ij`. Its relative speed is the
Keplerian shear speed calculated from the two circular frequencies:

\[
v_{\rm rel}=\bar r\,|\Omega_i-\Omega_j|,
\qquad
\Omega_i=\sqrt{GM_{\rm SMBH}/r_i^3}.
\]

The pair must also pass the strong-gravity gate:

\[
v_{\rm rel}^2 < \frac{2G(m_i+m_j)}{b_{ij}}.
\]

This condition says that gravity dominates the shear kinetic energy at the
encounter scale. It selects a strong encounter; it does **not** by itself show
that an initially unbound pair becomes a bound BBH. Gas dissipation at close
passage is represented only phenomenologically by the subsequent pair-up
delay.

## Encounter rate and timeline

For randomly phased circular orbits, each eligible pair has conjunction rate

\[
\lambda_{ij}=\frac{|\Omega_i-\Omega_j|}{2\pi}.
\]

At time `t`, the total population rate is

\[
\Lambda(t)=\sum_{i<j\,\in\,\mathrm{eligible\ singles}}\lambda_{ij}.
\]

The implementation samples the next encounter from
`Exponential(1 / Lambda)` and samples its pair proportional to `lambda_ij`.
An arrival updates the single-BH population, which updates the candidate set
and therefore the rate. A successful selected pair is removed from the single
population and receives the shared formation time

\[
t_{\rm pair}=t_{\rm encounter}+t_{\rm delay},
\qquad
t_{\rm delay}=\frac{3c_s^3}{4\pi G^2\rho_g(m_i+m_j)},
\quad \rho_g=\frac{\Sigma}{2H}.
\]

## Limitations

- Orbital phases are averaged over; the code does not store azimuths or
  schedule deterministic conjunctions.
- Pairing radii remain fixed during initialization; later work can evolve them
  between events if residual migration becomes important.
- The strong-gravity gate is not a gas-capture-energy calculation. A complete
  model requires a prescription for `Delta E_gas` and the boundness condition
  `0.5 * mu * v_rel^2 - Delta E_gas < 0`.
- The finite-layer `sbh_cross_section()` helper remains available for other
  kinetic models, but this timeline uses local conjunction rates instead of
  the global `sigma * v / V` bath rate.

## Physical references

- Bellovary et al., [Migration Traps in Disks Around Supermassive Black
  Holes](https://arxiv.org/abs/1511.00005), for migration-trap accumulation.
- Whitehead et al., [Gas Assisted Binary Black Hole Formation in AGN
  Discs](https://arxiv.org/abs/2309.11561), for the distinction between a
  strong encounter and gas-assisted bound capture.
