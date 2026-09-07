# Gap-opening conventions

The migration implementation keeps three dimensionless diagnostics separate.
Here `q = m/M_SMBH`, `h = H/R`, and `alpha` is the disk viscosity parameter.

| Name in code | Equation | Threshold/use | Interpretation | Donor path |
|---|---|---|---|---|
| `torque_attenuation_k` | `q^2 / (alpha h^5)` | enters `exp(-K/20)` and `1 + K/25` | attenuation and normalization of Type-I/thermal torque | `imri/src/disk_quantity/migration_torque.py:comp_gap_parameter_dimensionless` |
| `type_ii_transition_kii` | `q^2 / (alpha h^2)` | Type-II when greater than 11 for `condition="kii"` | project Type-II branch selector | `imri/src/physical_process/disk_migration.py:gap_opening_mask` |
| `hill_radius_over_scale_height` | `(q/3)^(1/3) / h` | must exceed 1 for the Crida selector | Hill sphere relative to disk thickness | same donor selector |
| `crida_gap_parameter` | `3/[4(R_H/H)] + 50 alpha h^2/q` | Type-II when less than 1 and `R_H/H > 1` | Crida-like pressure plus viscosity criterion | same donor selector |

These expressions are not aliases and their thresholds must not be exchanged.
The migration branch uses either `kii` or `crida` from configuration; it never
retunes a threshold to change a plotted result.
