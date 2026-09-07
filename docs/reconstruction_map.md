# Reconstruction map

| Responsibility | bsi_trap owner | IMRI donor | Decision | Status |
|---|---|---|---|---|
| Configuration | `simulation/models.py:DemoConfig` | `workflows/config.py` | ADAPT, keep one validated facade | done |
| Constants and units | `units.py` | `physical_process/constants.py` | CONSOLIDATE in CGS | done |
| pAGN construction | `physics/disk.py` | `disk_quantity/pagn_disk_adapter.py` | ADAPT equations/search | done |
| Disk profile model | `simulation/models.py:DiskProfile` | `PagnDiskProfile` | REPLACE raw-object API with CGS record | done |
| BH population | `physics/population.py` | none needed | ADAPT with permanent IDs | done |
| Migration/arrival | `physics/migration.py`, `physics/population.py` | `physical_process/disk_migration.py` | ADAPT signed torque integration | done |
| Migration traps | `physics/migration_torque.py` | `classify_torque_traps` | ADAPT sign classifier | done |
| Pairing | `physics/pairing.py` | none | REPLACE broken Hill probability with labelled phenomenology | done |
| Binary state | `simulation/models.py:BinaryPopulation` | none | REPLACE partial initializer | done |
| Binary-single interactions | `physics/bsi.py` | none | REPLACE monolithic stub with discrete events | done |
| GW evolution | `physics/evolution.py` | none required | REPLACE stub with Peters equations | done |
| Gas evolution | `physics/evolution.py:gas_*` | none selected | KEEP explicit zero terms | done |
| Merger/remnant | `physics/evolution.py:compute_merger_remnant` | none | ADAPT old approximation and label toy | done |
| Simulation driver | `simulation/run.py` | workflow structure only | REPLACE non-advancing loop with event queue | done |
| Results/summary | `simulation/models.py`, `dataoi/summary.py` | outputs conventions only | REPLACE undefined records with auditable result | done |
| Diagnostics | `scripts/scientific_diagnostics.py`, `scripts/reference_regression.py` | disk/migration/trap scripts | ADAPT production-API witnesses | done |
| Tests | `tests/` | donor functions as reference | CREATE focused fast/integration/regression checks | done |
