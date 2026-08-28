# 6fda O8 state validation

This folder isolates O8 FCH calculations from the production checkpoint. Each
occupation controller has its own checkpoint and logs. The production maxvol
O8 result is read as a baseline but is never modified.

This script uses the configuration-based checkpoint schema introduced after
0.7.1. Pre-refactor calculation HDF5 files must be regenerated with
`--recalculate`; spectra and figures do not need manual migration.

The calculations use the production `6fda-dam_relaxed_dft.xyz` geometry,
PBE/def2-SVPD, the production grid and convergence
tolerance, GPU4PySCF by default, and controller-preserving DIIS recovery.
Second-order SCF is disabled. XCH is disabled
because this test concerns FCH state identity and spin contamination.

Run the tests progressively:

```bash
conda activate pymbxas
cd /home/roncofaber/Insync/GDrive_LBL/WORK/MBXAS/shakeup/6fda/state_tests

# Analyze the existing production maxvol result without SCF.
python o8_state_test.py --analyze-only

# Calculate independent alternative FCH states. Each command is restartable.
python o8_state_test.py --method mom
python o8_state_test.py --method mixed

# Optional independent reproduction of the production controller.
python o8_state_test.py --method maxvol
```

Add `--cpu` only when testing CPU PySCF. Do not combine CPU and GPU results in
the same comparison unless that device comparison is intentional. Use
`--recalculate` only to replace the selected test checkpoint.

Outputs are grouped under `outputs/`:

- `checkpoints/`: one restartable HDF5 file per controller and device;
- `logs/`: separate PyMBXAS and raw PySCF logs;
- `reports/o8_state_comparison.{json,csv}`: energies, spin diagnostics,
  core-hole overlaps, determinant overlaps, local spin, and pairwise state
  comparisons;
- `figures/o8_state_comparison.png`: relative energy, `<S^2>`, and O8 local
  spin;
- `figures/o8_orbitals_*.png`: orbital-rearrangement diagrams.

The most important comparison is whether MOM and maxvol converge to the same
occupied subspace and similar spin state. A core-hole overlap near one only
confirms the identity of the empty 1s orbital; the occupied determinant
overlap and `<S^2>` test the rest of the electronic state.

Files whose checkpoint name lacks `dftgeom` came from an earlier diagnostic
script that accidentally selected `6fda-dam_relaxed.xyz`. They are retained
for provenance but are deliberately ignored because that geometry differs
from the production checkpoint.
