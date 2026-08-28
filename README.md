<div align="center">
  <img src="https://gitlab.com/uploads/-/system/project/avatar/47099716/pymbxas2_1_.png" height="120px"/>
</div>

# PyMBXAS

[![PyPI version](https://badge.fury.io/py/pymbxas.svg)](https://badge.fury.io/py/pymbxas)

PyMBXAS is a molecular many-body X-ray absorption spectroscopy package built
on [PySCF](https://pyscf.org/) and the
[Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/). It runs a
ground-state, full-core-hole, and optional excited-core-hole Delta-SCF
workflow, then evaluates the one-body determinant MBXAS transition amplitude.

Current scope:

- unrestricted UKS or UHF molecular calculations;
- all-electron K edges;
- optional IBO or Boys localization of degenerate core orbitals;
- XCH total-energy alignment;
- HDF5 checkpoints that can be reopened and extended;
- optional experimental overlap-weighted shake-up and spectator-spin
  post-processing.

Periodic MBXAS is deliberately unsupported because the molecular position
operator used for transition dipoles is not periodic.

## Installation

```bash
pip install pymbxas
```

Optional dependencies are grouped by feature:

```bash
pip install "pymbxas[gpu]"   # gpu4pyscf; requires a working CUDA device
pip install "pymbxas[plot]"  # matplotlib plotting helper
pip install "pymbxas[ml]"    # experimental explorer dependencies
```

To install the current development branch:

```bash
pip install "git+https://gitlab.com/roncofaber/pymbxas.git@dev"
```

## Basic calculation

```python
import ase.build

from pymbxas import (
    CalculationConfig, CheckpointConfig, PySCFMBXAS, RuntimeConfig,
    SCFConfig,
)

structure = ase.build.molecule("H2O")

calc = PySCFMBXAS(
    structure,
    config=CalculationConfig(xc="lda", basis="def2-svpd"),
    runtime=RuntimeConfig(
        device="cpu",
        checkpoint=CheckpointConfig(filename="pymbxas.h5"),
    ),
)

calc.run(
    "O",
    occupation="maxvol",
    scf=SCFConfig(max_cycles=150, convergence_tolerance=1e-6),
)
energy, intensity = calc.get_mbxas_spectra("O", sigma=0.5)
```

For inexpensive convergence tests, PySCF controls belong in `SCFConfig`, for
example `max_cycles=150`, `convergence_tolerance=1e-6`, and `grid_level=1`.
Production calculations should restore a suitably converged basis, grid, and
SCF tolerance.

Constrained FCH/XCH calculations use ordinary DIIS followed, if necessary, by
a short stabilized-DIIS stage within `max_cycles`. The recovery stage
defaults to 0.2 damping and a 0.2 Ha virtual level shift; it restarts from the
lowest-gradient orbitals seen so far while continuing to apply MOM/maxvol.
`diis_cycles`, `mixing_cycles`, `damping`, and `level_shift` expose these
choices. Second-order SCF is disabled for constrained states by
default because its continuous orbital rotations do not reapply the
occupation controller. `second_order=True` is available only as an
explicit diagnostic override.

FCH and XCH state tracking is selected through `occupation`. The recommended
and default `"maxvol"` controller selects the occupied subspace by its
collective overlap determinant from the first occupation call. `"mom"` uses
PySCF MOM, while `"mixed"` optionally uses MOM for `mom_warmup_calls` calls
before switching to maxvol. These alternatives are retained for diagnostics
and state comparisons; they are not the production recommendation. For FCH the
reference is the GS orbitals with the selected core occupation removed; for
XCH it is the converged FCH orbitals with the spectator occupation added.
Each excitation stores its resolved method, warm-up count, and FCH/XCH solver
settings in HDF5.
At normal verbosity, every SCF reports its solver path, cycles, final energy, electron
counts, validation overlaps, and elapsed time. FCH records also report
`<S^2>`, spin contamination, the occupied determinant overlap, and the
minimum target/current occupied-space singular value. Maxvol and mixed calculations
add one aggregate occupation record with their call count, determinant swaps,
occupied-orbital changes, last changing call, and selector time. Detailed
per-call maxvol records remain available at PySCF debug verbosity.
Occupation changes themselves are emitted immediately at PySCF INFO verbosity
so that a live log distinguishes state switching from density-mixing failure.

Logfile paths are resolved relative to `RuntimeConfig.work_directory`. Starting
a new calculation replaces logs with those names; reopening an HDF5
calculation appends restart activity. Reconfiguring PyMBXAS logging preserves
handlers installed by the calling application. Each raw PySCF block begins
with a structured header identifying its GS/FCH/XCH stage and, where relevant,
the atom, spin channel, theory, occupation controller, and device.
High-level console records use compact timestamps and full level names. Log
files additionally include the date and originating module. Scientific
summaries are printed as aligned, tab-indented blocks, with long field values
wrapped beneath their value column.

`energy` is returned in eV. The stored `absorption` arrays are Cartesian
transition amplitudes, not intensities; the spectrum applies the photon-energy
prefactor and isotropic Cartesian average before broadening.

## Restarting from HDF5

```python
from pymbxas import LoggingConfig, PySCFMBXAS, RuntimeConfig

# Loading restores scientific inputs exactly; runtime diagnostics may change.
calc = PySCFMBXAS.load(
    "pymbxas.h5",
    runtime=RuntimeConfig(
        logging=LoggingConfig(pyscf_verbosity=4),
    ),
)
calc.excite("N", occupation="maxvol")
```

The ground-state portion follows PySCF's checkpoint layout, so
`pyscf.scf.chkfile.load_scf` and `mf.from_chk` can read it. Legacy `.pkl`
calculations from version 0.5 and earlier are not supported.

## Experimental many-body satellites

The default spectrum is spin-complete f1: the excited-channel amplitude is
multiplied by the spectator channel's zero-order determinant weight. Request a
single higher total order to add all terms at that order:

```python
energy, corrected = calc.get_mbxas_spectra(
    "O",
    sigma=0.5,
    f_order=2,
)
```

- `f_order=1` returns f1, `f_order=2` returns f1+f2/MB2 including both `20`
  and spectator `11`, and `f_order=3` also adds all f3/MB3 terms.
- The spectator construction follows `f_order` automatically and the sum of
  the two channel extra-pair orders is capped consistently.
- `spectator_order` and `max_total_order` remain advanced diagnostic
  overrides. Explicit `spectator_order=None` omits the opposite-spin factor.
- Physical spectra always include both shake-up and shake-down configurations.
  `get_mbxas_decomposition()` resolves each higher-order contribution into those
  two constituent-level classes under `decomposition["decomposition"]`.

The determinant formulas follow `mbxas-qe`, but PyMBXAS enumerates them exactly
for small validation problems and uses QE-style adaptive `K`-element screening
for production MB2 spectra. Production f2 uses the complete occupied and
virtual FCH manifolds, pruned by the requested spectral energy window rather
than positional orbital cutoffs. Both the excited-channel MB3 `30` and
spectator-double `12` parts of f3 use their corresponding QE adaptive
product-threshold searches. Higher orders remain exact and protected by a
configuration-count guard. See [dev/shakeup.md](dev/shakeup.md).

PyMBXAS applies `sigma` once to each complete final-state transition. QE
broadens both spin factors before convolution; compare Gaussian calculations
using `sigma_PyMBXAS = sqrt(2) * sigma_QE`.

For diagnostics and plotting:

```python
spectra = calc.to_spectra(index=0)
decomposition = spectra.get_mbxas_decomposition(
    f_order=2,
    sigma=0.5,
)

f2_shakeup = decomposition["decomposition"][2]["shakeup"]
f2_shakedown = decomposition["decomposition"][2]["shakedown"]
captured_overlap = decomposition["overlap"]["fraction"]
spectra.print_mbxas_summary(decomposition)

from pymbxas.plotting import plot_mbxas_decomposition
figure, axes = plot_mbxas_decomposition(
    decomposition,
    show_resolved=True,
    show_cumulative=True,
)

# Or calculate and plot one site directly:
figure, axes = spectra.plot_mbxas_decomposition(
    f_order=2,
    sigma=0.5,
    show_resolved=True,
)

# Collections use the same schema. Their default is a mean; use a sum for
# independent absorbing sites.
collection_data = collection.get_mbxas_decomposition(
    f_order=2, sigma=0.5, average=False)
figure, axes = collection.plot_mbxas_decomposition(
    f_order=2, sigma=0.5, average=False, show_resolved=True)
```

The main panel contains the physical total and each f-order contribution.
`show_cumulative=True` adds distinct intermediate cumulative curves, while
`show_resolved=True` adds a panel partitioning every higher order into its
shake-up and shake-down parts. The optional overlap-probability panel remains
enabled by default. Matplotlib is imported only when plotting is requested.

GS-to-FCH orbital rearrangement diagrams are available for each site:

```python
data = spectra.get_orbital_rearrangement(
    energy_window=(-15, 15),  # eV relative to the global GS HOMO
    min_overlap=0.05,
)
figure, axes = spectra.plot_orbital_rearrangement(
    energy_window=(-15, 15), min_overlap=0.05, show_indices=True,
    show_dos=True, dos_sigma=0.25)
```

Alpha and beta appear in separate panels. Occupied levels use dark strokes and
unoccupied levels use lighter strokes; HOMO, LUMO, and the excited
core/core-hole pair are highlighted. In the FCH excited channel, the
constrained core hole is tracked separately and excluded when identifying the
ordinary LUMO. Dashed lines maximize the final squared GS-FCH MO overlap
under a global one-to-one assignment, and their opacity indicates confidence.
They are not a unique orbital identity inside a mixed or degenerate subspace.
Optional Gaussian-broadened orbital-level densities occupy only the unused
outer margins: GS extends left and FCH extends right, with a common scale
within each spin panel.
The default frontier window avoids compressing valence levels against the deep
core; use `include_core=True` or a wider window to include it.

Standalone `Spectra` files written before GS orbital energies were persisted
still load and produce spectra, but cannot make this diagram. Load the original
calculation checkpoint and call `to_spectra()` to recover the energies without
rerunning SCF.

The `examples/6fda_shakeup_compare.py` workflow writes artifacts by type under
`outputs/`: calculation checkpoints, logs, numerical spectrum data, spectrum
figures, and one GS-FCH orbital figure per oxygen site each have dedicated
subdirectories. It explicitly uses `6fda-dam_relaxed_dft.xyz`, tags new
checkpoints with `dftgeom`, defaults to direct maxvol, and disables
second-order recovery. Geometry validation prevents a checkpoint from the
other 6fda structure from being reused accidentally. PBE is the production
default; `--xc b3lyp` runs an otherwise identical B3LYP/def2-SVPD comparison
with independent artifacts.

## Command line

```bash
mbxas structure.xyz --to_excite O --output_file spectrum.h5 \
  --kernel_kwargs '{"xc":"lda","basis":"def2-svpd","gpu":false}'
```

The command writes a `Spectra` or `Spectras` HDF5 file.

## Development documentation

- [AGENTS.md](AGENTS.md): contributor and coding-agent instructions.
- [dev/method.md](dev/method.md): established MBXAS equations, conventions,
  approximations, and reference values.
- [dev/shakeup.md](dev/shakeup.md): exact experimental satellite behavior and
  the comparison with `mbxas-qe`.
- [dev/architecture.md](dev/architecture.md): modules, objects, persistence,
  GPU boundaries, and extension points.
- [CONTRIBUTING.md](CONTRIBUTING.md): testing and changelog workflow.

## References

- Liang et al., *Accurate x-ray spectral predictions: an advanced
  self-consistent-field approach inspired by many-body perturbation theory*,
  PRL 118, 096402 (2017).
- Liang and Prendergast, *Quantum many-body effects in x-ray spectra
  efficiently computed using a basic graph algorithm*, PRB 97, 205127 (2018).
- Liang and Prendergast, *Taming convergence in the determinant approach for
  x-ray excitation spectra*, PRB 100, 075121 (2019).
- Roychoudhury and Prendergast, *Efficient core-excited state orbital
  perspective on calculating x-ray absorption transitions in determinant
  framework*, PRB 107, 035146 (2023).

The `explorer/` Gaussian-process subsystem remains experimental and is not on
the core MBXAS execution path.
