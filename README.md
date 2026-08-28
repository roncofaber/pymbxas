# PyMBXAS

[![PyPI version](https://badge.fury.io/py/pymbxas.svg)](https://badge.fury.io/py/pymbxas)

PyMBXAS calculates molecular K-edge X-ray absorption spectra with PySCF. It combines unrestricted GS, FCH, and optional XCH calculations with determinant-based MBXAS transition amplitudes.

Current scope:

- all-electron molecular UKS and UHF calculations;
- IBO or Boys localization of degenerate core orbitals;
- MOM or maxvol constrained-state tracking, with maxvol as the default;
- HDF5 checkpoints that can be restarted and extended;
- experimental f2/f3 shake-up, shake-down, and spectator contributions;
- optional GPU execution through GPU4PySCF.

Periodic MBXAS is not supported because the position operator used for transition dipoles is not periodic.

## Installation

```bash
pip install pymbxas
```

Optional features:

```bash
pip install "pymbxas[plot]"
pip install "pymbxas[gpu]"
```

Install the development branch with:

```bash
pip install "git+https://github.com/roncofaber/pymbxas.git@dev"
```

## Quick start

```python
import ase.build

from pymbxas import CalculationConfig, PySCFMBXAS

structure = ase.build.molecule("H2O")
theory = CalculationConfig(xc="pbe", basis="def2-svpd")

calc = PySCFMBXAS(
    structure,
    calculation=theory,
    checkpoint="water.h5",
)
calc.run("O")

energy, intensity = calc.get_mbxas_spectra("O", sigma=0.5)
```

`run()` calculates the ground state if needed and then runs the requested excitations. Use `run_gs()` for only the ground state or `excite()` to add excitations to an existing ground state.

## Configuration

Configuration is separated by responsibility instead of nested at the script level:

| Class | Responsibility |
|---|---|
| `CalculationConfig` | Electronic model shared by GS, FCH, and XCH |
| `SCFConfig` | Solver controls supplied as `gs_scf`, `fch_scf`, or `xch_scf` |
| `ExcitationConfig` | Spin channel, XCH alignment, and occupation tracking |
| `RuntimeConfig` | Device and working directory for one execution |
| `LoggingConfig` | PyMBXAS and raw PySCF logging for one execution |
| `CheckpointConfig` | Advanced checkpoint and PySCF artifact policy |

```python
from pymbxas import (
    ExcitationConfig,
    LoggingConfig,
    RuntimeConfig,
    SCFConfig,
)

gs_scf = SCFConfig(max_cycles=120)
fch_scf = gs_scf.copy().set(mixing_cycles=30)
xch_scf = fch_scf.copy()

calc = PySCFMBXAS(
    structure,
    calculation=theory,
    gs_scf=gs_scf,
    checkpoint="calculation.h5",
)
calc.run(
    "O",
    excitation=ExcitationConfig(occupation="maxvol"),
    fch_scf=fch_scf,
    xch_scf=xch_scf,
    runtime=RuntimeConfig(device="gpu", work_directory="outputs"),
    logging=LoggingConfig(pyscf_logfile="pyscf.log"),
)
```

Configurations support validated `.set()` updates and independent `.copy()` variants. A calculator snapshots its inputs, so changing a template later does not alter an existing calculation.

## Restart and analysis

```python
calc = PySCFMBXAS.load("calculation.h5")
calc.excite("N")

spectra = calc.to_spectra(index=0)
figure, axes = spectra.plot_mbxas_decomposition(
    f_order=2,
    sigma=0.5,
    erange=(525, 555),
    show_resolved=True,
)
```

Calculation checkpoints store the electronic model and the exact GS, FCH, and XCH solver configurations. Pass `checkpoint=None` to disable automatic checkpointing.

## Documentation

- [Method, conventions, and approximations](dev/method.md)
- [Shake-up, shake-down, and comparison with `mbxas-qe`](dev/shakeup.md)
- [Architecture, configuration lifecycle, GPU path, and HDF5 schema](dev/architecture.md)
- [6fda production example](examples/6fda_shakeup_compare.py)
- [Contributing](CONTRIBUTING.md)

## Command line

```bash
mbxas structure.xyz --sites O -o spectrum.h5 \
  --calculation-config '{"xc":"pbe","basis":"def2-svpd"}' \
  --checkpoint calculation.h5
```

Run `mbxas --help` for the separate runtime, logging, excitation, and GS/FCH/XCH SCF options.

## References

The method follows Liang and Prendergast, PRL 118, 096402 (2017); PRB 97, 205127 (2018); PRB 100, 075121 (2019); and Roychoudhury and Prendergast, PRB 107, 035146 (2023).
