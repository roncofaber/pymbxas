## PyMBXAS

PyMBXAS is a set of tools and Python modules for setting up, manipulating,
running, visualizing and analyzing CleaRIXS for Q-chem.

### Requirements
You need both Q-Chem and CleaRIXS installed on your device.

Find the newest CleaRIXS release:
>https://github.com/subhayanroychoudhury/CleaRIXS

### Usage
Setting up a full calculation is as easy as doing:
```python
import ase
import ase.build
from pymbxas.calculators.qchem import Qchem_mbxas

# build molecule
molecule = ase.build.molecule("C6H6")

# set up parameters
charge       = 0
multiplicity = 1
qchem_params = {
    "jobtype"      : "sp",
    "exchange"     : "B3LYP",
    "basis"        : "def2-tzvpd",
}

# run all calculations
obj = Qchem_mbxas(
    molecule,
    charge,
    multiplicity,
    qchem_params = qchem_params,
    excitation   = 0, #excite 0th atom --> Boys automatically finds the 1s orbital
    )
```

### Roadmap
Expand the method to interface and help manage multiple DFT codes, expand spectra visualization and analysis capabilities, ... (pretty much everything is WIP at the moment).