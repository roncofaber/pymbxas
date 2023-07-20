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
import matplotlib.pyplot as plt
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
    excitation   = 0, #excite 0th atom --> code automatically finds the relevant 1s orbital
    )

# Obtained broadened MBXAS spectra
X, Y = obj.get_mbxas_spectra()

# Plot spectra
plt.figure()
plt.plot(X, Y)
plt.show()

```
You can easily reload a saved ".pkl" file as follow:
```python
from pymbxas.calculators.qchem import Qchem_mbxas

# load MBXAS object
obj = Qchem_mbxas(pkl_file="mbxas_obj.pkl") #change with actual name of file

# Obtained broadened MBXAS spectra
X, Y = obj.get_mbxas_spectra()

# Plot spectra
plt.figure()
plt.plot(X, Y)
plt.show()

```

### Roadmap
Expand the method to interface and help manage multiple DFT codes, expand spectra visualization and analysis capabilities, ... (pretty much everything is WIP at the moment).