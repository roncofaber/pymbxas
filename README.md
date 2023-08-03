## PyMBXAS

PyMBXAS is a set of tools and Python modules for setting up, manipulating,
running, visualizing and analyzing CleaRIXS for Q-chem.

### Requirements
You need both Q-Chem and CleaRIXS installed on your device.

Find the newest CleaRIXS release:
>https://github.com/subhayanroychoudhury/CleaRIXS

### Usage
Setting up a full calculation is as easy very easy. For Qchem do:
```python
import ase
import ase.build
import matplotlib.pyplot as plt
from pymbxas.calculators.qchem import Qchem_mbxas

# build molecule
molecule = ase.build.molecule("H2O")

# set up parameters
charge       = 0
multiplicity = 1
qchem_params = {
    "jobtype"      : "sp",
    "exchange"     : "B3LYP",
    "basis"        : "def2-tzvpd",
    "unrestricted" : True,
    "symmetry"     : False,
    "sym_ignore"   : True,
}

# run all calculations
obj = Qchem_mbxas(
    molecule,
    charge,
    multiplicity,
    qchem_params = qchem_params,
    excitation   = 0, #excite 0th atom --> code automatically finds the relevant 1s orbital
    target_dir   = "water", # specify name of target directory with save files
    )

# Obtained broadened MBXAS spectra
X, Y = obj.get_mbxas_spectra()

# Plot spectra
plt.figure()
plt.plot(X, Y)
plt.show()

```

For PySCF do this instead (WIP: merge both to have same data structure)
```python

import pymbxas
from pymbxas.calculators.pyscf import PySCF_mbxas

import ase
import ase.build
import ase.data.pubchem
#%%

# structure = ase.build.molecule("C6H6")
# structure = ase.build.molecule("H2O")
structure = ase.data.pubchem.pubchem_atoms_search(smiles="CCOC(=O)C(F)(F)F")

charge    = 0 # charge of system
spin      = 0 # spin of system
to_excite = 0 # index of atom to excite

obj = PySCF_mbxas(
    structure    = structure,
    charge       = charge,
    spin         = spin,
    excitation   = to_excite,
    xc           = "b3lyp",
    basis        = "def2-svpd",
    do_xch       = True, # do XCH for energy alignment
    target_dir   = structure.get_chemical_formula(), # specify a target directory where to run calc
    verbose      = 5,     # level of verbose of pySCF output (4-5 is good)
    print_fchk   = True,  # print fchk files (requires MOKIT)
    print_output = False, # print output to console
    run_calc     = True,  # run all calculations directly
    save         = True,  # save object as pkl file
    save_all     = False, # save all checkfile
    save_name    = "pyscf_obj.pkl", # name of saved file
    save_path    = None, # path of saved object
    )


#%%

E, I = obj.get_mbxas_spectra(sigma=0.03)


#%%

import matplotlib.pyplot as plt

plt.figure()
plt.plot(E, I)
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