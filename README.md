<div align="left">
  <img src="https://gitlab.com/uploads/-/system/project/avatar/47099716/pymbxas2_1_.png" height="80px"/>
</div>PyMBXAS: Python-based MBXAS implementation
-----------------------------------------------

PyMBXAS is a package for setting up, manipulating,
running and visualizing MBXAS calculations using Python. It has an object-oriented approach to simplify the task of spectra analysis and post-processing.
PyMBXAS leverages the [PySCF  electronic structure code](https://github.com/pyscf/pyscf) and the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/).

### Requirements
You need to have both PySCF and ASE installed in your Python environment. If you want to use GPU capabilities make sure to install [gpu4pyscf](https://github.com/pyscf/gpu4pyscf).

### Usage
To run a MBXAS calculation, you just need to set up the PySCF_mbxas object:

```python
import ase 
import ase.io
from pymbxas.calculators.pyscf import PySCF_mbxas

# read structure in ASE format
structure = ase.io.read("glycine.xyz")

# set up calculation parameters
channel   = 1 
charge    = 0 # charge of system
spin      = 0 # spin of system
to_excite = "N" # index(es)/symbols of atom(s) to excite
basis     = "def2-svpd"
xc        = "lda"#"b3lyp"
pbc       = False

# set up object
obj = PySCF_mbxas(
    structure    = structure,
    charge       = charge,
    spin         = spin,
    xc           = xc, 
    basis        = basis,
    do_xch       = True, # do XCH for energy alignment
    target_dir   = None, # specify a target directory where to run calc
    verbose      = 3,     # level of verbose of pySCF output (4-5 is good)
    print_fchk   = False,  # print fchk files (requires MOKIT)
    print_output = False, # print output to console
    save         = True,  # save object as pkl file
    save_name    = "pyscf_obj.pkl", # name of saved file
    save_path    = None, # path of saved object
    loc_type     = "ibo",
    gpu          = True # if you want to use the GPU code
    )

# run calculation (GS + FCH + XCH)
obj.kernel(to_excite)
```

The spectra can then be obtained with:

```python
import matplotlib.pyplot as plt

E, I = obj.get_mbxas_spectra(to_excite)#, erange=[395, 430], sigma=0.006)

plt.figure()
plt.plot(E,I)
plt.show()
```

Calculations can be stored as pkl files using the dill package. You can then simply reload a calculation doing:

```python
obj = PySCF_mbxas(pkl_file="pyscf_obj.pkl")

```

### Roadmap
Implement Machine Learning of spectral features (WIP). In the future, expand the method to interface and help manage multiple DFT codes, expand spectra visualization and analysis capabilities, ... (pretty much everything is WIP at the moment).