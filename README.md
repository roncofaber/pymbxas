<div align="center">
  <img src="https://gitlab.com/uploads/-/system/project/avatar/47099716/pymbxas2_1_.png" height="120px"/>
</div>

PyMBXAS: Python-based MBXAS implementation
-----------------------------------------------

PyMBXAS is a package for setting up, manipulating, running and visualizing MBXAS calculations using Python. It has an object-oriented approach to simplify the task of spectra analysis and post-processing.
PyMBXAS leverages the [PySCF  electronic structure code](https://github.com/pyscf/pyscf) and the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/).

### Requirements
You need to have both PySCF and ASE installed in your Python environment. If you want to use GPU capabilities make sure to install [gpu4pyscf](https://github.com/pyscf/gpu4pyscf).

### Installation
You can install the package by running:
```
pip install git+https://gitlab.com/roncofaber/pymbxas.git
```
or you can clone the repo and add it to your `PYTHONPATH`.

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
Output:
```
[10:52:59] |----------------------------------|
           |                                  |
           |>>>>>>   Starting PyMBXAS   <<<<<<|
           |                                  |
           |       ver  0.2.0a | 04 Apr. 2024 |
           |----------------------------------|
        
[10:52:59] |Started a new GS calculation
[10:53:03] |IBO localization : [3, 4, 2, 0, 1]
[10:53:03] |GS finished in 4.1 s.

[10:53:03] |-----> Exciting N  atom # 3 <-----|
[10:53:03] |>>> Started FCH calculation.
[10:53:08] |>>> FCH finished in 4.7 s.
[10:53:08] |>>> Started XCH calculation.
[10:53:12] |>>> XCH finished in 4.5 s.
[10:53:12] |>>> MBXAS finished in 0.0 s [✓].
[10:53:12] |----- Excitation successful! -----|

[10:53:12] |Saved everything as pyscf_obj.pkl
[10:53:12] |PyMBXAS finished successfully!
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