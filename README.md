<div align="center">
  <img src="https://gitlab.com/uploads/-/system/project/avatar/47099716/pymbxas2_1_.png" height="120px"/>
</div>

PyMBXAS: Python-based many-body XAS implementation
-----------------------------------------------
[![PyPI version](https://badge.fury.io/py/pymbxas.svg)](https://badge.fury.io/py/pymbxas)

PyMBXAS is a package for setting up, manipulating, running and visualizing Many-Body X-ray Adsorption Spectroscopy (MBXAS) calculations using Python. It has an object-oriented approach to simplify the task of spectra analysis and post-processing.
PyMBXAS leverages the [PySCF  electronic structure code](https://github.com/pyscf/pyscf) and the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/).

### Requirements
You need to have both PySCF and ASE installed in your Python environment. If you want to use GPU capabilities make sure to install [gpu4pyscf](https://github.com/pyscf/gpu4pyscf).

### Installation
You can install the latest stable release of the package by running:
```
pip install pymbxas
```
or if you want the most up to date version:
```
pip install git+https://gitlab.com/roncofaber/pymbxas.git
```
Alternatively, you can clone the repo and add it to your `PYTHONPATH`.

### Usage
To run a MBXAS calculation, you just need to set up the PySCF_mbxas object:

```python
import ase 
import ase.build
from pymbxas.calculators.pyscf import PySCF_mbxas

# read structure in ASE format
structure = ase.build.molecule("H2O")

# set up calculation parameters
channel   = 1 
charge    = 0 # charge of system
spin      = 0 # spin of system
to_excite = "O" # index(es)/symbols of atom(s) to excite
basis     = "def2-svpd"
xc        = "lda"#"b3lyp"
pbc       = False

# set up object (showing all extra arguments, many are optional)
obj = PySCF_mbxas(
    structure    = structure,
    charge       = charge,
    spin         = spin,
    xc           = xc,
    basis        = basis,
    pbc          = pbc, # enable PBC or not
    solvent      = None,  # specify solvent epsilon
    calc_type    = "UKS", # UKS or UHF
    do_xch       = True,  # do XCH to align energy
    loc_type     = "ibo", # localization routine
    
    pkl_file     = None, # reload previous calculation from pkl
    target_dir   = None, # run the calculation in a target dir
    
    xas_verbose  = 3,    # verbose level of pymbxas
    xas_logfile  = "pymbxas.log", # file for mbxas log
    dft_verbose  = 6,    # verbose level of pyscf
    dft_logfile  = "pyscf.log", # file for pyscf log
    dft_output   = False, # print pyscf output or not on terminal
    
    print_fchk   = False, # print FCHK files as calculation goes
    
    save         = True,  # save object as pkl file
    save_chk     = False, # save calculation as chkfile
    save_name    = "pyscf_obj.pkl", # name of saved file
    save_path    = None, # path of saved object
    gpu          = False,
    )

# run calculation (GS + FCH + XCH)
obj.kernel(to_excite)

# run calculation (GS + FCH + XCH)
obj.kernel(to_excite)
```
Output:
```
[16:01:53] |(I) 
           |----------------------------------|
           |                                  |
           |>>>>>>   Starting PyMBXAS   <<<<<<|
           |                                  |
           |       ver   0.5.0 | 11 Nov. 2024 |
           |----------------------------------|
        
[16:01:53] |(I) Started a new GS calculation
[16:01:56] |(I) GS finished in 2.9 s.

[16:01:56] |(I) -----> Exciting O  atom # 0 <-----|
[16:01:56] |(I) >>> Started FCH calculation.
[16:01:59] |(I) >>>>> FCH finished in 3.0 s.
[16:01:59] |(I) >>> Started XCH calculation.
[16:02:02] |(I) >>>>> XCH finished in 2.4 s.
[16:02:02] |(I) >>> Started MBXAS calculation.
[16:02:02] |(I) >>>>> MBXAS finished in 0.0 s [✓].
[16:02:02] |(I) ----- Excitation successful! -----|

[16:02:02] |(I) Saved everything as pyscf_obj.pkl
[16:02:02] |(I) PyMBXAS finished successfully!
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
Implement Machine Learning of spectral features. The `mbxasplorer` class implements Gaussian Process Regression to predict XAS spectra of molecules, but is still WIP. In the future, expand the method to interface and help manage multiple DFT codes, expand spectra visualization and analysis capabilities, ...
