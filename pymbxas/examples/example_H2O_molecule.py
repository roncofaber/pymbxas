#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:42:28 2024

@author: roncofaber
"""

import ase 
import ase.build
from pymbxas.calculators.pyscf import PySCF_mbxas
import matplotlib.pyplot as plt

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

# make spectra obj
sp = obj.to_spectra(0)


# get mbxas directly from obj
E1, I1 = obj.get_mbxas_spectra(to_excite, erange=[520, 640], sigma=0.5)

# get mbxas from spectra
E2, I2 = sp.get_mbxas_spectra(erange=[520, 640])

plt.figure()
plt.plot(E1, I1)
plt.plot(E2, I2)
plt.show()
