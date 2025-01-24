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
