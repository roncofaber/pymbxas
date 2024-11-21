"""
*****************************************************
| PyMBXAS: Python-based MBXAS implementation        |
*****************************************************

PyMBXAS is a package for setting up, manipulating, running and visualizing MBXAS calculations using Python. It has an object-oriented approach to simplify the task of spectra analysis and post-processing.
PyMBXAS leverages the PySCF  electronic structure code and the Atomic Simulation Environment (ASE).

The current implementation is being developed on:
    https://gitlab.com/roncofaber/pymbxas

The initial MBXAS work is based on:
    https://github.com/subhayanroychoudhury/CleaRIXS
    
If you want more information, please look at the following publications:
    
    1. https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.115115
    2. https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.035146

Example script:
    
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

    # plot result
    E, I = obj.get_mbxas_spectra(to_excite)#, erange=[520, 540], sigma=0.006)

    plt.figure()
    plt.plot(E,I)
    plt.show()

"""

# import os
import sys

if sys.version_info[0] == 2:
    raise ImportError('Please run with Python3. This is Python2.')

# package info
__version__ = '0.5.0'
__date__ = "11 Nov. 2024"
__author__ = "Fabrice Roncoroni"
__all__ = ["spectra", "spectras"]

from pymbxas.spectra  import Spectra
from pymbxas.spectras import Spectras
