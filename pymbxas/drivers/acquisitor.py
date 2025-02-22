#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:50:13 2025

@author: roncofaber
"""

from pymbxas.calculators.pyscf import PySCF_mbxas

#%%

def pyscf_acquire(structure, to_excite, **kwargs):
    """Performs a PySCF calculation and returns a Spectra object.

    Args:
        structure: The ASE structure.
        to_excite: Atom index(es)/symbol(s) to excite.
        **kwargs: Additional keyword arguments for PySCF_mbxas.

    Returns:
        Spectra object or None if calculation fails.
    """
    defaults = {
        "charge": 0,
        "spin": 0,
        "basis": "def2-svpd",
        "xc": "b3lyp",
        "pbc": False,
        "solvent": None,
        "calc_type": "UKS",
        "do_xch": True,
        "loc_type": "ibo",
        "pkl_file": None,
        "target_dir": None,
        "xas_verbose": 3,
        "xas_logfile": "pymbxas.log",
        "dft_verbose": 4,
        "dft_logfile": "pyscf.log",
        "dft_output": False,
        "print_fchk": False,
        "save": False,
        "save_chk": False,
        "save_name": "pyscf_obj.pkl",
        "save_path": None,
        "gpu": True,
    }
    params = defaults.copy()
    params.update(kwargs) # Update defaults with user-provided kwargs

    try:
        obj = PySCF_mbxas(structure=structure, **params)
        obj.kernel(to_excite)
        return obj.to_spectra()
    except Exception as e:
        print(f"Error during PySCF calculation: {e}")
        return None
