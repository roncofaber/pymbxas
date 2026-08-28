#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:05:13 2023

@author: roncoroni
"""

import logging

# pyscf stuff
from pyscf import scf, dft
import pyscf.pbc as pypbc


logger = logging.getLogger(__name__)

#%%

def make_density_fitter(mol, pbc, cderi=False):
    
    if pbc:
        df_obj = pypbc.df.DF(mol)
        
        # df_obj.auxbasis = pypbc.df.make_auxbasis(mol)
        
        if cderi:
            df_obj._cderi_to_save = 'saved_cderi.h5'
    else:
        raise NotImplementedError("Density fitting for non PBC systems has yet to be implemented!")
    
    df_obj.build()
    
    return df_obj

# Function to make a pyscf calculator that both work with PBC and not.
def make_pyscf_calculator(mol, xc=None, calc_type="UKS", pbc=False, solvent=None,
                          dens_fit=None, calc_name=None, save=False, is_gpu=False,
                          max_cycle=100, conv_tol=None, grid_level=None):

    # with PBC
    if pbc:
        calc = getattr(pypbc.dft, calc_type)(mol, xc=xc).density_fit()
        
        # density fit object is string
        if dens_fit is not None:
            calc.with_df = dens_fit
    
    # no PBC
    else:
        
        # generate calculator
        if "HF" in calc_type.upper():
            if xc is not None:
                logger.warning("HF calculator ignores the XC setting %r", xc)
            calc = getattr(scf, calc_type)(mol)
        elif "KS" in calc_type.upper():
            if xc is None:
                xc = "LDA" #default to LDA
            calc = getattr(dft, calc_type)(mol, xc=xc)
        else:
            raise ValueError("calc_type '{}' must contain 'HF' or 'KS'. Supported types: 'RHF', 'UHF', 'RKS', 'UKS'".format(calc_type))

    # add solvent treatment
    if solvent is not None:
        calc = calc.DDCOSMO()
        calc.with_solvent.eps = solvent

    # Use chkfile to store calculation (if you want)
    if isinstance(calc_name, str) and save:
       calc.chkfile = '{}.chkfile'.format(calc_name)

    calc.max_cycle = int(max_cycle)
    if conv_tol is not None:
        calc.conv_tol = float(conv_tol)
    if grid_level is not None and hasattr(calc, "grids"):
        calc.grids.level = int(grid_level)
    
    # add GPU compatibility
    if is_gpu:
        try:
            calc = calc.to_gpu()
        except ImportError:
            raise RuntimeError("CuPy is required for GPU calculations. Please install it.") 
            
    return calc
