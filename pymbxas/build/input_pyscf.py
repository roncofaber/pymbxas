#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:05:13 2023

@author: roncoroni
"""

# pyscf stuff
from pyscf import gto, scf, dft
import pyscf.pbc as pypbc
from pyscf.scf.addons import mom_occ

# pymbxas modules
from pymbxas.build.structure import ase_to_mole

#%%

def make_density_fitter(mol, pbc, cderi=False):
    
    if pbc:
        df_obj = pypbc.df.DF(mol)
        
        # df_obj.auxbasis = pypbc.df.make_auxbasis(mol)
        
        if cderi is not None:
            df_obj._cderi_to_save = 'saved_cderi.h5'
    else:
        raise "Density fitting for non PBC systems has yet to be implemented!"
    
    df_obj.build()
    
    return df_obj

# Function to make a pyscf calculator that both work with PBC and not.
def make_pyscf_calculator(mol, xc, is_pbc, dens_fit, calc_name=None,
                          save=False):
    
    if is_pbc:
        calc = pypbc.dft.UKS(mol, xc=xc).density_fit()
        
        # density fit object is string
        if dens_fit is not None:
            # calc.with_df._cderi = 'saved_cderi.h5'
            calc.with_df = dens_fit
            
    else:
        # generate KS calculator
        calc = dft.UKS(mol, xc=xc)

    
    # Use chkfile to store calculation (if you want)
    if isinstance(calc_name, str) and save:
       calc.chkfile = '{}.chkfile'.format(calc_name)
    
    
    return calc