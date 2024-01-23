#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of functions to work with Boys and give us amazing orbitals.

Created on Thu Jul 13 15:55:09 2023

@author: roncoroni
"""

from pyscf import lo
#%%

def do_localization_pyscf(dft_calc, deg_orbitals, loc_type):
    if "boys" in loc_type:
        mo_loc = do_boys_pyscf(dft_calc, deg_orbitals)
    elif "ibo" in loc_type:
        mo_loc = do_ibo_pyscf(dft_calc, deg_orbitals)
        
    return mo_loc
        
def do_boys_pyscf(dft_calc, deg_orbitals):

    mo_boys = dft_calc.mo_coeff.copy()
    for ii in [0,1]:

        # make loc object
        loc = lo.Boys(dft_calc.mol)

        # run boys on MO from calculation
        loc.run(mo_coeff=dft_calc.mo_coeff[ii][:, deg_orbitals[ii]])

        # reupdate old orbitals with boys
        mo_boys[ii][:, deg_orbitals[ii]] = loc.mo_coeff

    return mo_boys

def do_ibo_pyscf(dft_calc, deg_orbitals):

    mo_ibo = dft_calc.mo_coeff.copy()
    for ii in [0, 1]:
        pm = lo.ibo.ibo(dft_calc.mol, dft_calc.mo_coeff[ii][:, deg_orbitals[ii]])

        mo_ibo[ii][:, deg_orbitals[ii]] = pm
    
    return mo_ibo