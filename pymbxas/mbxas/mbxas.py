#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:56:01 2023

@author: roncoroni
"""

import numpy as np

#%%

# Function to run MBXAS of pyscf calculators
def run_MBXAS_pyscf(mol, gs_calc, fch_calc, gs_orb_idx, channel=1, xch_calc=None):

    # calculate dipole and basis overlap
    dipole = mol.intor('int1e_r')
    basis_overlap = mol.intor("int1e_ovlp")
    
    # calculate FCH dipole matrix
    dipole_KS = fch_calc.mo_coeff[channel].T@dipole@fch_calc.mo_coeff[channel]
    
    # calculate many body overlap
    mb_overlap = fch_calc.mo_coeff[channel].T@basis_overlap@gs_calc.mo_coeff[channel]
    
    # index of excited oribtal (first 0 element)
    exc_orb_idx = np.where(fch_calc.mo_occ[channel] == 0)[0][0]
    
    # define indexes of occupied and unoccupied orbitals for GS and FCH
    occ_idxs_fch = np.where(fch_calc.mo_occ[channel] == 1)[0]
    occ_idxs_gs  = np.delete(np.where(gs_calc.mo_occ[channel] == 1)[0], gs_orb_idx)
    uno_idxs_fch = np.where(fch_calc.mo_occ[channel] == 0)[0][1:] # remove first unocc as it's the CH
    
    # extract from the MB matrix only what is occupied
    AMat = mb_overlap[np.ix_(occ_idxs_fch, occ_idxs_gs)] # np.ix_ works with rows and cols
    
    # calculate determinant of AMat
    ADet = np.linalg.det(AMat)
    
    # extract A': size n_unocc, n_occ
    APrimeMat = mb_overlap[np.ix_(uno_idxs_fch, occ_idxs_gs)]
    
    # calculate KMat from A' and A
    KMat = APrimeMat @ np.linalg.inv(AMat)
    
    # get transition contribution from the excited orbital
    chb_xmat = dipole_KS[:, :, exc_orb_idx]
    chb_xmat_occ = chb_xmat[:,occ_idxs_fch]
    chb_xmat_uno = chb_xmat[:,uno_idxs_fch]
    
    # calculate absorption (FCH contribution - CH contribution)
    absorption = ADet*(chb_xmat_uno - (KMat @ chb_xmat_occ.T).T)
    
    # get energies
    energies = fch_calc.mo_energy[channel][uno_idxs_fch]

    if xch_calc is not None:
        energies += xch_calc.e_tot - gs_calc.e_tot - np.min(energies)

    return energies, absorption, mb_overlap, dipole_KS, basis_overlap