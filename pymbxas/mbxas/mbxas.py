#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:56:01 2023

@author: roncoroni
"""

import numpy as np

#%%

# Function to run MBXAS of pyscf calculators
def run_MBXAS_pyscf(gs_calc, fch_calc, channel=1, xch_calc=None):

    # calculate dipole
    dipole = gs_calc.mol.intor('int1e_r')

    # calculate FCH dipole matrix
    dipole_KS = fch_calc.mo_coeff[channel].T@dipole@fch_calc.mo_coeff[channel]

    # calculate basis overlap
    basis_overlap = gs_calc.mol.intor("int1e_ovlp")

    # calculate many body overlap
    mb_overlap = fch_calc.mo_coeff[channel].T@basis_overlap@gs_calc.mo_coeff[channel]

    # index of excited oribtal (first 0 element)
    exc_orb_idx = np.where(fch_calc.mo_occ[channel] == 0)[0][0]

    # define indexes of occupied and unoccupied orbitals
    occ_idxs = np.where(fch_calc.mo_occ[channel] == 1)[0]
    uno_idxs =np.where(fch_calc.mo_occ[channel] == 0)[0][1:] # remove first unocc as it's the CH

    # extract from the MB matrix only what is occupied
    AMat = mb_overlap[np.ix_(occ_idxs, occ_idxs)] # np.ix_ works with rows and cols

    # calculate determinant of AMat
    ADet = np.linalg.det(AMat)

    # extract A': size n_unocc, n_occ
    APrimeMat = mb_overlap[np.ix_(uno_idxs, occ_idxs)]

    # calculate KMat from A' and A
    KMat = APrimeMat @ np.linalg.inv(AMat)

    # get transition contribution from the excited orbital
    chb_xmat = dipole_KS[:, :, exc_orb_idx]
    chb_xmat_occ = chb_xmat[:,occ_idxs]
    chb_xmat_uno = chb_xmat[:,uno_idxs]

    # calculate absorption (FCH contribution - CH contribution)
    absorption = ADet*(chb_xmat_uno - (KMat @ chb_xmat_occ.T).T)

    # get energies
    energies = fch_calc.mo_energy[channel][uno_idxs]

    if xch_calc is not None:
        energies += xch_calc.e_tot - gs_calc.e_tot - np.min(energies)

    return energies, absorption, mb_overlap, dipole_KS, basis_overlap