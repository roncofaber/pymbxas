#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:56:01 2023

@author: roncoroni
"""

import numpy as np

#%%

def build_A_K(mb_overlap_channel, occ_idxs_fch, occ_idxs_gs, uno_idxs_fch):
    """Valence overlap determinant and K matrix for one spin channel.

    mb_overlap_channel: (norb_fch, norb_gs) overlap between one channel's
        FCH and GS orbitals, i.e. mb_overlap[channel].
    occ_idxs_fch, uno_idxs_fch: FCH occupied/unoccupied valence orbital
        indices for that channel (core orbital excluded from both).
    occ_idxs_gs: GS occupied valence orbital indices for that channel
        (excited core orbital excluded).

    Returns (AMat, ADet, KMat): AMat is the square valence overlap matrix,
    ADet its determinant, KMat = A'Mat @ inv(AMat) the matrix used both for
    the n=1 amplitude (Eq. 22, PRB 107,035146) and, at higher order, for
    shake-up minors (see mbxas.shakeup).
    """
    AMat = mb_overlap_channel[np.ix_(occ_idxs_fch, occ_idxs_gs)]
    ADet = np.linalg.det(AMat)
    APrimeMat = mb_overlap_channel[np.ix_(uno_idxs_fch, occ_idxs_gs)]
    KMat = APrimeMat @ np.linalg.inv(AMat)
    return AMat, ADet, KMat

# Function to run MBXAS of pyscf calculators
def run_MBXAS_pyscf(mol, gs_calc, fch_calc, gs_orb_idx, channel=1, xch_calc=None):
    try:
        from pyscf.pbc.gto import Cell
    except ImportError:
        Cell = None

    if Cell is not None and isinstance(mol, Cell):
        raise NotImplementedError(
            "MBXAS is not supported under periodic boundary conditions: "
            "the position operator used for the transition dipoles (int1e_r) "
            "is not periodic, so the lattice-summed integrals are not physically meaningful. "
            "Use a molecular or cluster model instead."
        )

    # Calculate dipole integrals and basis set overlap matrix
    dipole = mol.intor('int1e_r')  # shape: (3, nbasis, nbasis)
    basis_overlap = mol.intor("int1e_ovlp")  # shape: (nbasis, nbasis)

    # Calculate MB overlap and dipole matrices for both spin channels.
    # Index 0 = alpha, index 1 = beta.
    # The non-excited channel captures spin reorganization in the core-hole
    # field and is needed for future cross-spin effect calculations.
    mb_overlap = np.array([
        fch_calc.mo_coeff[ch].T @ basis_overlap @ gs_calc.mo_coeff[ch]
        for ch in range(2)
    ])  # shape: (2, norb_fch, norb_gs)

    dipole_KS = np.array([
        fch_calc.mo_coeff[ch].T @ dipole @ fch_calc.mo_coeff[ch]
        for ch in range(2)
    ])  # shape: (2, 3, norb, norb)

    # Index of the excited orbital in FCH calculation
    exc_orb_idx = np.where(fch_calc.mo_occ[channel] == 0)[0][0]

    # Occupied and unoccupied orbital indices for GS and FCH (excited channel)
    occ_idxs_fch = np.where(fch_calc.mo_occ[channel] == 1)[0]
    gs_occ_idxs = np.where(gs_calc.mo_occ[channel] == 1)[0]
    if gs_orb_idx not in gs_occ_idxs:
        raise ValueError(
            f"Orbital index {gs_orb_idx} is not occupied in the ground state "
            f"calculation for channel {channel}. Occupied indices: {gs_occ_idxs}"
        )
    occ_idxs_gs  = np.setdiff1d(gs_occ_idxs, [gs_orb_idx])
    uno_idxs_fch = np.where(fch_calc.mo_occ[channel] == 0)[0][1:]

    # Extract A/K matrices for the excited channel
    AMat, ADet, KMat = build_A_K(mb_overlap[channel], occ_idxs_fch, occ_idxs_gs, uno_idxs_fch)

    # Transition dipole moments from excited orbital (excited channel)
    chb_xmat     = dipole_KS[channel][:, :, exc_orb_idx]
    chb_xmat_occ = chb_xmat[:, occ_idxs_fch]
    chb_xmat_uno = chb_xmat[:, uno_idxs_fch]

    # Calculate absorption spectrum
    absorption = ADet*(chb_xmat_uno - (KMat @ chb_xmat_occ.T).T)

    # Get excitation energies
    energies = fch_calc.mo_energy[channel][uno_idxs_fch]

    # Optional: Correct energies with exchange-correlation calculation
    if xch_calc is not None:
        energies += xch_calc.e_tot - gs_calc.e_tot - np.min(energies)

    # Return results
    return energies, absorption, mb_overlap, dipole_KS, basis_overlap