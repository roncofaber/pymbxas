#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:56:01 2023

@author: roncoroni
"""

import numpy as np

#%%

def occ_unocc_indices(gs_mo_occ_channel, fch_mo_occ_channel, core_orb_idx):
    """Occupied/unoccupied valence orbital indices for one spin channel.

    gs_mo_occ_channel, fch_mo_occ_channel: (norb,) occupation numbers for
        the GS and FCH calculations, one spin channel.
    core_orb_idx: the excited core orbital's GS MO index, excluded from
        the GS occupied set.

    Returns (occ_idxs_gs, occ_idxs_fch, uno_idxs_fch). uno_idxs_fch drops
    the core-hole index (position 0 of the FCH unoccupied set).
    """
    gs_occ_idxs = np.where(gs_mo_occ_channel == 1)[0]
    if core_orb_idx not in gs_occ_idxs:
        raise ValueError(
            f"Orbital index {core_orb_idx} is not occupied in the ground "
            f"state calculation for this channel. Occupied indices: {gs_occ_idxs}"
        )
    occ_idxs_gs  = np.setdiff1d(gs_occ_idxs, [core_orb_idx])
    occ_idxs_fch = np.where(fch_mo_occ_channel == 1)[0]
    uno_idxs_fch = np.where(fch_mo_occ_channel == 0)[0][1:]
    return occ_idxs_gs, occ_idxs_fch, uno_idxs_fch


def spectator_occ_unocc_indices(gs_mo_occ_channel, fch_mo_occ_channel):
    """Occupied/unoccupied valence orbital indices for the spectator
    (non-excited) spin channel's own shake-up (mbxas.shakeup), the
    cross-spin contribution of mbxas-qe's spin_convolve_spectrum.

    Unlike occ_unocc_indices, there is no core orbital to remove and no
    core-hole index to drop from the unoccupied set: this channel keeps
    its full ground-state electron count in the FCH calculation, so its
    valence relaxation is a plain particle-hole excitation, not a
    core-hole one.

    Returns (occ_idxs_gs, occ_idxs_fch, uno_idxs_fch).
    """
    occ_idxs_gs  = np.where(gs_mo_occ_channel == 1)[0]
    occ_idxs_fch = np.where(fch_mo_occ_channel == 1)[0]
    if len(occ_idxs_gs) != len(occ_idxs_fch):
        raise ValueError(
            "Spectator channel electron count changed between GS and FCH "
            f"({len(occ_idxs_gs)} -> {len(occ_idxs_fch)}); this channel "
            "should be the one without a core hole. Pass the excited "
            "channel to occ_unocc_indices instead."
        )
    uno_idxs_fch = np.where(fch_mo_occ_channel == 0)[0]
    return occ_idxs_gs, occ_idxs_fch, uno_idxs_fch


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
    occ_idxs_gs, occ_idxs_fch, uno_idxs_fch = occ_unocc_indices(
        gs_calc.mo_occ[channel], fch_calc.mo_occ[channel], gs_orb_idx)

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