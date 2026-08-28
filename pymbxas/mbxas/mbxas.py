#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:56:01 2023

@author: roncoroni
"""

import numpy as np

#%%

def core_hole_index(mb_overlap_channel, fch_mo_occ_channel, core_orb_idx):
    """Locate the FCH core hole by overlap with the selected GS core MO.

    MOM occupations need not be Aufbau ordered, so the core hole is not
    necessarily the first zero in ``fch_mo_occ_channel``.  Rows of
    ``mb_overlap_channel`` are FCH MOs and columns are GS MOs.
    """
    unoccupied = np.flatnonzero(np.asarray(fch_mo_occ_channel) == 0)
    if unoccupied.size == 0:
        raise ValueError("The FCH excited channel has no unoccupied orbitals")
    if not 0 <= int(core_orb_idx) < mb_overlap_channel.shape[1]:
        raise ValueError(
            f"Ground-state core orbital index {core_orb_idx} is outside the "
            f"overlap matrix with {mb_overlap_channel.shape[1]} columns")
    overlaps = np.abs(np.asarray(mb_overlap_channel)[unoccupied, int(core_orb_idx)])
    return int(unoccupied[np.argmax(overlaps)])


def occ_unocc_indices(gs_mo_occ_channel, fch_mo_occ_channel, core_orb_idx,
                      core_hole_idx=None):
    """Occupied/unoccupied valence orbital indices for one spin channel.

    gs_mo_occ_channel, fch_mo_occ_channel: (norb,) occupation numbers for
        the GS and FCH calculations, one spin channel.
    core_orb_idx: the excited core orbital's GS MO index, excluded from
        the GS occupied set.

    ``core_hole_idx`` is the FCH MO index identified by orbital overlap.  It
    should be supplied for non-Aufbau states; the legacy lowest-index fallback
    is retained for callers that only have occupations.

    Returns (occ_idxs_gs, occ_idxs_fch, uno_idxs_fch). ``uno_idxs_fch`` drops
    the identified core-hole MO while preserving every ordinary virtual.
    """
    gs_occ_idxs = np.where(gs_mo_occ_channel == 1)[0]
    if core_orb_idx not in gs_occ_idxs:
        raise ValueError(
            f"Orbital index {core_orb_idx} is not occupied in the ground "
            f"state calculation for this channel. Occupied indices: {gs_occ_idxs}"
        )
    occ_idxs_gs  = np.setdiff1d(gs_occ_idxs, [core_orb_idx])
    occ_idxs_fch = np.where(fch_mo_occ_channel == 1)[0]
    unoccupied = np.where(fch_mo_occ_channel == 0)[0]
    if unoccupied.size == 0:
        raise ValueError("The FCH excited channel has no unoccupied orbitals")
    if core_hole_idx is None:
        core_hole_idx = int(unoccupied[0])
    if core_hole_idx not in unoccupied:
        raise ValueError(
            f"FCH core-hole index {core_hole_idx} is not unoccupied; "
            f"unoccupied indices: {unoccupied}")
    uno_idxs_fch = unoccupied[unoccupied != core_hole_idx]
    return occ_idxs_gs, occ_idxs_fch, uno_idxs_fch


def spectator_occ_unocc_indices(gs_mo_occ_channel, fch_mo_occ_channel):
    """Occupied/unoccupied valence orbital indices for the spectator
    (non-excited) spin channel's own overlap-satellite kernel. It supplies
    the spectator factor in PyMBXAS's cross-spin convolution; see
    dev/shakeup.md for how that differs from mbxas-qe's full construction.

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

    Returns (AMat, ADet, KMat, APrimeMat): AMat is the square valence overlap
    matrix, ADet its determinant, KMat = A'Mat @ inv(AMat) the matrix used
    both for the n=1 amplitude (Eq. 22, PRB 107,035146) and for the exact
    higher-order determinant expansions (see mbxas.shakeup). APrimeMat is
    also returned for determinant-overlap spectra.
    """
    AMat = mb_overlap_channel[np.ix_(occ_idxs_fch, occ_idxs_gs)]
    ADet = np.linalg.det(AMat)
    APrimeMat = mb_overlap_channel[np.ix_(uno_idxs_fch, occ_idxs_gs)]
    KMat = APrimeMat @ np.linalg.inv(AMat)
    return AMat, ADet, KMat, APrimeMat

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

    # Locate the core hole by its overlap with the selected GS core orbital;
    # MOM states are not guaranteed to have Aufbau-ordered occupations.
    exc_orb_idx = core_hole_index(
        mb_overlap[channel], fch_calc.mo_occ[channel], gs_orb_idx)

    # Occupied and unoccupied orbital indices for GS and FCH (excited channel)
    occ_idxs_gs, occ_idxs_fch, uno_idxs_fch = occ_unocc_indices(
        gs_calc.mo_occ[channel], fch_calc.mo_occ[channel], gs_orb_idx,
        core_hole_idx=exc_orb_idx)

    # Extract A/K matrices for the excited channel
    AMat, ADet, KMat, _ = build_A_K(mb_overlap[channel], occ_idxs_fch, occ_idxs_gs, uno_idxs_fch)

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
