#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of functions to work with Boys and give us amazing orbitals.

Created on Thu Jul 13 15:55:09 2023

@author: roncoroni
"""

import numpy as np
# import cupy as cp

from .basis import get_basis_set_info
#%%

# function to generate a list of 1s orbitals for a specific channel
def find_1s_in_channel(boys_coeff, atom_coeffs, atom_labels, symbols):
    symbol_list = np.concatenate([[cc]*len(atom_coeffs[cc]) for cc in range(len(symbols))])
    labels_list = np.concatenate(atom_labels)
    dominant_atoms = np.argmax(np.abs(boys_coeff), axis=1)

    orb_types   = labels_list[dominant_atoms]
    orb_symbols = symbol_list[dominant_atoms]

    found_elements = []
    s1_list = [None]*len(symbols)
    for idx, (orb_typ, orb_id) in enumerate(zip(orb_types, orb_symbols)):

        if orb_typ == "s" and orb_id not in found_elements:
            found_elements.append(orb_id)
            s1_list[orb_id] = idx
    return s1_list

# find the 1s orbitals of the system
def find_1s_orbitals(gs_data, use_localized=False):

    # get basis set information
    atom_coeffs, atom_labels, symbols, nbasis, indexing = get_basis_set_info(gs_data['basis'])

    s_orbitals   = {}
    # iterate over channels and find 1s orbitals
    for channel in ["alpha", "beta"]:

        if use_localized:
            orb_coeff = np.array(gs_data["localized_coefficients"][channel])
        else:
            orb_coeff = np.array(gs_data["coefficients"][channel])

        # calculate which ones are the 1s orbitals
        s1_list = find_1s_in_channel(orb_coeff, atom_coeffs, atom_labels, symbols)
        s_orbitals[channel] = s1_list

    return s_orbitals

# calculate the boys overlap
def calculate_boys_overlap(boys_coeffs, fch_coeffs, basis_overlap):

    boys_overlap = {}
    # iterate over channels and calculate overlap
    for channel in ["alpha", "beta"]:

        boys_coeff = np.array(boys_coeffs[channel])
        exci_coeff = np.array(fch_coeffs[channel])

        # calculate boys overlap factor
        OVLP = np.linalg.multi_dot([boys_coeff, basis_overlap, exci_coeff.T])
        boys_overlap[channel] = OVLP

    return boys_overlap


def find_1s_orbitals_pyscf(molecule, coefficients, energies, occupations,
                           ato_idxs, check_deg=True):

    # occupied orbitals only
    occ_idxs = np.where(occupations == 1)[0]
    
    # orbital labels
    ao_labels = np.array(molecule.ao_labels(fmt=False))

    orb_list = []
    for to_excite in ato_idxs:
        
        symbol = molecule.atom_symbol(to_excite)
    
        for cc, orb in enumerate(coefficients.T[occ_idxs]):
    
            # square coeff
            orb2 = orb**2
    
            # find indexes where orbital has weight
            rel_idxs = np.where(orb2 > 0.3*np.max(orb2))[0]
            
            # if isinstance(rel_idxs, cp.ndarray):
            #     rel_idxs = rel_idxs.get()
    
            rel_labels = ao_labels[rel_idxs]
            # there is weight on our orbital of interest
            if any([all(lab == (to_excite, symbol, "1s", "")) for lab in rel_labels]):
    
                # if not in list add it and all the orbitals with similar energy
                if cc in orb_list:
                    continue
    
                if check_deg:
                    degenerate_orbs = np.where(np.abs(energies - energies[cc]) < 1e-1)[0].tolist()
                    orb_list.extend(degenerate_orbs)
                else:
                    orb_list.append(cc)

    return orb_list