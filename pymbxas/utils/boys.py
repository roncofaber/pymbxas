#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of functions to work with Boys and give us amazing orbitals.

Created on Thu Jul 13 15:55:09 2023

@author: roncoroni
"""

import numpy as np
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
def find_1s_orbitals(gs_data):

    # get basis set information
    atom_coeffs, atom_labels, symbols, nbasis = get_basis_set_info(gs_data['basis'])

    s_orbitals   = {}
    # iterate over channels and find 1s orbitals
    for channel in ["alpha", "beta"]:

        boys_coeff = np.array(gs_data["localized_coefficients"][channel])

        # calculate which ones are the 1s orbitals
        s1_list = find_1s_in_channel(boys_coeff, atom_coeffs, atom_labels, symbols)
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