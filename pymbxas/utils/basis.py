#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to read basis set and get info

Created on Thu Jul 13 15:57:59 2023

@author: roncoroni
"""

import numpy as np

#%%

# get info from basis set
def get_basis_set_info(basis):
    atom_coeffs = []
    atom_labels = []
    symbols     = []
    nbasis = 0
    for cc, atom in enumerate(basis['atoms']):
        istart = nbasis
        atom_label = []
        for shell in atom['shells']:
            nbasis += shell['functions']
            atom_label.extend(shell['functions']*[shell['shell_type']])

        atom_coeffs.append(np.array(range(istart, nbasis)))
        atom_labels.append(np.array(atom_label))
        symbols.append(atom["symbol"])

    return atom_coeffs, atom_labels, symbols, nbasis