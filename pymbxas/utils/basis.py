#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to read basis set and get info

Created on Thu Jul 13 15:57:59 2023

@author: roncoroni
"""

import numpy as np

#%%

# function that finds the permutation needed to go from Qchem internal order
# to format of fchk files (IQmol compatible)
def calculate_shell_reordering(shell, start_idx):

    index_list = []

    if shell["shell_type"].startswith("d"):
        index_list = [ii + start_idx for ii in [4,2,0,1,3]]
    elif shell["shell_type"].startswith("f"):
        index_list = [ii + start_idx for ii in [6,4,2,0,1,3,5]]
    elif any([shell["shell_type"].startswith(ss) for ss in ["s", "p"]]):
        index_list = [ii + start_idx for ii in range(shell["functions"])]
    else:
        raise Exception("Sorry, shell type {} not recognized".format(
            shell["shell_type"]))

    return index_list


# get info from basis set
def get_basis_set_info(basis):
    atom_coeffs = []
    atom_labels = []
    symbols     = []
    nbasis = 0
    indexing    = []
    for cc, atom in enumerate(basis['atoms']):
        istart = nbasis
        atom_label = []
        for shell in atom['shells']:

            # store internal ordering
            indexing.extend(calculate_shell_reordering(shell, nbasis))

            # update number of basis
            nbasis += shell['functions']

            # store shell types
            atom_label.extend(shell['functions']*[shell['shell_type']])

        atom_coeffs.append(np.array(range(istart, nbasis)))
        atom_labels.append(np.array(atom_label))
        symbols.append(atom["symbol"])

    return atom_coeffs, atom_labels, symbols, nbasis, indexing

# calculate AO permutation from structure permutation
def get_AO_permutation(mol, permutation):
    
    # permute AO according to the permutation
    aoslices = mol.aoslice_by_atom()
    
    AO_permutation = []
    for idx in permutation:
        tslice = list(range(aoslices[idx][2], aoslices[idx][3]))
        
        AO_permutation.append(tslice)
    
    return np.concatenate(AO_permutation)

def get_l_val(mol):
    
    l_values = {
        "s" : 0,
        "p" : 1,
        "d" : 2,
        "f" : 3,
        "g" : 4,
        }
    return np.array([l_values[ii[2][-1]] for ii in mol.ao_labels(fmt=False)])