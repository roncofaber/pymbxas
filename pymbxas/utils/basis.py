#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to work with basis sets

Created on Thu Jul 13 15:57:59 2023

@author: roncoroni
"""

import numpy as np

#%%

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