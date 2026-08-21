#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Series of functions to help fixing keywords of PyMBXAS dict.

Created on Mon Jun 26 11:44:43 2023

@author: roncofaber
"""

#%%

#TODO expand this to work with mixed PBCs, works only for full pbc at the moment
def check_pbc(pbc, structure):   
    if pbc is None:
        return all(structure.get_pbc())
    else:
        assert(isinstance(pbc, bool))
        return pbc