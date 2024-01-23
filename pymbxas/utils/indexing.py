#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:04:46 2023

@author: roncoroni
"""

import numpy as np

#%%


# return copy of input as list if not one
def as_list(inp):
    return [inp] if not isinstance(inp, list) else inp.copy()

# return list of indexes from mixed input of indexes and string (elements)
def atoms_to_indexes(system, symbols):

    # check if symbols is a list of strings
    if symbols == "all":
        return list(range(len(system.get_chemical_symbols())))

    symbols = as_list(symbols)

    indexes = []
    for symbol in symbols:
        if not isinstance(symbol, str):
            indexes.append(symbol)
        else:
            for cc, atom in enumerate(system.get_chemical_symbols()):
                if atom == symbol:
                    indexes.append(cc)
    return np.unique(indexes).tolist()