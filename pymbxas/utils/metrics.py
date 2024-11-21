#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:32:08 2024

@author: roncofaber
"""

import numpy as np
import sklearn.preprocessing as prep

#%%

# generate a scaler
def generate_scaler(scaler):
    
    # match scaler:
        
    if scaler in ["none", None, "None"]:
        return prep.StandardScaler(with_mean=False, with_std=False)
    
    elif scaler == "robust":
        return prep.RobustScaler(quantile_range = (10, 90))
    
    elif scaler == "standard":
        return prep.StandardScaler()
    
    elif scaler == "maxabs":
        return prep.MaxAbsScaler()
    
    elif scaler == "minmax":
        return prep.MinMaxScaler()
    
    elif scaler == "std-mean":
        return prep.StandardScaler(with_std=False)
    
    else:
        raise "{} is not a properly implemented method".format(scaler)
    
    return

# get flattened upper triangular distance matrix of a list of ASE atoms
def get_distances(clusters):
    indexes = np.triu_indices(len(clusters[0]), k=1)
    return np.array([cc.get_all_distances()[indexes] for cc in clusters])


