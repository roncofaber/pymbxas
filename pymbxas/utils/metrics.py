#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:32:08 2024

@author: roncofaber
"""

import numpy as np
import sklearn.preprocessing as prep
from pymbxas.utils.auxiliary import as_list

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
    clusters = as_list(clusters)
    indexes = np.triu_indices(len(clusters[0]), k=1)
    return np.array([cc.get_all_distances()[indexes] for cc in clusters])

def get_relevant_distances(clusters, idxs):
    clusters = as_list(clusters)
    
    distances = []
    for cluster in clusters:
        dm = cluster.get_all_distances()
        distances.append(dm[idxs[:,0], idxs[:,1]])
    
    return np.array(distances)
    
    


