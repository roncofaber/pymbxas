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

class EmptyScaler:
    def fit(self, X, y=None):
        # No fitting necessary, just return self
        
        Xmax = np.max(X)
        
        self._Xmax = Xmax
        
        return self

    def transform(self, X):
        # Return the data as is
        return X/self._Xmax

    def fit_transform(self, X, y=None):
        # Fit and transform the data (no change)
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        # Return the data as is
        return self._Xmax*X
    
    @property
    def var_(self):
        return self._Xmax
    
    @property
    def mean_(self):
        assert False, "YOU SHOULD NOT BE HERE EMPTYSCALRE"
        return None
    
# generate a scaler
def generate_scaler(scaler):
    
    # match scaler:
        
    if scaler in ["none", None, "None"]:
        return EmptyScaler()
    
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
    
    distances = []
    for cluster in as_list(clusters):
        dm = cluster.get_all_distances()
        distances.append(dm[idxs[:,0], idxs[:,1]])
    
    return np.array(distances)
    
    


