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
    

def get_zmatlike_distances(structures):

    zmat = []    
    for structure in as_list(structures):
        zmat.append(zmatlike(structure))
        
    return np.array(zmat)

def zmatlike(atoms):
    """
    Generates a feature vector of distances from an ASE Atoms object (optimized).

    Args:
        atoms: An ase.Atoms object.

    Returns:
        A 1D numpy array containing the distances. Returns None if the structure
        is unsuitable for this representation.
    """
    num_atoms = len(atoms)
    if num_atoms < 4:
        print("Error: At least 4 atoms are required for this representation.")
        return None

    positions = atoms.get_positions()
    
    #Pre-allocate the array to store distances
    feature_vector = np.empty(3 + (num_atoms - 3) * 3)

    #Efficiently calculate distances for atoms 2 and 3
    feature_vector[0] = np.linalg.norm(positions[1] - positions[0])
    feature_vector[1] = np.linalg.norm(positions[2] - positions[0])
    feature_vector[2] = np.linalg.norm(positions[2] - positions[1])

    #Vectorized calculation for atoms 4 and beyond
    ref_positions = positions[:3]  #Positions of the first three atoms
    other_positions = positions[3:] #Positions of the remaining atoms

    #Calculate all pairwise distances between ref_positions and other_positions
    distances = np.linalg.norm(other_positions[:, np.newaxis, :] - ref_positions, axis=2)
    feature_vector[3:] = distances.flatten()

    return feature_vector

