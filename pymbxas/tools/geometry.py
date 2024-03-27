#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:19:06 2024

@author: roncoroni
"""

import numpy as np
from pyscf import data

#%%

def get_center_of_mass(mol):
    """Calculates the center of mass of a molecule.

    Args:
        coordinates (np.ndarray): Array of atomic coordinates (shape: (N, 3))
        masses (np.ndarray): Array of atomic masses (shape: (N))

    Returns:
        np.ndarray: Center of mass coordinates (shape: (3,))
    """
    
    coordinates = mol.atom_coords()
    masses = np.array([data.elements.COMMON_ISOTOPE_MASSES[m] #* data.nist.AMU2AU
                       for m in mol.atom_charges()])
    
    return np.average(coordinates, axis=0, weights=masses)


def get_inertia_tensor(mol, vectors=False):
    """Calculates the inertia tensor of a molecule.
    """
    
    Itensor = mol.inertia_moment()
    
    # evals, evecs = np.linalg.eigh(Itensor)
    evals, evecs = np.linalg.eig(Itensor)
    
    if vectors:
        return evals, evecs.transpose()
    else:
        return evals


def get_angular_momentum(mol, velocities):
    """Calculates the angular momentum of a molecule.

    Args:
        coordinates (np.ndarray): Array of atomic coordinates (shape: (N, 3))
        velocities (np.ndarray): Array of atomic velocities (shape: (N, 3))
        com (np.ndarray): Center of mass coordinates (shape: (3,))
        masses (np.ndarray): Array of atomic masses (shape: (N))

    Returns:
        np.ndarray: Angular momentum vector (shape: (3,))
    """
    
    coordinates = mol.atom_coords()
    masses = np.array([data.elements.COMMON_ISOTOPE_MASSES[m] #* data.nist.AMU2AU
                       for m in mol.atom_charges()])
    com = get_center_of_mass(mol)
    
    coordinates -= com  # Center coordinates around COM
    
    return np.sum(np.cross(coordinates, masses[:, np.newaxis] * velocities), axis=0)