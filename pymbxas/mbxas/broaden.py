#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:03:09 2023

@author: roncoroni
"""

import numpy as np
from ase import units
Ha = units.Ha
#%%

def gaussian_broadening(x, sigma):
    return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def broadened_spectrum(x, energies, intensities, sigma):

    x_shifted = x[:, np.newaxis] - energies  # Efficiently subtract energies from each x value

    return np.sum(intensities * gaussian_broadening(x_shifted, sigma)[:, np.newaxis], axis=2)


# function to get MBXAS spectra
def get_mbxas_spectra(energies, intensities, sigma=0.005, npoints=3001, tol=0.01,
                      erange=None, isotropic=False):
    
    if erange is not None:
        min_E, max_E = np.array(erange)/Ha
        tol = 0
    else:
        min_E = np.min(energies)
        max_E = np.max(energies)

    dE = max_E - min_E
    
    energy = np.linspace(min_E - tol*dE, max_E + tol*dE, npoints)
    rel_idxs = (energies > min_E-dE) & (energies < max_E+dE)
    
    if isotropic:
        spectras = broadened_spectrum(energy, energies[rel_idxs],
                                         intensities[rel_idxs], sigma)  # Vectorized calculation
        
    # else:
    if not isotropic:
        spectras = broadened_spectrum(energy, energies[rel_idxs],
                                     intensities[:,rel_idxs]**2,
                                     sigma)  # Vectorized calculation
    
    return Ha*energy, spectras.T
