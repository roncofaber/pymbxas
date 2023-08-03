#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:03:09 2023

@author: roncoroni
"""

import numpy as np

#%%

# broaden spectrum to plot it
def broadened_spectrum(x, energies, intensities, sigma):

    intensities = np.asarray(intensities)

    def gaussian_broadening(x, sigma):
        return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    broadened_spec = np.zeros_like(x)
    for energy, intensity in zip(energies, intensities): #square intensity
        broadened_spec += intensity * gaussian_broadening(x - energy, sigma)

    return broadened_spec

# function to get MBXAS spectra
def get_mbxas_spectra(energies, intensities, sigma=0.3, npoints=3001, tol=0.01):

    Ha = 27.211407953

    min_E = np.min(energies)
    max_E = np.max(energies)

    dE = max_E - min_E

    energy = np.linspace(min_E - tol*dE, max_E + tol*dE, npoints)

    spectras = []
    for intensity in intensities:
        spectra = broadened_spectrum(energy, energies,
                                          intensity**2,
                                          sigma)
        spectras.append(spectra)

    return Ha*energy, spectras