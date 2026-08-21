#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:03:09 2023

@author: roncoroni
"""

import numpy as np

#%%

#TODO check
def gaussian_broadening(x, sigma):
    # return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    return np.exp(-0.5*(x/sigma)**2)

def broadened_spectrum(egrid, energies, intensities, sigma):
    x_shifted = egrid[:, np.newaxis] - energies  # (npoints, M)
    gauss = gaussian_broadening(x_shifted, sigma)  # (npoints, M)
    if intensities.ndim == 1:
        # intensities: (M,) -> spectra: (npoints,)
        return np.sum(intensities * gauss, axis=1)
    else:
        # intensities: (naxes, M) -> spectra: (naxes, npoints)
        return np.sum(intensities[:, np.newaxis, :] * gauss[np.newaxis, :, :], axis=2)

def get_mbxas_spectra(energies, intensities, sigma=0.5, npoints=3001, tol=0.01, erange=None):
    """
    Generate MBXAS spectra with Gaussian broadening.

    Parameters:
    energies (array): Array of energy values (eV).
    intensities (array): Array of intensity values corresponding to the energies.
        Can be 1D (N,) or 2D (naxes, N).
    sigma (float): Standard deviation for Gaussian broadening (default is 0.5 eV).
    npoints (int): Number of points in the resulting spectra (default is 3001).
    tol (float): Tolerance for extending the energy range (default is 0.01).
    erange (list or array, optional): Energy range [min_E, max_E] or array of energy values.

    Returns:
    tuple: Tuple containing:
        - energy (array): Array of energy values for the spectra.
        - spectra (array): Broadened spectra, shape (npoints,) or (naxes, npoints).
    """

    # Determine the energy range for the spectra
    if erange is not None:
        if len(erange) == 2:
            min_E, max_E = np.array(erange)
            egrid = np.linspace(min_E, max_E, npoints)
        else:
            egrid = np.array(erange)
            min_E = egrid.min()
            max_E = egrid.max()
    else:
        min_E = np.min(energies)
        max_E = np.max(energies)
        dE = max_E - min_E
        egrid = np.linspace(min_E - tol * dE, max_E + tol * dE, npoints)

    # Define relevant indexes for energies within the range
    rel_idxs = (energies > min_E - 5 * sigma) & (energies < max_E + 5 * sigma)

    # Index intensities along the transitions axis (last axis)
    intensities = np.asarray(intensities)
    if intensities.ndim == 1:
        intensities_filtered = intensities[rel_idxs]
    else:
        intensities_filtered = intensities[:, rel_idxs]

    # Generate the broadened spectra
    spectra = broadened_spectrum(egrid, energies[rel_idxs], intensities_filtered, sigma)

    return egrid, spectra
