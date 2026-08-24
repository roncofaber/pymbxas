#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:03:09 2023

@author: roncoroni
"""

import numpy as np

#%%

def gaussian_broadening(x, sigma):
    return np.exp(-0.5*(x/sigma)**2) / (sigma * np.sqrt(2 * np.pi))

def broadened_spectrum(egrid, energies, intensities, sigma, nsigma=6):
    """Sum of Gaussians of width sigma, one per (energy, intensity) stick,
    sampled on egrid. Each stick's support is truncated at nsigma*sigma
    (mbxas-qe's spec.f90:stick_to_spec uses the same nsigma=6 truncation)
    and scatter-added onto egrid, rather than densely broadcasting every
    stick against every grid point: for shake-up stick combinatorics
    (e.g. cross-channel order-2 on a many-orbital real molecule, which
    can reach millions of sticks) a dense (npoints, M) array is tens to
    hundreds of GB, while the truncated form is (M, W) with W independent
    of npoints. Requires egrid to be uniformly spaced; falls back to the
    dense form otherwise (only the small, non-shake-up main-spectrum path
    calls this with a non-uniform egrid).
    """
    egrid = np.asarray(egrid)
    energies = np.asarray(energies)
    intensities = np.asarray(intensities)
    is_1d = intensities.ndim == 1
    npoints, M = egrid.shape[0], energies.shape[0]

    out = np.zeros(npoints) if is_1d else np.zeros((intensities.shape[0], npoints))
    if M == 0:
        return out

    de = egrid[1] - egrid[0] if npoints > 1 else 0.0
    uniform = npoints > 1 and np.allclose(np.diff(egrid), de, rtol=1e-8, atol=1e-12)
    if not uniform:
        x_shifted = egrid[:, np.newaxis] - energies  # (npoints, M)
        gauss = gaussian_broadening(x_shifted, sigma)  # (npoints, M)
        if is_1d:
            return gauss @ intensities
        return np.einsum("ac,pc->ap", intensities, gauss)

    half_window = max(1, int(np.ceil(nsigma * sigma / abs(de))))
    offsets = np.arange(-half_window, half_window + 1)  # (W,)

    i0 = np.round((energies - egrid[0]) / de).astype(np.int64)
    idx = i0[:, None] + offsets[None, :]  # (M, W)
    valid = (idx >= 0) & (idx < npoints)
    idx = np.clip(idx, 0, npoints - 1)

    gauss = gaussian_broadening(egrid[idx] - energies[:, None], sigma)  # (M, W)
    gauss = np.where(valid, gauss, 0.0)
    idx_flat = idx.ravel()

    if is_1d:
        contrib = (intensities[:, None] * gauss).ravel()
        return np.bincount(idx_flat, weights=contrib, minlength=npoints)

    for a in range(intensities.shape[0]):
        contrib = (intensities[a][:, None] * gauss).ravel()
        out[a] = np.bincount(idx_flat, weights=contrib, minlength=npoints)
    return out

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
