#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of functions to work with Boys and give us amazing orbitals.

Created on Thu Jul 13 15:55:09 2023

@author: roncoroni
"""

import numpy as np
# import cupy as cp

#%%

def find_1s_orbitals_pyscf(molecule, coefficients, energies, occupations,
                           ato_idxs, check_deg=True):

    # occupied orbitals only
    occ_idxs = np.where(occupations == 1)[0]

    # orbital labels
    ao_labels = np.array(molecule.ao_labels(fmt=False))

    orb_list = []
    for to_excite in ato_idxs:

        symbol = molecule.atom_symbol(to_excite)

        for cc, orb_idx in enumerate(occ_idxs):

            orb = coefficients.T[orb_idx]
            # square coeff
            orb2 = orb**2

            # find indexes where orbital has weight
            rel_idxs = np.where(orb2 > 0.3*np.max(orb2))[0]

            # if isinstance(rel_idxs, cp.ndarray):
            #     rel_idxs = rel_idxs.get()

            rel_labels = ao_labels[rel_idxs]
            # there is weight on our orbital of interest
            if any([all(lab == (to_excite, symbol, "1s", "")) for lab in rel_labels]):

                # if not in list add it and all the orbitals with similar energy
                if orb_idx in orb_list:
                    continue

                if check_deg:
                    degenerate_mask = np.abs(energies[occ_idxs] - energies[orb_idx]) < 1e-1
                    degenerate_orbs = occ_idxs[degenerate_mask]
                    orb_list.extend(degenerate_orbs.tolist())
                else:
                    orb_list.append(orb_idx)

    return sorted(list(set(orb_list)))