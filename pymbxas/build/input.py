#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:08:26 2023

@author: roncoroni
"""

from pyqchem import Structure, QchemInput
from pyqchem.qc_input import CustomSection
from pymbxas.utils.check_keywords import determine_occupation

#%%

def make_qchem_input(molecule, charge, multiplicity,
                     qchem_params, occupation = None):

    # make molecule
    molecule_str = Structure(
        coordinates  = molecule.get_positions(),
        symbols      = molecule.get_chemical_symbols(),
        charge       = charge,
        multiplicity = multiplicity)

    # check occupation if needed

    if occupation is not None:

        # check occupation format
        occupation = determine_occupation(occupation)
        occ_section = CustomSection(title='occupied',
                                    keywords={' ' : occupation})

        if isinstance(qchem_params["extra_sections"], list):
            qchem_params["extra_sections"].append(occ_section)
        else:
            qchem_params["extra_sections"] = occ_section

    # generate input
    qchem_input = QchemInput(
            molecule_str,
            **qchem_params,
            )

    return qchem_input