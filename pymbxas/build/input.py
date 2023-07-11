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

gs_def_params = {
    "extra_rem_keywords" : {"TRANS_MOM_SAVE" : True},
    }

xas_def_params = {
    "extra_rem_keywords" : {
        "TRANS_MOM_READ" : True,
        'MOM_METHOD'     : 'IMOM',
        "SCF_GUESS"      : 'read',
        "MOM_START"      : 1 ,
        "use_libqints"   : 1
    },
    }


#%%


def make_qchem_input(molecule, charge, multiplicity,
                     qchem_params, calc_type, occupation = None,
                     scf_guess = None):

    # make molecule
    molecule_str = Structure(
        coordinates  = molecule.get_positions(),
        symbols      = molecule.get_chemical_symbols(),
        charge       = charge,
        multiplicity = multiplicity)

    # copy parameters to not mess stuff
    qchem_params = qchem_params.copy()

    # update params depending on calculation
    if calc_type == "gs":
        qchem_params.update(gs_def_params)
    elif calc_type in ["fch", "xch"]:
        qchem_params.update(xas_def_params)
    qchem_params["scf_guess"] = scf_guess

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