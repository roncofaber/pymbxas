#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:08:26 2023

@author: roncoroni
"""
import copy
import collections

from pyqchem import Structure, QchemInput
from pyqchem.qc_input import CustomSection
from pymbxas.utils.check_keywords import determine_occupation

#%%

gs_def_params = {
    "extra_rem_keywords" : {"TRANS_MOM_SAVE" : True,
                            # "BOYSCALC"       : "2",
                            },
    }

xas_def_params = {
    "extra_rem_keywords" : {
        "TRANS_MOM_READ" : True,
        'MOM_METHOD'     : 'MOM', # or IMOM?
        "SCF_GUESS"      : 'read',
        "MOM_START"      : 1 ,
        "use_libqints"   : 1
    },
    }

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def update_input(qchem_params, extra_input, calc_type, use_boys=False,
                 scf_guess=None, occupation=None):

    # copy input
    qchem_params = copy.deepcopy(qchem_params)
    extra_input  = copy.deepcopy(extra_input)

    # remove mom_type #TODO ugly and temporary
    if "mom_type" in qchem_params:

        mom_type = qchem_params["mom_type"]
        del qchem_params["mom_type"]

        if calc_type == "gs":
            pass
        else:
            if mom_type == 0: # 0 --> all MOM
                extra_input["extra_rem_keywords"]["MOM_METHOD"] = "MOM"
            if mom_type == 1: # 1 --> all IMOM
                extra_input["extra_rem_keywords"]["MOM_METHOD"] = "IMOM"

    # update with default params
    qchem_params = update(qchem_params, extra_input)

    if use_boys:
        qchem_params["extra_rem_keywords"]["BOYSCALC"] = 2
    else:
        qchem_params["extra_rem_keywords"]["BOYSCALC"] = 0

    if scf_guess is not None:
        qchem_params["scf_guess"] = scf_guess

    # add extra sections if nonexistend #TODO can be cleaner code
    if "extra_sections" not in qchem_params:
        qchem_params["extra_sections"] = None

    # check occupation if needed
    if occupation is not None:
        # check occupation format
        occ_section = CustomSection(title='occupied',
                                    keywords={' ' : determine_occupation(occupation)})

        if isinstance(qchem_params["extra_sections"], list):
            qchem_params["extra_sections"].append(occ_section)
        else:
            qchem_params["extra_sections"] = occ_section

    return qchem_params

# Function to generate an input for a QCHEM calculation,
# specify calc type to add default params needed for MBXAS
def make_qchem_input(molecule, charge, multiplicity,
                     qchem_params, calc_type, occupation = None,
                     scf_guess = None, use_boys=False):

    # make molecule
    molecule_str = Structure(
        coordinates  = molecule.get_positions(),
        symbols      = molecule.get_chemical_symbols(),
        charge       = charge,
        multiplicity = multiplicity)

    # update params depending on calculation
    if calc_type == "gs":
        qchem_params = update_input(qchem_params, gs_def_params, calc_type,
                                    use_boys, scf_guess, occupation)
    elif calc_type in ["fch", "xch"]:
        qchem_params = update_input(qchem_params, xas_def_params, calc_type,
                                    use_boys, scf_guess, occupation)

    # generate input
    qchem_input = QchemInput(
            molecule_str,
            **qchem_params,
            )

    return qchem_input