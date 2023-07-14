#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Series of functions to help fixing keywords of PyMBXAS dict.

Created on Mon Jun 26 11:44:43 2023

@author: roncofaber
"""

#%%

# default excitation dict
def_exc_dict = {
    "channel" : "beta",
    "eject"   : False,
    "inject"  : False,
    }

def determine_excitation(excitation):

    exc_dict = def_exc_dict.copy()

    # if integer, it means User want to excite Nth atom
    if isinstance(excitation, int):
        exc_dict["ato_idx"] = excitation
    elif isinstance(excitation, dict):
        exc_dict.update(excitation)


    return exc_dict

# Function to transform an occ. dict to input ready for QCHEM
def determine_occupation(occupation):

    if isinstance(occupation, str):
        return occupation

    elif isinstance(occupation, dict):

        # check inject
        if "inject" not in occupation:
            occupation["inject"] = False

        # generate occupation string
        occ_string = ""
        for cc, channel in enumerate(["alpha", "beta"]):

            indexes = [str(x) for x in range(1, occupation["nelectrons"][cc]+1)
                       if not (x == occupation["eject"]
                               and channel == occupation["channel"])]

            if occupation["inject"] and channel == occupation["channel"]:
                indexes.append(str(occupation["inject"]))

            occ_string += " ".join(indexes)

            occ_string += "\n  "

        return occ_string