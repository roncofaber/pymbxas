#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:44:43 2023

@author: roncofaber
"""

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
        for channel in ["alpha", "beta"]:

            indexes = [str(x) for x in range(1, occupation["nelectrons"]+1)
                       if not (x == occupation["eject"]
                               and channel == occupation["channel"])]

            if occupation["inject"] and channel == occupation["channel"]:
                indexes.append(str(occupation["inject"]))

            occ_string += " ".join(indexes)

            occ_string += "\n  "

        return occ_string