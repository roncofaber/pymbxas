#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:01:06 2023

@author: roncofaber
"""

import os
import pymbxas.build.input as inp

#%%

def write_mbxas_input(mbxas_parameters = {}, run_path = "."):

    # define MBXAS input file
    mbxas_f = run_path + "/INP_FILE"

    # copy default params and update with input ones
    mbxas_params = {
        "gridP"    : 100,
        "highE"    : 1127.5,
        "lowE"     : 20,
        "sigma"    : 0.3,
        "do_align" : False,
        "DoRIXS"   : False,

        "Gamma"         : "0+0.3j",
        "check_amp"     : True,
        "printsticks"   : True,
        "printspec"     : True,
        "Dodebug"       : False,
        "calc_abs"      : True,
        "printinteg"    : False,
        "input_file"    : "{}/qchem.input".format(run_path),
        "output_file"   : "{}/qchem.output".format(run_path),
        "printanalysis" : True
        }

    mbxas_params.update(mbxas_parameters)

    # write input file
    with open(mbxas_f, "w") as fout:
        for key, value in mbxas_params.items():
              fout.write("{} = {}\n".format(key, value))

    return


def write_qchem_job(molecule, charge, multiplicity,
                    qchem_params, run_path, occupation = None,
                    from_scratch = True):

    # make dir if not existent
    if not os.path.isdir(run_path):
        os.mkdir(run_path)

    # if from scratch write new file, otherwise append
    if from_scratch:
        write_mode = "w"
    else:
        write_mode = "a"

    # generate qchem input
    qchem_input = inp.make_qchem_input(molecule, charge,
                                       multiplicity, qchem_params,
                                       occupation=occupation)



    # write input in target path (append mode)
    with open(run_path + "qchem.input", write_mode) as fout:

        if write_mode == "a":
            fout.write("\n@@@\n\n")

        fout.write(qchem_input.get_txt())

    return