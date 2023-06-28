#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:01:06 2023

@author: roncofaber
"""

import os

from pyqchem import Structure, QchemInput
from pyqchem.qc_input import CustomSection

from pymbxas.utils.check_keywords import determine_occupation

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
    molecule_str_input = QchemInput(
            molecule_str,
            **qchem_params,
            )
    
    # write input in target path (append mode)
    with open(run_path + "qchem.input", write_mode) as fout:
        
        if write_mode == "a":
            fout.write("\n@@@\n\n")
        
        fout.write(molecule_str_input.get_txt())
        
    return