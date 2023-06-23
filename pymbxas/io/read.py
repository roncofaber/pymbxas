#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:32:15 2023

@author: roncofaber
"""

import os
from pyqchem.parsers.basic import basic_parser_qchem
from pyqchem.parsers.parser_optimization import basic_optimization
#%%

#Read outputs
def read_qchem_job(output, run_path, jobtype):
    
    if not os.path.isdir(run_path):
        print("Doesnt exist!")
    
    file_to_read = "{}/{}".format(run_path, output)
    
    print(file_to_read)
    with open(file_to_read, "r") as fin:
        output = fin.read()
        
    if jobtype == "sp":
        parsed_data = basic_parser_qchem(output)
        
    elif jobtype == "opt":
        parsed_data = basic_optimization(output)
        
    return output, parsed_data